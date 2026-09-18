import asyncio
import copy
import time
from threading import Thread
from typing import Dict, Optional

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from sqlalchemy import create_engine
import gradio as gr

from agent.chat_model import llm, llm_light, zhipuai_client

# 1. 提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),  # 动态注入系统提示词
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ("human", "{input}")
])

chain = prompt | llm  # 基础的执行链

# 内存缓存：存储各会话的摘要（不写入数据库）
session_summary_cache: Dict[str, Optional[str]] = {}

# 2. 存储聊天记录：（内存，关系型数据库或者redis数据库）
engine = create_engine('mysql+pymysql://root:193052xjn@localhost:3306/my_langchain_test?charset=utf8mb4')


def get_session_history(session_id: str):
    """从关系型数据库的历史消息列表中 返回当前会话 的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine
    )


# langchain所有消息类型：SystemMessage(系统提示词), HumanMessage(用户输入), AIMessage(AI回复消息), ToolMessage(工具返回消息)

# 3. 创建带历史记录功能的处理链
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)


# 4. 剪辑和摘要上下文，历史记录。 ： 保留最近的前2条消息，把之前的所有消息形成摘要
async def async_summarize_messages(session_id: str):
    """剪辑和摘要上下文，历史记录"""
    try:
        # 获取当前会话ID的所有历史聊天记录
        chat_history = get_session_history(session_id)
        stored_messages = chat_history.messages

        if len(stored_messages) <= 2:  # 保留最近2条消息的阈值
            session_summary_cache[session_id] = None  # 无须摘要
            return

        # 剪辑消息列表
        messages_to_summarize = stored_messages[:-2]

        summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", "请将以下对话历史压缩为一条保留关键信息的摘要信息。"),
            ("placeholder", "{chat_history}"),
            ("human", "请生成包含上述对话核心内容的摘要，保留重要事实和决策。")
        ])

        summarization_chain = summarization_prompt | llm_light

        # 生成摘要（AIMessage）
        summary_ai_msg = await summarization_chain.ainvoke({'chat_history': messages_to_summarize})
        session_summary_cache[session_id] = summary_ai_msg.content
        return
    except Exception as e:
        session_summary_cache[session_id] = None
        return


# 启动一步摘要的线程（非阻塞）
def start_async_summary(session_id: str):
    """启动线程执行异步摘要，不阻塞主流程"""
    def run_async():
        asyncio.run(async_summarize_messages(session_id))
    Thread(target=run_async, daemon=True).start()


# 轻量级的前置处理函数（读取内存摘要，不修改数据库）
def prepare_chat_context(current_input):
    """仅判断是否需要触发摘要逻辑，快速返回"""
    session_id = current_input["config"]["configurable"]["session_id"]
    if not session_id:
        raise ValueError("必须通过config参数提供session_id")

    user_input = current_input["input"]

    # 读取数据库历史（仅读）
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages

    # 读取内存中的摘要（无则为None），最近两条历史记录
    cached_summary = session_summary_cache.get(session_id, None)

    user_need_long = any([
        "详细" in user_input,
        "完整" in user_input,
        "展开" in user_input,
        "长一点" in user_input,
        "全部" in user_input,
        "具体" in user_input,
        "解释" in user_input,
    ])

    # 确定系统提示词（用内存摘要，不修改数据库）
    if user_need_long:
        # 用户要长文 → 不限制长度
        system_message = f"你是智能助手，详细完整回答问题。历史摘要：{cached_summary}" if cached_summary else "你是智能助手，详细完整回答问题。"
    else:
        # 默认 → 限制长度
        system_message = f"你是智能助手，回答简洁，控制在200字内。历史摘要：{cached_summary}" if cached_summary else "你是智能助手，回答简洁，控制在200字内。"

    return {
        "chat_history": stored_messages if len(stored_messages) <= 2 else stored_messages[-2:],
        "system_message": system_message
    }


# 最终的链
# RunnablePassthrough默认会将输入数据原样传递到下游，而.assign()方法允许在保留原始输入的同时，通过指定键值对（如 messages_summarized=summarize_messages）向输入字典中添加新字段
final_chain = RunnablePassthrough.assign(context=prepare_chat_context) | RunnablePassthrough.assign(
    input=lambda x: x['input'],
    chat_histroy=lambda x: x['context']['chat_history'],
    system_message=lambda x: x['context']['system_message']
) | chain_with_message_history


# web界面中的核心函数
def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({'role': 'user', 'content': [{'type': 'text', 'text': user_message.strip()}]})
    return chat_history, ''

def execute_chain(chat_history):
    if not chat_history or chat_history[-1].get('role') != 'user':
        yield copy.deepcopy(chat_history)
        return
    input = chat_history[-1].get('content', '')
    # 核心：解析Gradio嵌套格式为纯字符串
    if isinstance(input, list):
        # 处理Gradio 6.0+的 [{"type":"text", "text":"xxx"}] 格式
        text_parts = []
        for item in input:
            if item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        user_input_text = '\n'.join(text_parts).strip()
    else:
        # 兼容纯字符串格式
        user_input_text = str(input).strip()

    if not user_input_text:
        yield copy.deepcopy(chat_history)
        return

    session_id = "user004"
    config = {"configurable": {"session_id": "user004"}}
    full_response = ""

    try:
        chat_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ""}]
        })
        for chunk in final_chain.stream(input={'input': user_input_text, 'config': config}, config=config):
            # 拼接流式返回的内容（不同LLM的chunk格式可能略有差异）
            if hasattr(chunk, 'content'):
                full_response += chunk.content
            elif isinstance(chunk, str):
                full_response += chunk
            else:
                # 兜底：兼容不同格式的chunk
                full_response += str(chunk)

            # 实时更新聊天历史，yield返回给Gradio
            chat_history[-1]['content'] = [{'type': 'text', 'text': full_response}]  # 标准化AI回复格式
            yield copy.deepcopy(chat_history)  # 流式返回更新后的历史
            time.sleep(0.05)  # 可选：控制输出速度，避免刷屏
        start_async_summary(session_id)
    except Exception as e:
        # 异常处理：返回错误信息
        error_msg = f"执行失败：{str(e)}"
        chat_history[-1]["content"] = [{"type": "text", "text": error_msg}]
        yield copy.deepcopy(chat_history)
    return


def read_audio(audio_message):
    """读取音频文件"""
    if audio_message:
        with open(audio_message, "rb") as audio_file:
            resp = zhipuai_client.audio.transcriptions.create(
                model="glm-asr-2512",
                file=audio_file,
                stream=False
            )
            text = resp.model_extra['text'].strip()
            return text
    return ''


# 开发一个聊天机器人的Web界面
with gr.Blocks(title='多模态聊天机器人', theme=gr.themes.Soft()) as block:
    # 聊天历史记录的组件
    chatbot = gr.Chatbot(type='messages', height=500, label='聊天机器人')

    with gr.Row():
        # 文字输入的区域
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                placeholder='请给机器人发送消息...',
                label='文字输入',
                max_lines=5
            )
            submit_btn = gr.Button('发送', variant="primary")

        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=['microphone'],
                label='语音输入',
                type='filepath',
                format='wav'
            )

    # 文本框提交的事件
    submit_event = user_input.submit(add_message, [chatbot, user_input], [chatbot, user_input], queue=False
                                     ).then(execute_chain, [chatbot], [chatbot], queue=True)

    submit_btn.click(add_message, [chatbot, user_input], [chatbot, user_input], queue=False
                                     ).then(execute_chain, [chatbot], [chatbot], queue=True)

    # 语音输入框的改变事件
    audio_input.change(read_audio, [audio_input], [user_input])


if __name__ == '__main__':
    block.launch(
        server_name="127.0.0.1",
        server_port=7860
    )