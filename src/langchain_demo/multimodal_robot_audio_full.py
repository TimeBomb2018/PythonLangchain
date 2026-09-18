import asyncio
import base64
import copy
import io
import time
from threading import Thread
from typing import Dict, Optional

from PIL import Image
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from sqlalchemy import create_engine
import gradio as gr

from agent.chat_model import llm, llm_light
from chat.chat_audio_omni import chain_history

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


def get_last_user_after_assistant(history):
    """反向遍历找到最后一个assistant的位置，并返回后面的所有user消息"""
    if not history:
        return None
    if history[-1]['role'] == 'assistant':
        return None

    last_assistant_idx = -1
    for i in range(len(history) -1, -1, -1):
        if history[i]['role'] == 'assistant':
            last_assistant_idx = i
            break

    if last_assistant_idx == -1:
        return history
    else:
        return history[last_assistant_idx+1:]

def transcribe_audio(audio_path):
    """使用Base64处理语音转换"""
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        audio_message = {
            'type': 'input_audio',
            'input_audio': {
                "data": f"data:;base64,{audio_data}",
                "format": "wav"
            }
        }
        return audio_message
    except Exception as e:
        print(e)
        return {}

def transcribe_image(image_path):
    """使用Base64处理图片转换"""
    with Image.open(image_path) as img:
        img_format = img.format if img.format else 'JPEG'
        buffered = io.BytesIO()
        # 保留原始格式（避免JPEG强制转换导致透明通道丢失）
        img.save(buffered, format=img_format)

        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            'type': 'image_url',
            'image_url': {"url": f"data:image/{img_format};base64,{image_data}"}
        }

# web界面中的核心函数
def add_message(history, messages):
    """将用户的消息添加到聊天记录中"""
    for m in messages['files']:
        print(m)
        history.append({'role': 'user', 'content': {'path': m}})
    if messages['text'] is not None:
        history.append({'role': 'user', 'content': messages['text']})
    return history, gr.MultimodalTextbox(value=None, interactive=False)  # 返回更新后的历史和重置的输入框

def submit_messages(history):
    """提交用户输入的消息，生成机器人回复"""
    user_messages = get_last_user_after_assistant(history)
    print(user_messages)
    content = []
    if user_messages:
        for x in user_messages:
            file_message = {}
            if isinstance(x['content'], str): # 文字输入消息
                content.append({'type': 'text', 'text': x['content']})
            elif isinstance(x['content'], tuple): # 多媒体输入消息
                file_path = x['content'][0]
                if file_path.endswith('.wav'):
                    file_message = transcribe_audio(file_path)
                elif file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('jpeg'):
                    file_message = transcribe_image(file_path)
                content.append(file_message)
            else:
                pass
    input_message = HumanMessage(content=content)
    resp = chain_history.invoke({'message': input_message}, config={'configurable': {"session_id": 'user_omni001'}})
    history.append({'role': 'assistant', 'content': resp.content})
    return history



# 开发一个聊天机器人的Web界面
with gr.Blocks(title='多模态聊天机器人', fill_height=True, theme=gr.themes.Soft()) as block:
    gr.Markdown("🤖 多模态聊天机器人", elem_classes="text-center")
    # 聊天历史记录的组件
    chatbot = gr.Chatbot(
        type='messages',
        height=500,
        label='聊天记录',
        avatar_images=(
            "/Users/xiejunnan/测试/langchain_test/image/user.png",
            "/Users/xiejunnan/测试/langchain_test/image/ai.png"
        ),
        elem_id="chatbot-container"
    )
    # 创建多模态输入框
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", ".wav", '.mp4'],
        file_count='multiple',
        placeholder="请输入消息或上传图片/音频...",
        show_label=False,
        sources=["upload", "microphone"]
    )

    chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input]).then(
        submit_messages, [chatbot], [chatbot]
    ).then(
        lambda: gr.MultimodalTextbox(interactive=True),
        None,
        [chat_input]
    )


if __name__ == '__main__':
    block.launch(
        server_name="127.0.0.1",
        server_port=7860
    )