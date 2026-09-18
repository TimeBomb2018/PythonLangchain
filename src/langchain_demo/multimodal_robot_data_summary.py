from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from sqlalchemy import create_engine

from agent.chat_model import llm

# 1. 提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),  # 动态注入系统提示词
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ("human", "{input}")
])

chain = prompt | llm  # 基础的执行链

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

def summarize_messages(current_input):
    """剪辑和摘要上下文，历史记录"""
    session_id = current_input["config"]["configurable"]["session_id"]
    if not session_id:
        raise ValueError("必须通过config参数提供session_id")

    # 获取当前会话ID的所有历史聊天记录
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages

    if len(stored_messages) <= 2:  # 保留最近2条消息的阈值
        return {
            "original_messages": stored_messages,  # 不满足摘要条件时返回原始消息
            "summary": None
        }

    # 剪辑消息列表
    last_two_messages = stored_messages[-2:]
    messages_to_summarize = stored_messages[:-2]

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话历史压缩为一条保留关键信息的摘要信息。"),
        ("placeholder", "{chat_history}"),
        ("human", "请生成包含上述对话核心内容的摘要，保留重要事实和决策。")
    ])

    summarization_chain = summarization_prompt | llm

    # 生成摘要（AIMessage）
    summary_message = summarization_chain.invoke({'chat_history': messages_to_summarize})
    return {
        "original_messages": last_two_messages,  # 保留的原始消息
        "summary": summary_message   # 生成的摘要
    }


# 最终的链
# RunnablePassthrough默认会将输入数据原样传递到下游，而.assign()方法允许在保留原始输入的同时，通过指定键值对（如 messages_summarized=summarize_messages）向输入字典中添加新字段
final_chain = RunnablePassthrough.assign(messages_summarized=summarize_messages) | RunnablePassthrough.assign(
    input=lambda x: x['input'],
    chat_histroy=lambda x: x['messages_summarized']['original_messages'],
    system_message=lambda x: f"你是一个智能助手，尽你所能回答所有问题。摘要: {x['messages_summarized']['summary'].content}" if x['messages_summarized'].get('summary') else '无摘要'
) | chain_with_message_history


result = final_chain.invoke({'input': '用我的名字造一个句子', 'config': {"configurable": {"session_id": "user123"}}},
                            config={"configurable": {"session_id": "user123"}})
print(result)