from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from agent.chat_model import llm

# 1. 提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，尽你所能回答所有问题。提供的聊天历史包含与你对话用户的相关信息。"),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    # ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

chain = prompt | llm  # 基础的执行链

# 2. 存储聊天记录：（内存，关系型数据库或者redis数据库）
store = {}  # 用来保存历史消息， key: 会话ID session_id

def get_session_history(session_id: str):
    """从内存中的历史消息列表中 返回当前会话 的所有历史消息"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# langchain所有消息类型：SystemMessage(系统提示词), HumanMessage(用户输入), AIMessage(AI回复消息), ToolMessage(工具返回消息)

# 3. 创建带历史记录功能的处理链
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)


result1 = chain_with_message_history.invoke({'input': '你好，我是谢俊男'}, config={"configurable": {"session_id": "user123"}})
print(result1)

result2 = chain_with_message_history.invoke({'input': '我的名字是什么？'}, config={"configurable": {"session_id": "user123"}})
print(result2)

result3 = chain_with_message_history.invoke({'input': '和我同名的人有多少？'}, config={"configurable": {"session_id": "user123"}})
print(result3)