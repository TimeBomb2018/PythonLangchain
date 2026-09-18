import uuid

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from sqlalchemy import create_engine

from agent.chat_model import llm_omni

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个多模态AI助手，可以处理文本、音频和图像输入'),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm_omni

engine = create_engine('mysql+pymysql://root:193052xjn@localhost:3306/my_langchain_test?charset=utf8mb4')


def get_session_history(session_id: str):
    """从关系型数据库的历史消息列表中 返回当前会话 的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine
    )

chain_history = RunnableWithMessageHistory(
    chain,
    get_session_history
)
#
# config = {'configurable': {"session_id": str(uuid.uuid4())}}
#
# user_msg = HumanMessage(content='你好')
#
# resp = llm_omni.invoke([user_msg])
# print(resp)