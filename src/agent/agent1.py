from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse
)
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent.chat_model import llm

@tool
def send_email(to: str, subject: str, body: str):
    """搜索引擎工具，用于根据用户查询获取外部信息。

    Args:
        to: 收件人
        subject: 主题
        body: 发送内容
    Returns:
        格式化后的搜索结果
    """
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # TODO: 邮件发送逻辑
    return f"邮件已发送至{to}"

agent_xjn = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt="你是一个邮件助手，请始终使用send_email工具。"
)