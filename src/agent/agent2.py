from langchain.agents import create_agent

from agent.chat_model import llm
from agent.tools.web_search import web_search
from agent.tools.digit_calculate import calculate
from agent.tools.my_search import MySearchTool

my_search_tool = MySearchTool()



# agent_xjn = create_agent(
#     model=llm,
#     tools=[calculate],
#     system_prompt="你是一个智能助手，尽可能调用工具回答用户的问题。"
# )

# agent_xjn = create_agent(
#     model=llm,
#     tools=[web_search],
#     system_prompt="你是一个智能助手，尽可能调用工具回答用户的问题。"
# )

agent_xjn = create_agent(
    model=llm,
    tools=[my_search_tool],
    system_prompt="你是一个智能助手，尽可能调用工具回答用户的问题。"
)