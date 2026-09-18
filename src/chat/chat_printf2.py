from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.chat_model import llm

prompt = ChatPromptTemplate.from_template(
    "尽你所能用中文回答用户的问题。"
    "你必须始终输出一个包含\"title\", \"year\", \"director\", \"rating\"键的JSON对象。其中\"title\"代表：电影标题；\"year\"代表：电影发行年份；\"director\"代表：电影导演；\"rating\"代表：电影评分（满分10分）"
    "{question}"
)

chain = prompt | llm | SimpleJsonOutputParser()   # 管道左边的输出作为管道右边的输入
resp = chain.invoke({"question": "提供电影《盗梦空间》的详细信息"})
print(resp)
