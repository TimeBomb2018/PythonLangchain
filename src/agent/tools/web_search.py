from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.chat_model import zhipuai_client


# 方式一
@tool("web_search", parse_docstring=True)
def web_search(query: str) -> str:
    """互联网搜索的工具，可以搜索所有公开的信息。

    Args:
        query: 需要进行互联网查询的查询信息

    Returns:
        str: 返回搜索的结果信息
    """
    try:
        resp = zhipuai_client.web_search.web_search(
            search_engine='search_pro',
            search_query=query
        )
        if resp.search_result:
            return "\n\n".join([d.content for d in resp.search_result])
        return "没有搜索到任何结果"
    except Exception as e:
        print(e)
        return f"Error: {e}"

# # 方式二
# class SearchArgs(BaseModel):
#     query: str = Field(..., description="需要进行互联网查询的查询信息")
#
# @tool("web_search", args_schema=SearchArgs, description="互联网搜索的工具，可以搜索所有公开的信息")
# def web_search(query: str) -> str:
#     pass

if __name__ == '__main__':
    print(web_search.name) # 工具的名字
    print(web_search.description) # 工具的描述
    print(web_search.args) # 工具的参数
    print(web_search.args_schema.model_json_schema()) # 工具的参数的json schema(描述json字符串)

    result = web_search.invoke({'query': '如何使用langchain?'})
    print(result)