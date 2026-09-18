from typing import Type, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from agent.chat_model import zhipuai_client


class SearchArgs(BaseModel):
    query: str = Field(..., description="需要进行网络搜索的信息。")


# 网络搜索的工具
class MySearchTool(BaseTool):
    name: str = "search_tool"

    description: str = "搜索互联网上公开内容的工具"

    # 第一种写法
    args_schema: Type[BaseTool] = SearchArgs

    # 第二种写法
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.args_schema = create_model("SearchInput", query=(str, Field(..., description="需要进行网络搜索的信息。")))

    def _run(self, query: str) -> str:
        try:
            response = zhipuai_client.web_search.web_search(
                search_engine="search_pro",
                search_query=query
            )
            if response.search_result:
                return "\n\n".join([d.content for d in response.search_result])
            return "没有搜索到任何结果"
        except Exception as e:
            print(e)
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query=query)
