from pydantic import BaseModel, Field

from agent.chat_model import llm


class Movie(BaseModel):
    """电影详情"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="电影发行年份")
    director: str = Field(..., description="电影导演")
    rating: float = Field(..., description="电影评分（满分10分）")

# model_with_structure = llm.with_structured_output(Movie, include_raw=True)
model_with_structure = llm.with_structured_output(Movie)
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
print(response)
