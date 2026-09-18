from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI
from openai import OpenAI
from volcenginesdkarkruntime import Ark, AsyncArk

from agent.custom_chat_model import BailianCustomChatModel
from agent.env_utils import (
    SILICON_FLOW_API_KEY,
    SILICON_FLOW_BASE_URL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    ZHIPU_API_KEY,
    ZHIPUAI_BASE_URL,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)

# 调用硅基流动的千问模型
# llm = ChatOpenAI(
#     model="Qwen/Qwen2.5-7B-Instruct",
#     temperature=0.5,
#     api_key=SILICON_FLOW_API_KEY,
#     base_url=SILICON_FLOW_BASE_URL
# )

# qwen-plus-2025-12-01（特别快，但是快到期）
# qwen3.5-flash-2026-02-23
# 调用阿里百炼的千问模型
llm = ChatOpenAI(
    model="qwen3.5-flash-2026-02-23",
    temperature=0.7,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    streaming=True
)
#
# llm_light = ChatOpenAI(
#     model="qwen3.5-27b",
#     temperature=0.1,
#     api_key=QWEN_API_KEY,
#     base_url=QWEN_BASE_URL
# )

llm_omni = ChatOpenAI(
    model="qwen2.5-omni-7b",
    temperature=0.7,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)

# llm_omni = ChatTongyi(
#     model="qwen2.5-omni-7b",
#     api_key=QWEN_API_KEY,
# )

# 调用智谱AI的模型
# llm = ChatOpenAI(
#     model="glm-4.5-air",
#     temperature=0.7,
#     api_key=ZHIPU_API_KEY,
#     base_url=ZHIPUAI_BASE_URL,
#     streaming=True
# )

llm_light = ChatOpenAI(
    model="glm-4.7",
    temperature=0.1,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPUAI_BASE_URL
)

llm_vision = ChatOpenAI(
    model="glm-4.6v",
    temperature=0.7,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPUAI_BASE_URL
)

qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)

sys_client = ChatOpenAI(
    base_url="https://devspeedbear.redbearai.com/api/v1",
    api_key="sk-VveFOAp9yEvgvaIcQsJc-5c4DT06Plak8LMlUwkSfGg",
    model="deepseekv4"
)

# 用自定义的方式调用阿里百炼的千问模型
# llm = BailianCustomChatModel(
#     model_name="qwen3.5-flash-2026-02-23",
#     api_key=QWEN_API_KEY,
#     base_url=QWEN_BASE_URL,
#     # enable_thinking=True,
#     temperature=0.7
# )

# 速率限制
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=10,  # 每1秒允许10个请求
#     check_every_n_seconds=1,  # 每1秒检查一次是否允许发出请求
#     max_bucket_size=10  # 控制最大突发请求数量
# )
#
# llm = init_chat_model(
#     model="qwen-plus-2025-12-01",
#     model_provider="openai",
#     rate_limiter=rate_limiter,
#     api_key=QWEN_API_KEY,
#     base_url=QWEN_BASE_URL
# )

zhipuai_client = ZhipuAI(
    api_key=ZHIPU_API_KEY
)

volcano_client = Ark(
    base_url=VOLCANO_BASE_URL,
    api_key=VOLCANO_API_KEY
)

async_volcano_client = AsyncArk(
    base_url=VOLCANO_BASE_URL,
    api_key=VOLCANO_API_KEY
)


