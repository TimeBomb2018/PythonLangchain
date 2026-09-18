import os

from dotenv import load_dotenv

load_dotenv(override=True)
# 硅基流动
SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY")
SILICON_FLOW_BASE_URL = os.getenv("SILICON_FLOW_BASE_URL")
# 阿里百炼
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
# 智谱AI
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPUAI_BASE_URL = os.getenv("ZHIPUAI_BASE_URL")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# 火山字节
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY")
VOLCANO_BASE_URL = os.getenv("VOLCANO_BASE_URL")