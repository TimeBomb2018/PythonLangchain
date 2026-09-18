import os
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from openai import OpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from agent.env_utils import QWEN_BASE_URL, QWEN_API_KEY


class BailianCustomChatModel(BaseChatOpenAI):
    """自定义聊天模型，直接返回AIMessage"""

    # 模型配置参数
    model_name: str = "qwen-plus-2025-12-01"
    api_key :str = QWEN_API_KEY
    base_url: str = QWEN_BASE_URL
    enable_thinking: bool = False
    temperature: float = 0.7
    max_tokens: int = 8192
    max_retries: int = 3

    @property
    def _llm_type(self) -> str:
        return f"bailian_chat_{self.model_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """核心生成方法，返回ChatResult"""

        # 将langchain消息转换为OpenAI格式
        openai_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                openai_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})

        # 调用API
        client = OpenAI(
            api_key=self.api_key or os.getenv("QWEN_API_KEY"),
            base_url=self.base_url
        )

        call_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        if self.enable_thinking:
            call_params["extra_body"] = {"enable_thinking": True}

        try:
            completion = client.chat.completions.create(**call_params)
            rc = None
            if completion.choices and completion.choices[0].message.content:
                if getattr(completion.choices[0].message, "model_extra", None):
                    if "reasoning_content" in getattr(completion.choices[0].message, "model_extra", None):
                        rc = getattr(completion.choices[0].message, "model_extra", None)["reasoning_content"]
                aimessage = AIMessage(
                    content=completion.choices[0].message.content,
                    additional_kwargs={
                        "model": self.model_name,
                        "usage": getattr(completion, "usage", {}),
                        "reasoning_content": rc if rc else ""
                    }
                )
                return ChatResult(generations=[ChatGeneration(message=aimessage)])
            else:
                raise ValueError("模型返回空响应")
        except Exception as e:
            raise RuntimeError(f"调用模型失败: {str(e)}")





