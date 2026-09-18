from typing import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse
)


class LoggingMiddleware(AgentMiddleware):
    """
    日志中间件 - 记录每次模型调用的信息

    这是一个最简单的Middleware示例，用于理解基本工作流程
    """
    def __init__(self, name: str = "Logger"):
        super().__init__()
        self.name = name
        self.call_count = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        拦截模型调用，打印日志信息
        """
        self.call_count += 1

        # 1. 调用前：记录请求信息
        print(f"\n{'=' * 60}")
        print(f"[{self.name}] 第 {self.call_count} 次模型调用")
        print(f"{'=' * 60}")

        # 打印工具信息
        if hasattr(request, 'tools') and request.tools:
            tool_names = [t.name for t in request.tools]
            print(f"可用工具({len(tool_names)})个: {tool_names}")

        # 打印状态信息
        if hasattr(request, 'state') and request.state:
            print(f"当前状态：{request.state}")

        # 2. 调用下一个处理器（这里是实际的模型调用）
        response = handler(request)

        # 3. 调用后：可以处理响应（这里只是打印）
        print(f"模型调用完成")
        print(f"{'=' * 60}\n")

        return response


