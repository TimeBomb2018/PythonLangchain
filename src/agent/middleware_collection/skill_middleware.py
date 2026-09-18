from typing import List, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from agent.core.skill_map import get_tools_for_skills


class SkillMiddleware(AgentMiddleware):
    """
    Skill中间件 - 实现动态工具过滤

    这是 Claude Skills 的核心组件

    工作原理：
    1. 在每次模型调用前拦截请求
    2. 从request.state中读取skills_loaded列表
    3. 根据skills_loaded过滤工具列表
    4. 使用request.override()替换工具列表
    5. 传递给下一个handler

    这样，模型在每次调用时只会看到相关的工具！
    """
    def __init__(self, verbose: bool = True):
        """
        初始化 SkillMiddleware

        Args:
            verbose: 是否打印详细日志（用于调试和演示）
        """
        super().__init__()
        self.verbose = verbose
        self.call_count = 0

    @staticmethod
    def _get_skills_from_state(request: ModelRequest) -> List[str]:
        """
        从请求状态提取 skills_loaded

        注意：AgentState 是 TypedDict, 本质上是dict
        所以我们使用字典方式访问
        """
        skills_loaded = []

        if hasattr(request, 'state') and request.state is not None:
            if isinstance(request.state, dict):
                skills_loaded = request.state.get("skills_loaded", [])
            else:
                skills_loaded = getattr(request.state, "skills_loaded", [])

        return skills_loaded

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        【核心方法】拦截模型调用，动态过滤工具

        这是整个Claude Skills 系统最关键的方法！
        """
        self.call_count += 1
        skills_loaded = self._get_skills_from_state(request)
        filtered_tools = get_tools_for_skills(skills_loaded)

        if self.verbose:
            print(f"\n{'-' * 60}")
            print(f"[SkillMiddleware]第{self.call_count}次模型调用")
            print(f"{'-'} * 60")
            print(f"skills_loaded: {skills_loaded}")
            print(f"过滤后工具（{len(filtered_tools)}个）: {[t.name for t in filtered_tools]}")

            # 对比原始工具数量
            if hasattr(request, "tools") and request.tools:
                original_count = len(request.tools)
                print(f"工具数量变化：{original_count} -> {len(filtered_tools)}")

        filtered_request = request.override(tools=filtered_tools)

        if self.verbose:
            print(f"已将过滤后的工具传递给模型")
            print(f"{'-' * 60}\n")

        return handler(filtered_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """
        异步版本 - 与同步版本逻辑相同

        LangChain 可能使用异步调用，所以需要同时实现两个版本
        """
        self.call_count += 1
        skills_loaded = self._get_skills_from_state(request)
        filtered_tools = get_tools_for_skills(skills_loaded)

        if self.verbose:
            print(f"\n{'-' * 60}")
            print(f"[SkillMiddleware]第{self.call_count}次模型调用")
            print(f"{'-'} * 60")
            print(f"skills_loaded: {skills_loaded}")
            print(f"过滤后工具（{len(filtered_tools)}个）: {[t.name for t in filtered_tools]}")

            # 对比原始工具数量
            if hasattr(request, "tools") and request.tools:
                original_count = len(request.tools)
                print(f"工具数量变化：{original_count} -> {len(filtered_tools)}")

        filtered_request = request.override(tools=filtered_tools)
        return await handler(filtered_request)



