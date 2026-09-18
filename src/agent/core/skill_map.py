from collections.abc import Callable
from typing import List, Any, Callable

from langchain_core.tools import BaseTool

from agent.tools.data_analysis import DATA_ANALYSIS_TOOLS, TEXT_PROCESSING_TOOLS, LOADER_TOOLS

SKILL_TOOL_MAPPING = {
    "data_analysis": DATA_ANALYSIS_TOOLS,
    "text_processing": TEXT_PROCESSING_TOOLS
}

def get_tools_for_skills(skills_loaded: List[str]) -> list[BaseTool]:
    """
    根据已加载的技能列表，返回应该暴露给模型的工具

    核心逻辑：
    1. Loader工具始终包含
    2. 根据skills_loaded添加对应的技能工具

    Args:
        skills_loaded: 已加载的技能名称列表

    Returns：
        过滤后的工具列表
    """
    # 始终包含Loader工具
    tools = list[Callable[..., Any]](LOADER_TOOLS)

    # 根据已加载的技能添加对应工具
    for skill_name in skills_loaded:
        if skill_name in SKILL_TOOL_MAPPING:
            tools.extend(SKILL_TOOL_MAPPING[skill_name])

    return tools

# 测试工具过滤函数
print("测试 get_tools_for_skills函数")
print(f"\n1.skills_loaded = []")
tools = get_tools_for_skills([])
print(f"返回{len(tools)}个工具: {[t.name for t in tools]}")