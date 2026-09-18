from typing import List
import statistics

from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool
def skill_data_analysis(runtime) -> Command:
    """
    加载数据分析技能
    """
    instructions = """数据分析技能已成功加载！
    
    现在你可以使用以下工具：
    · calculate_statistics(numbers): 计算一组数字的统计信息
    · generate_chart(data, chart_type): 生成数据图表
    
    请继续使用这些工具完成用户的数据分析任务。
    """
    return Command(
        update={
            "messages": [ToolMessage(
                content=instructions,
                tool_call_id=runtime.tool_call_id
            )],
            "skills_loaded": ["data_analysis"]  # 关键：直接更新状态
        }
    )

@tool
def skill_text_processing(runtime) -> Command:
    """
    加载文本处理技能

    调用次工具后，你将获得以下文本处理相关的工具：
    - summarize_text: 生成文本摘要
    - extract_keywords: 提取关键词

    使用场景：当用户需要处理文本、生成摘要或提取关键信息时，
    请先调用此工具加载文本处理技能。
    """
    instructions = """文本处理技能已成功加载！
    
    现在你可以使用以下工具：
    · summarize_text(text, max_length): 生成文本摘要
    · extract_keywords(text, num_keywords): 提取关键词
    
    请继续使用这些工具完成用户的文本处理任务。
    """

    return Command(
        update={
            "messages": [ToolMessage(
                content=instructions,
                tool_call_id=runtime.tool_call_id
            )],
            "skills_loaded": ["text_processing"]
        }
    )

# ===================== 数据分析工具 ======================
# 这些工具只有在加载了data_analysis技能后才可见

@tool
def calculate_statistics(numbers: List[float]) -> str:
    """
    计算一组数字的统计信息，包括平均值、最大值、最小值、标准差等。

    Args:
        numbers: 要分析的数字列表
    """
    if not numbers:
        return "错误：数字列表为空"

    result = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }

    if len(numbers) > 1:
        result["stdev"] = statistics.stdev(numbers)

    return f"统计结果：{result}"

@tool
def generate_chart(data: List[float], chart_type: str = "bar") -> str:
    """
    根据数据生成图表（模拟）

    Args:
        data: 数据列表
        chart_type: 图标类型（bar, line, pie)
    """
    return f"已生成{chart_type}图表，包含{len(data)}个数据点"

# ===================== 文本处理工具 ======================
# 这些工具只有在加载了text_processing技能后才可见

@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    生成文本摘要。

    Args:
        text: 要摘要的文本
        max_length: 摘要最大长度
    """
    if len(text) <= max_length:
        return f"摘要：{text}"
    return f"摘要：{text[:max_length]}..."

@tool
def extract_keywords(text: str, num_keywords: int = 5) -> str:
    """
    从文本中提取关键词。

    Args:
        text: 要分析的文本
        num_keywords: 要提取的关键词数量
    """
    # 简单模拟：取前几个单词S
    words = text.split()[:num_keywords]
    return f"关键词: {','.join(words)}"


# 组织工具
LOADER_TOOLS = [skill_data_analysis, skill_text_processing]
DATA_ANALYSIS_TOOLS = [calculate_statistics, generate_chart]
TEXT_PROCESSING_TOOLS = [summarize_text, extract_keywords]
ALL_TOOLS = LOADER_TOOLS + DATA_ANALYSIS_TOOLS + TEXT_PROCESSING_TOOLS

print("工具定义完成")
print(f"Loader工具（{len(LOADER_TOOLS)}）: {[t.name for t in LOADER_TOOLS]}")
print(f"数据分析工具（{len(DATA_ANALYSIS_TOOLS)}）: {[t.name for t in DATA_ANALYSIS_TOOLS]}")
print(f"文本处理工具（{len(TEXT_PROCESSING_TOOLS)}）: {[t.name for t in TEXT_PROCESSING_TOOLS]}")
print(f"总计：{len(ALL_TOOLS)}个工具")