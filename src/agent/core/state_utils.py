"""关键状态"""
from typing import Annotated, List

from langgraph.graph import MessagesState


# # 第一种模式: 替换模式
# def skill_list_reducer(current: List[str], new: List[str]) -> List[str]:
#     """替换模式：用新列表替换旧列表"""
#     return new
#
# class SkillState(MessagesState):
#     """
#     Skill状态Schema
#
#     使用MessageState作为基类，它已经包含了messages字段
#     我们只需要添加skills_loaded字段
#     """
#     skills_loaded: Annotated[List[str], skill_list_reducer] = []


# 第二种模式：累计模式
def skil_list_accumulator(current: List[str], new: List[str]) -> List[str]:
    """
    累计模式：合并已加载的Skills
    保持所有已加载的技能，而不是替换
    """
    if not current:
        return new
    combined = current + [s for s in new if s not in current]
    return combined

# 使用累计模式的reducer
class SkillState(MessagesState):
    """
    Skill状态Schema
    """
    skills_loaded: Annotated[List[str], skil_list_accumulator] = []