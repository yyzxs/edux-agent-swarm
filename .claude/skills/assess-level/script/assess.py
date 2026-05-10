"""
assess_level — 学力水平诊断
用 LLM 分析学生真实回答质量，区分概念混淆和计算粗心，
返回掌握度等级、置信度、具体证据和误解清单
"""
import sys
import json
import re
from pathlib import Path
sys.path.insert(0, str(next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())))

from memory.student_profile import MasteryLevel


async def assess_level(
    knowledge_point: str,
    student_response: str = "",
    error_history: list = None,
    interaction_count: int = 0,
    student_id: str = ""
) -> dict:
    """
    评估学生对特定知识点的掌握水平

    Args:
        knowledge_point: 知识点名称（如 "正弦定理"）
        student_response: 学生最近一次回答或表现描述
        error_history: 该知识点的历史错误记录 [{"error_type": "...", "question_snippet": "...", "timestamp": "..."}]
        interaction_count: 该知识点的历史互动次数
        student_id: 学生ID（用于加载画像上下文）
    """
    # 快速路径：从未接触
    if interaction_count == 0 and not student_response:
        return {
            "success": True,
            "knowledge_point": knowledge_point,
            "mastery_level": "not_started",
            "confidence": 1.0,
            "evidence": "从未互动过此知识点",
            "misconceptions": [],
            "summary": f"「{knowledge_point}」尚未开始学习"
        }

    # 构建评估上下文
    error_context = ""
    if error_history:
        # 只取该知识点的错误，最近5条
        relevant_errors = [
            e for e in error_history
            if (isinstance(e, dict) and
                (e.get("knowledge_point_id") == knowledge_point or
                 knowledge_point in e.get("question_snippet", "")))
        ][-5:]
        if relevant_errors:
            lines = []
            for e in relevant_errors:
                et = e.get("error_type", "unknown")
                qs = e.get("question_snippet", "")[:120]
                lines.append(f"- [{et}] {qs}")
            error_context = "历史错误记录：\n" + "\n".join(lines)

    # LLM 驱动的深度评估
    try:
        from core.llm_client import LLMClient
        llm = LLMClient()

        prompt = f"""你是教育评估专家。评估学生对「{knowledge_point}」的掌握水平。

**学生最近回答/表现**：
{student_response or "无最近回答"}

{error_context}

**历史互动次数**：{interaction_count}

**掌握度等级标准**：
- not_started: 完全没有接触过
- introduced: 能复述定义，但无法独立应用到新题目
- practicing: 能解标准题型，但变式或综合题会出错
- proficient: 能独立解大部分题目，偶尔有小错
- mastered: 能灵活运用、举一反三，能向他人讲解

**关键判断原则**：
1. "计算粗心"和"概念混淆"必须区分——前者不降级，后者必须降级
2. 如果学生展示了正确思路只是最终结果算错，应判 practicing 或 proficient，不应判 introduced
3. 如果学生把核心概念搞反了（如 sin/cos 混淆），应判 introduced
4. 互动次数是参考不是决定因素——3 次高质量互动可能已 proficient，10 次低质量互动可能仍是 practicing

返回 JSON（不要 markdown 代码块）：
{{"level": "practicing", "confidence": 0.65, "evidence": "学生正确应用了公式但代入数值时出错,属于计算失误而非概念问题", "misconceptions": ["混淆了弧度制和角度制"]}}"""

        response = await llm.chat([{"role": "user", "content": prompt}])
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            level = result.get("level", "introduced")
            return {
                "success": True,
                "knowledge_point": knowledge_point,
                "mastery_level": level,
                "confidence": result.get("confidence", 0.5),
                "evidence": result.get("evidence", ""),
                "misconceptions": result.get("misconceptions", []),
                "assessment_method": "llm",
                "summary": (
                    f"「{knowledge_point}」掌握度：{level}（置信度 {result.get('confidence', 0.5):.0%}）。"
                    f"{result.get('evidence', '')}"
                )
            }
    except Exception:
        pass

    # 降级：规则引擎兜底
    return _rule_based_fallback(knowledge_point, error_history, interaction_count)


def _rule_based_fallback(knowledge_point: str, error_history: list, interaction_count: int) -> dict:
    """规则引擎兜底，LLM 不可用时使用"""
    error_count = len(error_history) if error_history else 0

    if interaction_count == 0:
        level, confidence = MasteryLevel.NOT_STARTED, 1.0
    elif interaction_count <= 2 and error_count > 0:
        level, confidence = MasteryLevel.INTRODUCED, 0.5
    elif interaction_count <= 2:
        level, confidence = MasteryLevel.INTRODUCED, 0.6
    elif 3 <= interaction_count <= 6 and error_count > 2:
        level, confidence = MasteryLevel.PRACTICING, 0.4
    elif 3 <= interaction_count <= 6:
        level, confidence = MasteryLevel.PRACTICING, 0.65
    elif interaction_count >= 7 and error_count <= 1:
        level, confidence = MasteryLevel.PROFICIENT, 0.8
    elif interaction_count >= 10 and error_count == 0:
        level, confidence = MasteryLevel.MASTERED, 0.9
    else:
        level, confidence = MasteryLevel.PRACTICING, 0.5

    return {
        "success": True,
        "knowledge_point": knowledge_point,
        "mastery_level": level.value,
        "confidence": confidence,
        "evidence": f"规则引擎评估（interaction_count={interaction_count}, error_count={error_count}）",
        "misconceptions": [],
        "assessment_method": "rule_fallback",
        "summary": f"「{knowledge_point}」掌握度：{level.value}（置信度 {confidence:.0%}，规则引擎）"
    }
