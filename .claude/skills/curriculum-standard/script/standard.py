"""
curriculum_standard — 课程标准/考纲检索
检索课标要求，返回掌握层次、考试频率、教学建议等结构化信息
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


async def curriculum_standard(
    query: str,
    grade: str = "",
    subject: str = ""
) -> dict:
    """
    检索课程标准和考纲要求

    Args:
        query: 搜索关键词（如 "三角函数教学要求"）
        grade: 年级过滤（如 "高一"）
        subject: 学科过滤（如 "数学"）
    """
    try:
        from knowledge import MilvusKB
        kb = MilvusKB()

        results = kb.search(query=query, top_k=5, filter_type="curriculum_standard")

        if not results:
            return {
                "success": True,
                "query": query,
                "standards": [],
                "message": f"未找到与「{query}」相关的课程标准"
            }

        standards = []
        for r in results:
            content = r.get("content", "")
            standards.append({
                "content": content[:500],
                "grade": r.get("grade", grade or "未标注"),
                "subject": r.get("subject", subject or "未标注"),
                "mastery_level": _infer_mastery_level(content),
                "exam_frequency": _infer_exam_frequency(content),
                "score": round(r.get("score", 0), 3)
            })

        return {
            "success": True,
            "query": query,
            "standards": standards,
            "count": len(standards),
            "summary": _generate_standard_summary(standards, query)
        }

    except Exception as e:
        return {"success": False, "query": query, "standards": [], "error": str(e)}


def _infer_mastery_level(content: str) -> str:
    """从课标文本推断掌握层次"""
    content_lower = content.lower()
    if any(kw in content_lower for kw in ["掌握", "熟练", "灵活", "综合"]):
        return "掌握"
    elif any(kw in content_lower for kw in ["理解", "应用", "运用"]):
        return "理解"
    elif any(kw in content_lower for kw in ["了解", "知道", "认识", "初步"]):
        return "了解"
    elif any(kw in content_lower for kw in ["经历", "体验", "探究"]):
        return "经历"
    return "未标注"


def _infer_exam_frequency(content: str) -> str:
    """从课标文本推断考试频率"""
    content_lower = content.lower()
    if any(kw in content_lower for kw in ["必考", "重点", "核心", "主干"]):
        return "high"
    elif any(kw in content_lower for kw in ["常考", "常见", "高频"]):
        return "high"
    elif any(kw in content_lower for kw in ["偶尔", "选考", "拓展"]):
        return "low"
    return "medium"


def _generate_standard_summary(standards: list, query: str) -> str:
    """生成课标摘要"""
    if not standards:
        return f"未找到「{query}」的课标要求"
    levels = set(s["mastery_level"] for s in standards)
    freqs = set(s["exam_frequency"] for s in standards)
    return (
        f"「{query}」课标要求：{', '.join(levels)}层次，"
        f"考试频率 {'高' if 'high' in freqs else '中等' if 'medium' in freqs else '低'}。"
        f"共 {len(standards)} 条相关标准"
    )
