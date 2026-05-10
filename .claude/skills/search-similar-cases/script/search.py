"""
search_similar_cases — 搜索相似历史学习案例（长期记忆）
返回相似案例 + 为什么相似 + 上次怎么解决的
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())))


async def search_similar_cases(query: str, limit: int = 3, student_id: str = "") -> dict:
    """
    从长期记忆中搜索相似的学习历史

    Args:
        query: 查询文本
        limit: 返回结果数量
        student_id: 学生ID（可选，用于过滤该学生的历史）
    """
    try:
        from memory.long_term import LongTermMemory
        memory = LongTermMemory()

        results = memory.search_similar_sessions(query=query, limit=limit)

        if not results:
            return {
                "success": True,
                "query": query,
                "similar_cases": [],
                "count": 0,
                "message": "未找到相似的历史学习案例"
            }

        cases = []
        for i, r in enumerate(results):
            content = r.get("content", "")
            score = r.get("score", 0)

            cases.append({
                "case_id": r.get("memory_id", f"case_{i}"),
                "content": content[:400],
                "similarity_score": round(score, 3),
                "timestamp": r.get("timestamp", ""),
                "why_similar": _infer_similarity_reason(query, content),
                "resolution_hint": _extract_resolution(content)
            })

        return {
            "success": True,
            "query": query,
            "similar_cases": cases,
            "count": len(cases),
            "top_match_score": round(cases[0]["similarity_score"], 3) if cases else 0,
            "hint": (
                f"找到 {len(cases)} 个相似历史案例，最佳匹配相似度 {cases[0]['similarity_score']:.0%}。"
                f"上次处理方式：{cases[0]['resolution_hint'][:100]}"
                if cases else "无相似案例"
            )
        }

    except Exception as e:
        return {"success": False, "query": query, "similar_cases": [], "error": str(e)}


def _infer_similarity_reason(query: str, content: str) -> str:
    """推断为什么这个案例被认为是相似的（基于关键词重叠）"""
    query_words = set(query)
    content_words = set(content)
    overlap = query_words & content_words
    # 只关心有意义的字符重叠
    meaningful = [c for c in overlap if '一' <= c <= '鿿']  # 中文字符
    if len(meaningful) >= 3:
        return f"共享关键词：{'、'.join(meaningful[:5])}"
    return "语义相似匹配"


def _extract_resolution(content: str) -> str:
    """从历史案例中提取解决方式的关键信息"""
    # 尝试找到回答部分
    markers = ["回答：", "建议：", "方案：", "答复：", "answer", "学习建议", "复习计划"]
    for marker in markers:
        idx = content.find(marker)
        if idx != -1:
            return content[idx:idx + 200].strip()
    # fallback: 取后半段（通常是回答）
    half = len(content) // 2
    return content[half:half + 200].strip()
