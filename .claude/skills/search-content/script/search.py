"""
search_content — 教育内容搜索
从 Milvus 知识库语义检索，返回去重、多样性排序、分类型提取的结构化结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())))

from loguru import logger


def _deduplicate_and_diversify(results: list, limit: int) -> list:
    """去重并做多样性排序：避免返回5条内容几乎相同的结果"""
    seen_content = set()
    diverse = []
    for r in results:
        content_sig = r.get("content", "")[:100].strip()
        if content_sig not in seen_content:
            seen_content.add(content_sig)
            diverse.append(r)
        if len(diverse) >= limit:
            break
    return diverse


def _classify_content(content: str) -> str:
    """自动判断内容类型：定义型/公式型/例题型/其他"""
    if any(kw in content for kw in ["定义", "是指", "概念", "称为", "指的是"]):
        return "definition"
    if any(kw in content for kw in ["公式", "定理", "定律", "推导", "证明"]):
        return "formula"
    if any(kw in content for kw in ["例", "解：", "已知", "求：", "练习", "题目"]):
        return "example"
    return "concept"


async def search_content(query: str, category: str = "all", limit: int = 5) -> dict:
    """
    搜索教育知识库内容

    Args:
        query: 搜索关键词（如 "三角函数诱导公式"）
        category: 学科分类过滤：math / physics / english / chemistry / all
        limit: 返回结果数量
    """
    try:
        from knowledge import MilvusKB
        kb = MilvusKB()

        results = kb.search(
            query=query,
            top_k=limit * 2,  # 多取一些，去重后还有空间
            filter_type=category if category != "all" else None
        )

        if not results:
            return {
                "success": True,
                "query": query,
                "results": [],
                "content_types": {},
                "message": f"未找到与「{query}」相关的教育内容，建议尝试不同的关键词"
            }

        # 去重 + 多样性排序
        diverse = _deduplicate_and_diversify(results, limit)

        # 按内容类型分类
        structured = []
        type_counts = {}
        for r in diverse:
            content = r.get("content", "")
            ctype = _classify_content(content)
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            structured.append({
                "content": content[:500],
                "content_type": ctype,
                "score": round(r.get("score", 0), 3),
                "category": r.get("category", ""),
                "source": r.get("source", "EduX知识库"),
                "key_excerpt": _extract_key_sentence(content, ctype)
            })

        return {
            "success": True,
            "query": query,
            "results": structured,
            "count": len(structured),
            "content_types": type_counts,
            "search_quality": "exact" if diverse and diverse[0].get("score", 0) > 0.85 else "fuzzy",
            "hint": (
                f"找到 {len(structured)} 条结果（{type_counts.get('definition', 0)}条定义、"
                f"{type_counts.get('formula', 0)}条公式、{type_counts.get('example', 0)}条例题）"
            )
        }

    except Exception as e:
        logger.error(f"search_content error: {e}")
        return {
            "success": False,
            "query": query,
            "results": [],
            "error": f"知识库搜索失败: {str(e)}",
            "fallback_suggestion": "建议尝试更短的关键词，或切换到其他学科分类"
        }


def _extract_key_sentence(content: str, content_type: str) -> str:
    """从内容中提取最核心的一句话"""
    sentences = content.replace("\n", " ").split("。")
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if content_type == "definition" and any(kw in s for kw in ["是", "定义", "指"]):
            return s[:200]
        if content_type == "formula" and any(kw in s for kw in ["公式", "定理", "=", "等于"]):
            return s[:200]
    # fallback: 返回第一句
    return sentences[0].strip()[:200] if sentences else content[:200]
