"""
deep_research — 深度教育研究
综合网络搜索、知识库和证据综合，处理复杂教育问题
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())))


async def deep_research(query: str, depth: str = "normal") -> dict:
    """
    深度研究教育问题

    Args:
        query: 研究问题（如 "项目式学习对数学成绩的影响"）
        depth: 研究深度 — quick / normal / comprehensive
    """
    try:
        from research import DeepResearchWorkflow

        limit_map = {"quick": 2, "normal": 5, "comprehensive": 10}
        search_limit = limit_map.get(depth, 5)

        workflow = DeepResearchWorkflow()

        # 查询规划
        sub_queries = await workflow.plan_queries(query)

        # 并行搜索
        results = await workflow.parallel_search(sub_queries[:search_limit])

        # 证据综合
        synthesis = await workflow.synthesize(query, results)

        return {
            "success": True,
            "query": query,
            "depth": depth,
            "sources_count": len(results),
            "synthesis": synthesis,
            "key_findings": synthesis.get("key_findings", []),
            "evidence_level": synthesis.get("evidence_level", "moderate"),
            "confidence": synthesis.get("confidence", 0.5)
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "fallback": f"深度研究暂不可用，建议尝试直接搜索知识库：{query}"
        }
