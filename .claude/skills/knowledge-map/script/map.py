"""
knowledge_map — 知识点关系图谱查询
静态概念图 + Milvus 动态增强：先查本地依赖关系，再用知识库补充描述
"""
import sys
import json
from pathlib import Path
_PROJECT_ROOT = next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_concept_graph(subject: str = "math") -> dict:
    graph_path = _PROJECT_ROOT / "knowledge" / "data" / f"{subject}_concept_graph.json"
    if graph_path.exists():
        with open(graph_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _compute_dependency_depth(kp: str, graph: dict, visited: set = None) -> int:
    """计算前置依赖深度"""
    if visited is None:
        visited = set()
    if kp in visited or kp not in graph:
        return 0
    visited.add(kp)
    prereqs = graph[kp].get("prerequisites", [])
    if not prereqs:
        return 1
    return 1 + max(_compute_dependency_depth(p, graph, visited) for p in prereqs)


def _compute_downstream_count(kp: str, graph: dict, visited: set = None) -> int:
    """计算下游影响面（该知识点解锁了多少后续知识点）"""
    if visited is None:
        visited = set()
    if kp in visited or kp not in graph:
        return 0
    visited.add(kp)
    unlocks = graph[kp].get("unlocks", [])
    count = len(unlocks)
    for u in unlocks:
        count += _compute_downstream_count(u, graph, visited)
    return count


async def knowledge_map(
    knowledge_point: str,
    query_type: str = "all",
    subject: str = "math"
) -> dict:
    """
    查询知识点关系

    Args:
        knowledge_point: 知识点名称（如 "诱导公式"）
        query_type: 查询类型 — prerequisites(前置) / extensions(延伸) / all(全部)
        subject: 学科（math/physics/english/chemistry）
    """
    try:
        graph = _load_concept_graph(subject)
        node = graph.get(knowledge_point)

        # 如果本地图谱没有该知识点，降级到 Milvus 语义搜索
        if not node:
            return await _fallback_milvus_search(knowledge_point)

        # 从 Milvus 补充描述内容
        def enrich(kp_list: list) -> list:
            result = []
            for kp in kp_list:
                kp_node = graph.get(kp, {})
                desc = ""
                try:
                    from knowledge import MilvusKB
                    kb = MilvusKB()
                    sr = kb.search(query=kp, top_k=1)
                    if sr:
                        desc = sr[0].get("content", "")[:300]
                except Exception:
                    pass
                result.append({
                    "name": kp,
                    "difficulty": kp_node.get("difficulty", 0),
                    "exam_weight": kp_node.get("exam_weight", "unknown"),
                    "description": desc
                })
            return result

        prereqs = enrich(node.get("prerequisites", []))
        extensions = enrich(node.get("unlocks", []))
        dep_depth = _compute_dependency_depth(knowledge_point, graph)
        downstream_count = _compute_downstream_count(knowledge_point, graph)

        learning_order = _compute_learning_order(knowledge_point, graph)

        result = {
            "success": True,
            "knowledge_point": knowledge_point,
            "source": "concept_graph",
            "node_info": {
                "difficulty": node.get("difficulty", 0),
                "exam_weight": node.get("exam_weight", "unknown"),
                "category": node.get("category", ""),
                "dependency_depth": dep_depth,
                "downstream_impact": downstream_count
            },
            "prerequisites": prereqs,
            "extensions": extensions,
            "recommended_learning_order": learning_order[:6] if query_type in ("all", "extensions") else [],
            "insight": (
                f"「{knowledge_point}」难度 {node.get('difficulty', '?')}/5，"
                f"前置依赖链深度 {dep_depth} 层，"
                f"掌握后解锁 {downstream_count} 个下游知识点。"
                f"前置要求：{'、'.join(p['name'] for p in prereqs) if prereqs else '无'}。"
            )
        }

        if query_type == "prerequisites":
            result.pop("extensions", None)
            result.pop("recommended_learning_order", None)
        elif query_type == "extensions":
            result.pop("prerequisites", None)

        return result

    except Exception as e:
        return {"success": False, "knowledge_point": knowledge_point, "error": str(e)}


def _compute_learning_order(kp: str, graph: dict) -> list:
    """计算从该知识点出发的推荐学习顺序（拓扑排序简化版）"""
    order = []
    visited = set()

    def dfs(node):
        if node in visited or node not in graph:
            return
        visited.add(node)
        for prereq in graph[node].get("prerequisites", []):
            dfs(prereq)
        if node not in order:
            order.append(node)

    dfs(kp)
    # 添加直接延伸
    for ext in graph.get(kp, {}).get("unlocks", []):
        if ext not in visited and ext in graph:
            order.append(ext)
    return order


async def _fallback_milvus_search(knowledge_point: str) -> dict:
    """本地图谱没有该知识点时，降级到 Milvus 语义搜索"""
    try:
        from knowledge import MilvusKB
        kb = MilvusKB()
        results = kb.search(query=knowledge_point, top_k=8)

        prereqs = []
        extensions = []
        node_info = {}

        for r in results:
            content = r.get("content", "")
            metadata = r.get("metadata", {})
            name = metadata.get("name", content[:50])
            relation = metadata.get("relation", "")

            if relation == "prerequisite":
                prereqs.append({"name": name, "content": content[:200], "score": r.get("score", 0)})
            elif relation == "extension":
                extensions.append({"name": name, "content": content[:200], "score": r.get("score", 0)})
            elif not node_info and r.get("score", 0) > 0.8:
                node_info = {"name": name, "content": content[:500]}

        return {
            "success": True,
            "knowledge_point": knowledge_point,
            "source": "milvus_fallback",
            "node_info": node_info,
            "prerequisites": prereqs,
            "extensions": extensions,
            "insight": f"「{knowledge_point}」不在本地概念图谱中，以上结果来自知识库语义搜索。建议补充到概念图谱以获得更准确的关系数据。"
        }
    except Exception as e:
        return {"success": False, "knowledge_point": knowledge_point, "error": str(e)}
