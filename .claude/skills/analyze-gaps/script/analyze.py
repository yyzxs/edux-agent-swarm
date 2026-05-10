"""
analyze_gaps — 学习薄弱点分析
根因分析 + 影响评估 + 分组：不只列出薄弱点，更回答"为什么薄弱"和"影响多大"
"""
import sys
import json
from pathlib import Path
_PROJECT_ROOT = next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_concept_graph(subject: str = "") -> dict:
    """加载知识点依赖关系图"""
    graph_path = _PROJECT_ROOT / "knowledge" / "data" / f"{subject or 'math'}_concept_graph.json"
    if graph_path.exists():
        with open(graph_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


async def analyze_gaps(
    student_id: str = "",
    subject: str = "",
    grade: str = ""
) -> dict:
    """
    分析学习薄弱环节：追根因、评估影响、给优先级

    Args:
        student_id: 学生ID
        subject: 学科过滤（math/physics/english/chemistry，空=全部）
        grade: 年级（用于参考课标）
    """
    try:
        from memory.student_profile import StudentProfileManager, MasteryLevel

        manager = StudentProfileManager()
        profile = manager.get_profile(student_id)
        weak_points = profile.get_weak_points(max_count=10)

        if subject:
            weak_points = [n for n in weak_points if n.category == subject or subject in n.category]

        if not weak_points:
            return {
                "success": True,
                "gaps": [],
                "gap_groups": [],
                "summary": "未发现明显薄弱环节，继续保持！"
            }

        concept_graph = _load_concept_graph(subject or "math")

        gaps = []
        for node in weak_points:
            kp_name = node.name
            error_ratio = node.error_count / max(node.interaction_count, 1)

            # 1. 根因分析：找前置依赖中未掌握的知识点
            graph_node = concept_graph.get(kp_name, {})
            prereqs = graph_node.get("prerequisites", [])
            root_causes = []
            for prereq in prereqs:
                prereq_node = profile.knowledge_graph.get(prereq)
                prereq_graph = concept_graph.get(prereq, {})
                if prereq_node and prereq_node.mastery in (
                    MasteryLevel.NOT_STARTED, MasteryLevel.INTRODUCED
                ):
                    root_causes.append({
                        "knowledge_point": prereq,
                        "mastery": prereq_node.mastery.value,
                        "relation": f"「{kp_name}」依赖「{prereq}」，但「{prereq}」掌握度为 {prereq_node.mastery.value}",
                        "prereq_difficulty": prereq_graph.get("difficulty", 0)
                    })

            # 2. 影响评估：这个薄弱点阻塞了哪些后续知识点
            downstream = graph_node.get("unlocks", [])
            blocked = []
            for ds in downstream:
                ds_node = profile.knowledge_graph.get(ds)
                if ds_node and ds_node.mastery in (MasteryLevel.NOT_STARTED,):
                    blocked.append({"knowledge_point": ds, "status": "未开始，被阻塞"})
                elif ds_node and ds_node.mastery == MasteryLevel.INTRODUCED:
                    blocked.append({"knowledge_point": ds, "status": "学习中，受前置薄弱影响"})

            # 3. 错误模式归类
            error_types = {}
            for e in profile.error_log:
                ekp = e.knowledge_point_id if hasattr(e, 'knowledge_point_id') else e.get("knowledge_point_id", "")
                if ekp == node.id or ekp == kp_name or kp_name in str(e):
                    et = e.error_type if hasattr(e, 'error_type') else e.get("error_type", "unknown")
                    error_types[et] = error_types.get(et, 0) + 1
            primary_error_type = max(error_types, key=error_types.get) if error_types else "unknown"

            # 4. 课标关联
            exam_weight = graph_node.get("exam_weight", "unknown")

            # 5. 综合优先级评分
            urgency_score = (
                error_ratio * 0.35 +
                len(root_causes) * 0.25 +
                len(blocked) * 0.15 +
                (3.0 if exam_weight == "high" else 1.5 if exam_weight == "medium" else 0.5) * 0.15 +
                (1.0 if node.mastery in (MasteryLevel.INTRODUCED, MasteryLevel.NOT_STARTED) else 0) * 0.10
            )

            gaps.append({
                "knowledge_point": kp_name,
                "mastery": node.mastery.value,
                "confidence": node.confidence,
                "error_count": node.error_count,
                "interaction_count": node.interaction_count,
                "error_ratio": round(error_ratio, 2),
                "primary_error_type": primary_error_type,
                "error_distribution": error_types,
                "root_causes": root_causes,
                "blocks_learning_of": blocked,
                "exam_weight": exam_weight,
                "urgency_score": round(urgency_score, 3),
                "urgency": "high" if urgency_score > 0.5 else "medium" if urgency_score > 0.25 else "low"
            })

        gaps.sort(key=lambda g: g["urgency_score"], reverse=True)

        # 分组：共享根因的薄弱点合并，让 Agent 知道"这 3 个问题其实是一个问题"
        groups = _group_by_root_cause(gaps)

        high_count = sum(1 for g in gaps if g["urgency"] == "high")

        if gaps:
            top = gaps[0]
            root_info = ""
            if top["root_causes"]:
                root_info = f"根因：「{top['root_causes'][0]['knowledge_point']}」未掌握导致「{top['knowledge_point']}」薄弱"
            summary = (
                f"发现 {len(gaps)} 个薄弱环节，{high_count} 个高优先级。"
                f"最紧急：「{top['knowledge_point']}」（{top['mastery']}）。{root_info}"
            )
        else:
            summary = "未发现薄弱环节"

        return {
            "success": True,
            "gaps": gaps,
            "gap_groups": groups,
            "count": len(gaps),
            "high_urgency_count": high_count,
            "medium_urgency_count": sum(1 for g in gaps if g["urgency"] == "medium"),
            "summary": summary
        }

    except Exception as e:
        return {"success": False, "gaps": [], "error": str(e)}


def _group_by_root_cause(gaps: list) -> list:
    """将共享相同根因的薄弱点分组"""
    groups = {}
    for gap in gaps:
        for rc in gap.get("root_causes", []):
            key = rc["knowledge_point"]
            if key not in groups:
                groups[key] = {
                    "root_cause": key,
                    "affected": [],
                    "relation_sample": rc["relation"]
                }
            if gap["knowledge_point"] not in groups[key]["affected"]:
                groups[key]["affected"].append(gap["knowledge_point"])

    result = []
    for key, g in groups.items():
        if len(g["affected"]) >= 2:
            g["insight"] = f"「{key}」未掌握导致了 {len(g['affected'])} 个后续知识点的薄弱：{'、'.join(g['affected'])}"
            result.append(g)

    result.sort(key=lambda g: len(g["affected"]), reverse=True)
    return result
