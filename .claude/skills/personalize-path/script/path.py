"""
personalize_path — 个性化学习路径生成
基于学生真实状态动态规划：前置依赖链解析、复习/新学区分、时间估算、素材匹配
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def _load_concept_graph(subject: str = "") -> dict:
    graph_path = Path(__file__).resolve().parents[4] / "knowledge" / "data" / f"{subject or 'math'}_concept_graph.json"
    if graph_path.exists():
        with open(graph_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_prerequisite_chain(topic: str, graph: dict, profile, visited: set = None) -> list:
    """
    递归解析前置依赖链，返回按学习顺序排列的知识点列表。
    已掌握的前置知识不进入链条。
    """
    from memory.student_profile import MasteryLevel

    if visited is None:
        visited = set()
    if topic in visited:
        return []
    visited.add(topic)

    graph_node = graph.get(topic, {})
    prereqs = graph_node.get("prerequisites", [])

    chain = []
    for prereq in prereqs:
        if prereq in visited:
            continue
        p_node = profile.knowledge_graph.get(prereq)
        # 如果前置知识点未掌握或刚接触，需要先学/复习
        if not p_node or p_node.mastery in (MasteryLevel.NOT_STARTED, MasteryLevel.INTRODUCED):
            sub_chain = _resolve_prerequisite_chain(prereq, graph, profile, visited)
            chain.extend(sub_chain)
            if prereq not in chain:
                chain.append(prereq)

    return chain


async def personalize_path(
    student_id: str = "",
    topic: str = "",
    learning_style: str = "unknown",
    target_mastery: str = "proficient"
) -> dict:
    """
    生成个性化学习路径

    Args:
        student_id: 学生ID
        topic: 目标学习主题（如 "诱导公式"）
        learning_style: 学习风格（visual/textual/example/abstract/hands_on/unknown）
        target_mastery: 目标掌握度（introduced/practicing/proficient/mastered）
    """
    try:
        from core.personalization import choose_teaching_approach, estimate_optimal_session_length
        from memory.student_profile import StudentProfileManager, MasteryLevel, LearningStyle

        manager = StudentProfileManager()
        profile = manager.get_profile(student_id)

        style = LearningStyle(learning_style) if learning_style != "unknown" else profile.learning_style
        approach = choose_teaching_approach(profile, topic)

        # 1. 查询当前知识点状态
        current_node = profile.knowledge_graph.get(topic)
        current_mastery = current_node.mastery.value if current_node else "not_started"

        # 2. 解析前置依赖链
        concept_graph = _load_concept_graph("math")
        prereq_chain = _resolve_prerequisite_chain(topic, concept_graph, profile)

        # 3. 区分：需复习 vs 需新学
        to_review = []
        to_learn = []
        for p in prereq_chain:
            p_node = profile.knowledge_graph.get(p)
            if p_node and p_node.mastery in (MasteryLevel.INTRODUCED, MasteryLevel.PRACTICING):
                to_review.append({"name": p, "current_mastery": p_node.mastery.value, "confidence": p_node.confidence})
            else:
                to_learn.append({"name": p, "difficulty": concept_graph.get(p, {}).get("difficulty", 2)})

        # 4. 匹配知识库素材
        learning_materials = {}
        try:
            from knowledge import MilvusKB
            kb = MilvusKB()
            for kp in to_learn[:4] + [topic]:
                results = kb.search(query=kp, top_k=2)
                if results:
                    learning_materials[kp] = [
                        {"content": r.get("content", "")[:300], "source": r.get("source", "EduX知识库"),
                         "score": round(r.get("score", 0), 2)}
                        for r in results
                    ]
        except Exception:
            pass

        # 5. 时间估算
        session_len = estimate_optimal_session_length(profile)

        # 6. 动态构建路径步骤
        path = []
        step = 0

        if to_review:
            step += 1
            review_names = [r["name"] for r in to_review]
            path.append({
                "step": step,
                "phase": "复习前置知识",
                "action": f"快速回顾：{' → '.join(review_names)}",
                "items": review_names,
                "estimated_minutes": min(5 * len(to_review), session_len // 3),
                "method": "快速回忆 + 1道检测题，确认是否真正掌握",
                "materials": [learning_materials.get(name, []) for name in review_names]
            })

        for item in to_learn[:3]:
            step += 1
            path.append({
                "step": step,
                "phase": f"学习前置知识「{item['name']}」",
                "action": f"掌握 {item['name']} 的核心概念和基本应用",
                "items": [item["name"]],
                "estimated_minutes": min(10 + item["difficulty"] * 5, session_len // 2),
                "method": approach["method"],
                "materials": learning_materials.get(item["name"], [])
            })

        step += 1
        path.append({
            "step": step,
            "phase": f"学习「{topic}」核心",
            "action": f"从 {current_mastery} 提升到 {target_mastery}",
            "items": [topic],
            "estimated_minutes": min(20, session_len // 2),
            "method": approach["method"],
            "description": approach["description"],
            "example_type": approach["example_type"],
            "materials": learning_materials.get(topic, [])
        })

        step += 1
        path.append({
            "step": step,
            "phase": "练习巩固",
            "action": "3道递进式练习题（易→中→难），覆盖标准题型和变式",
            "items": [],
            "estimated_minutes": min(15, session_len // 3),
            "method": "引导式练习，不给答案只给提示"
        })

        step += 1
        path.append({
            "step": step,
            "phase": "关联总结",
            "action": f"用{style.value}方式构建{topic}的知识网络",
            "items": [],
            "estimated_minutes": 5,
            "method": "思维导图 / 公式卡片 / 费曼讲解法"
        })

        total_min = sum(p["estimated_minutes"] for p in path)

        return {
            "success": True,
            "topic": topic,
            "current_mastery": current_mastery,
            "target_mastery": target_mastery,
            "learning_style": style.value,
            "pace": profile.preferred_pace,
            "teaching_approach": approach,
            "total_estimated_minutes": total_min,
            "recommended_sessions": max(1, round(total_min / session_len)),
            "session_length_minutes": session_len,
            "prerequisites_resolved": len(prereq_chain) - len(to_learn) - len(to_review),
            "prerequisites_to_review": to_review,
            "prerequisites_to_learn": [item["name"] for item in to_learn],
            "path": path,
            "suggestion": (
                f"建议分 {max(1, round(total_min / session_len))} 次学习完成，每次约 {session_len} 分钟。"
                f"当前「{topic}」掌握度为 {current_mastery}，目标 {target_mastery}。"
            )
        }

    except Exception as e:
        return {"success": False, "topic": topic, "error": str(e)}
