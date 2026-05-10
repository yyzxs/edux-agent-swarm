"""
StudentProfile 纯逻辑单元测试

不依赖 LLM、向量数据库、网络 —— 可直接确定性运行。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.student_profile import (
    StudentProfile,
    StudentProfileManager,
    KnowledgeNode,
    ErrorRecord,
    SessionSnapshot,
    LearningStyle,
    MasteryLevel,
    DifficultyLevel,
)


# ── 辅助 ───────────────────────────────────────────

def make_demo_profile(student_id: str = "s1") -> StudentProfile:
    p = StudentProfile(student_id=student_id, grade_level="高二")
    p.update_mastery("math.trig.sin", MasteryLevel.PRACTICING, confidence=0.6)
    p.update_mastery("math.trig.cos", MasteryLevel.PROFICIENT, confidence=0.85)
    p.update_mastery("math.trig.tan", MasteryLevel.INTRODUCED, confidence=0.3)
    p.add_error("math.trig.tan", "概念混淆", "tan 的定义是什么")
    p.add_error("math.trig.tan", "计算错误", "tan45° 的值")
    return p


# ── 序列化 ──────────────────────────────────────────

class TestSerialization:
    """to_serializable → from_serializable 往返"""

    def test_roundtrip_preserves_core_fields(self):
        orig = make_demo_profile()
        data = orig.to_serializable()
        restored = StudentProfile.from_serializable(data)

        assert restored.student_id == orig.student_id
        assert restored.grade_level == orig.grade_level
        assert restored.learning_style == orig.learning_style
        assert restored.style_confidence == orig.style_confidence
        assert restored.total_interactions == orig.total_interactions

    def test_roundtrip_preserves_knowledge_graph(self):
        orig = make_demo_profile()
        restored = StudentProfile.from_serializable(orig.to_serializable())

        assert set(restored.knowledge_graph.keys()) == set(orig.knowledge_graph.keys())
        for kp_id in orig.knowledge_graph:
            orig_node = orig.knowledge_graph[kp_id]
            rest_node = restored.knowledge_graph[kp_id]
            assert rest_node.name == orig_node.name
            assert rest_node.mastery == orig_node.mastery
            assert rest_node.confidence == orig_node.confidence
            assert rest_node.interaction_count == orig_node.interaction_count
            assert rest_node.error_count == orig_node.error_count

    def test_roundtrip_preserves_error_log(self):
        orig = make_demo_profile()
        restored = StudentProfile.from_serializable(orig.to_serializable())

        assert len(restored.error_log) == len(orig.error_log)
        for r_err, o_err in zip(restored.error_log, orig.error_log):
            assert r_err.knowledge_point_id == o_err.knowledge_point_id
            assert r_err.error_type == o_err.error_type

    def test_roundtrip_preserves_review_queue(self):
        orig = make_demo_profile()
        orig.schedule_review("math.trig.sin", MasteryLevel.PRACTICING)
        orig.schedule_review("math.trig.cos", MasteryLevel.PROFICIENT)
        restored = StudentProfile.from_serializable(orig.to_serializable())

        assert len(restored.review_queue) == len(orig.review_queue)
        restored_kps = {r["kp_id"] for r in restored.review_queue}
        orig_kps = {r["kp_id"] for r in orig.review_queue}
        assert restored_kps == orig_kps

    def test_empty_profile_roundtrip(self):
        orig = StudentProfile(student_id="empty")
        restored = StudentProfile.from_serializable(orig.to_serializable())
        assert restored.student_id == "empty"
        assert restored.knowledge_graph == {}
        assert restored.error_log == []
        assert restored.review_queue == []


# ── 掌握度更新 ─────────────────────────────────────

class TestMasteryUpdate:
    def test_interaction_count_increments(self):
        p = StudentProfile(student_id="s1")
        p.update_mastery("kp1", MasteryLevel.INTRODUCED, 0.5)
        assert p.knowledge_graph["kp1"].interaction_count == 1
        p.update_mastery("kp1", MasteryLevel.PRACTICING, 0.7)
        assert p.knowledge_graph["kp1"].interaction_count == 2

    def test_confidence_updates(self):
        p = StudentProfile(student_id="s1")
        p.update_mastery("kp1", MasteryLevel.INTRODUCED, 0.4)
        assert p.knowledge_graph["kp1"].confidence == 0.4
        p.update_mastery("kp1", MasteryLevel.PRACTICING, 0.8)
        assert p.knowledge_graph["kp1"].confidence == 0.8

    def test_last_reviewed_set(self):
        p = StudentProfile(student_id="s1")
        before = datetime.now().isoformat()
        p.update_mastery("kp1", MasteryLevel.INTRODUCED, 0.5)
        assert p.knowledge_graph["kp1"].last_reviewed is not None
        assert p.knowledge_graph["kp1"].last_reviewed >= before

    def test_get_or_create_reuses_existing(self):
        p = StudentProfile(student_id="s1")
        n1 = p.get_or_create_node("kp1", name="n1", category="math")
        n2 = p.get_or_create_node("kp1", name="n2", category="physics")
        assert n1 is n2
        assert n2.name == "n1"  # 已存在的不会被覆盖


# ── 错误记录 ───────────────────────────────────────

class TestErrorLogging:
    def test_add_error_appends_and_increments(self):
        p = StudentProfile(student_id="s1")
        p.add_error("kp1", "计算错误", "1+1=?")
        assert len(p.error_log) == 1
        assert p.error_log[0].knowledge_point_id == "kp1"
        assert p.error_log[0].error_type == "计算错误"
        assert p.knowledge_graph["kp1"].error_count == 1

    def test_multiple_errors_accumulate(self):
        p = StudentProfile(student_id="s1")
        p.add_error("kp1", "概念混淆", "q1")
        p.add_error("kp1", "计算错误", "q2")
        p.add_error("kp2", "审题错误", "q3")
        assert len(p.error_log) == 3
        assert p.knowledge_graph["kp1"].error_count == 2
        assert p.knowledge_graph["kp2"].error_count == 1

    def test_question_snippet_truncated(self):
        p = StudentProfile(student_id="s1")
        long_q = "x" * 300
        p.add_error("kp1", "审题错误", long_q)
        assert len(p.error_log[0].question_snippet) == 200


# ── 复习调度 ───────────────────────────────────────

class TestReviewScheduling:
    def test_intervals_by_mastery(self):
        p = StudentProfile(student_id="s1")
        intervals = {
            MasteryLevel.INTRODUCED: 1,
            MasteryLevel.PRACTICING: 3,
            MasteryLevel.PROFICIENT: 7,
            MasteryLevel.MASTERED: 30,
        }
        now = datetime.now()
        for level, days in intervals.items():
            p.schedule_review(f"kp_{level.value}", level)
            entry = next(r for r in p.review_queue if r["kp_id"] == f"kp_{level.value}")
            due = datetime.fromisoformat(entry["due_date"])
            delta = (due - now).days
            assert delta == days, f"{level.value}: expected {days}, got {delta}"

    def test_schedule_replaces_old_entry(self):
        p = StudentProfile(student_id="s1")
        p.schedule_review("kp1", MasteryLevel.INTRODUCED)
        assert len(p.review_queue) == 1
        p.schedule_review("kp1", MasteryLevel.PRACTICING)
        assert len(p.review_queue) == 1
        assert p.review_queue[0]["mastery_at_schedule"] == "practicing"

    def test_get_due_reviews_filters_correctly(self):
        p = StudentProfile(student_id="s1")
        # 手动注入已过期和未过期的条目
        p.review_queue = [
            {"kp_id": "overdue", "due_date": (datetime.now() - timedelta(days=1)).isoformat(), "mastery_at_schedule": "introduced"},
            {"kp_id": "future",  "due_date": (datetime.now() + timedelta(days=7)).isoformat(), "mastery_at_schedule": "practicing"},
        ]
        due = p.get_due_reviews()
        due_ids = {r["kp_id"] for r in due}
        assert "overdue" in due_ids
        assert "future" not in due_ids

    def test_no_reviews_returns_empty(self):
        p = StudentProfile(student_id="s1")
        assert p.get_due_reviews() == []


# ── 薄弱点与强项 ────────────────────────────────────

class TestWeakPointsAndStrengths:
    def test_weak_points_order(self):
        p = StudentProfile(student_id="s1")
        p.update_mastery("a", MasteryLevel.INTRODUCED, 0.5)
        p.update_mastery("b", MasteryLevel.INTRODUCED, 0.5)
        p.update_mastery("c", MasteryLevel.INTRODUCED, 0.5)
        p.add_error("a", "概念混淆", "q1")
        p.add_error("a", "计算错误", "q2")
        p.add_error("b", "审题错误", "q3")
        p.add_error("c", "逻辑错误", "q4")
        p.add_error("c", "计算错误", "q5")
        p.add_error("c", "审题错误", "q6")
        # error_count: a=2, b=1, c=3 → 排序应为 c, a, b
        weak = p.get_weak_points()
        assert weak[0].id == "c"
        assert weak[1].id == "a"
        assert weak[2].id == "b"

    def test_strengths_only_returns_proficient_or_mastered(self):
        p = StudentProfile(student_id="s1")
        p.update_mastery("a", MasteryLevel.MASTERED, 0.9)
        p.update_mastery("b", MasteryLevel.PROFICIENT, 0.8)
        p.update_mastery("c", MasteryLevel.PRACTICING, 0.7)
        p.update_mastery("d", MasteryLevel.INTRODUCED, 0.3)
        strengths = p.get_strengths()
        ids = {n.id for n in strengths}
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids
        assert "d" not in ids

    def test_weak_points_respects_max_count(self):
        p = StudentProfile(student_id="s1")
        for i in range(10):
            p.add_error(f"kp{i}", "计算错误", f"q{i}")
        assert len(p.get_weak_points(max_count=3)) == 3


# ── 风格推断 ───────────────────────────────────────

class TestStyleInference:
    def test_update_same_style_increases_confidence(self):
        p = StudentProfile(student_id="s1", learning_style=LearningStyle.VISUAL, style_confidence=0.4)
        p.update_style_signal(LearningStyle.VISUAL, weight=0.1)
        assert p.style_confidence == 0.5
        assert p.learning_style == LearningStyle.VISUAL

    def test_low_confidence_switches_style(self):
        p = StudentProfile(student_id="s1", learning_style=LearningStyle.VISUAL, style_confidence=0.1)
        p.update_style_signal(LearningStyle.EXAMPLE_DRIVEN, weight=0.1)
        assert p.learning_style == LearningStyle.EXAMPLE_DRIVEN
        assert p.style_confidence == 0.1

    def test_high_confidence_does_not_switch(self):
        p = StudentProfile(student_id="s1", learning_style=LearningStyle.VISUAL, style_confidence=0.8)
        p.update_style_signal(LearningStyle.EXAMPLE_DRIVEN, weight=0.1)
        assert p.learning_style == LearningStyle.VISUAL

    def test_confidence_capped_at_one(self):
        p = StudentProfile(student_id="s1", learning_style=LearningStyle.VISUAL, style_confidence=0.95)
        p.update_style_signal(LearningStyle.VISUAL, weight=0.2)
        assert p.style_confidence == 1.0


# ── 进度摘要 ───────────────────────────────────────

class TestProgressSummary:
    def test_counts(self):
        p = make_demo_profile()
        summary = p.progress_summary()
        assert summary["total_knowledge_points"] == 3
        assert summary["mastered_count"] == 1  # cos is PROFICIENT
        assert summary["total_errors"] == 2
        assert summary["total_interactions"] == 0  # make_demo 不增 total_interactions

    def test_empty_profile_summary(self):
        p = StudentProfile(student_id="empty")
        summary = p.progress_summary()
        assert summary["total_knowledge_points"] == 0
        assert summary["mastered_count"] == 0
        assert summary["mastery_ratio"] == 0.0


# ── StudentProfileManager ───────────────────────────

class TestProfileManager:
    def test_creates_new_profile_when_not_cached(self):
        mgr = StudentProfileManager()
        profile = mgr.get_profile("new_student")
        assert profile.student_id == "new_student"
        assert profile.learning_style == LearningStyle.UNKNOWN

    def test_returns_cached_profile(self):
        mgr = StudentProfileManager()
        p1 = mgr.get_profile("s1")
        p2 = mgr.get_profile("s1")
        assert p1 is p2

    def test_record_interaction_increments(self):
        mgr = StudentProfileManager()
        profile = mgr.get_profile("s1")
        assert profile.total_interactions == 0
        mgr.record_interaction("s1")
        assert profile.total_interactions == 1
