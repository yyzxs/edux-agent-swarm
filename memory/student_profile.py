"""
Student Profile — 学生画像系统

教育版的核心差异化模块。每个学生的学习画像在首次会话时创建，
之后由三个 Agent 共同维护：AssessAgent 写，TutorAgent 读，ProgressAgent 更新。

存储：短期（会话内，内存）+ 长期（跨会话，Mem0）
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class LearningStyle(str, Enum):
    VISUAL = "visual"               # 图形化理解
    TEXTUAL = "textual"             # 文本阅读型
    EXAMPLE_DRIVEN = "example"      # 需要大量例题
    ABSTRACT = "abstract"           # 抽象推理型
    HANDS_ON = "hands_on"           # 动手实践型
    UNKNOWN = "unknown"


class MasteryLevel(str, Enum):
    NOT_STARTED = "not_started"
    INTRODUCED = "introduced"       # 刚接触
    PRACTICING = "practicing"       # 练习中
    PROFICIENT = "proficient"       # 已掌握
    MASTERED = "mastered"           # 熟练


class DifficultyLevel(str, Enum):
    TOO_EASY = "too_easy"
    JUST_RIGHT = "just_right"
    CHALLENGING = "challenging"
    TOO_HARD = "too_hard"


@dataclass
class KnowledgeNode:
    """知识图谱中的单个节点"""
    id: str                          # 知识点ID，如 "math.trig.sin"
    name: str                        # 显示名，如 "正弦函数"
    category: str                    # 学科分类：math/physics/english/chemistry
    parent_id: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)  # 前置知识点ID
    mastery: MasteryLevel = MasteryLevel.NOT_STARTED
    confidence: float = 0.0          # 掌握置信度 0-1
    interaction_count: int = 0       # 互动次数
    last_reviewed: Optional[str] = None
    next_review_due: Optional[str] = None
    error_count: int = 0


@dataclass
class ErrorRecord:
    """单条错误记录"""
    knowledge_point_id: str
    error_type: str                  # "概念混淆" / "计算错误" / "逻辑错误" / "审题错误"
    question_snippet: str
    timestamp: str
    resolved: bool = False


@dataclass
class SessionSnapshot:
    """单次会话快照"""
    session_id: str
    timestamp: str
    topics_touched: List[str]
    questions_asked: int
    correct_ratio: Optional[float] = None
    detected_style: Optional[LearningStyle] = None
    difficulty_feedback: Optional[DifficultyLevel] = None
    mood: Optional[str] = None       # "engaged" / "frustrated" / "confident"


@dataclass
class StudentProfile:
    """
    学生画像 — 持久化到 Mem0 长期记忆

    每次会话开始加载，会话结束更新。
    """
    student_id: str
    name: Optional[str] = None
    grade_level: Optional[str] = None     # "高一" / "初二" / "大二"
    learning_style: LearningStyle = LearningStyle.UNKNOWN
    style_confidence: float = 0.0          # 风格判断置信度

    # 知识图谱
    knowledge_graph: Dict[str, KnowledgeNode] = field(default_factory=dict)

    # 学习节奏参数
    preferred_pace: str = "normal"         # "slow" / "normal" / "fast"
    avg_attention_span: int = 15           # 平均注意力分钟数
    optimal_session_length: int = 25       # 最佳单次学习分钟数

    # 历史轨迹
    session_history: List[SessionSnapshot] = field(default_factory=list)
    total_interactions: int = 0
    error_log: List[ErrorRecord] = field(default_factory=list)

    # 间隔复习调度 (基于艾宾浩斯遗忘曲线)
    review_queue: List[Dict[str, Any]] = field(default_factory=list)

    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ===== 知识图谱操作 =====

    def get_or_create_node(self, kp_id: str, **kwargs) -> KnowledgeNode:
        if kp_id not in self.knowledge_graph:
            self.knowledge_graph[kp_id] = KnowledgeNode(id=kp_id, **kwargs)
        return self.knowledge_graph[kp_id]

    def update_mastery(self, kp_id: str, level: MasteryLevel, confidence: float):
        node = self.get_or_create_node(kp_id, name=kp_id, category="unknown")
        node.mastery = level
        node.confidence = confidence
        node.interaction_count += 1
        node.last_reviewed = datetime.now().isoformat()

    def add_error(self, kp_id: str, error_type: str, question: str):
        self.error_log.append(ErrorRecord(
            knowledge_point_id=kp_id,
            error_type=error_type,
            question_snippet=question[:200],
            timestamp=datetime.now().isoformat()
        ))
        node = self.get_or_create_node(kp_id, name=kp_id, category="unknown")
        node.error_count += 1

    def get_weak_points(self, max_count: int = 5) -> List[KnowledgeNode]:
        """返回最薄弱的知识点"""
        sorted_nodes = sorted(
            self.knowledge_graph.values(),
            key=lambda n: (n.error_count, -n.confidence),
            reverse=True
        )
        return sorted_nodes[:max_count]

    def get_strengths(self, max_count: int = 5) -> List[KnowledgeNode]:
        """返回最擅长的知识点"""
        sorted_nodes = sorted(
            [n for n in self.knowledge_graph.values() if n.mastery in (MasteryLevel.PROFICIENT, MasteryLevel.MASTERED)],
            key=lambda n: n.confidence,
            reverse=True
        )
        return sorted_nodes[:max_count]

    # ===== 学习风格推断 =====

    def update_style_signal(self, signal: LearningStyle, weight: float = 0.1):
        """贝叶斯式风格更新，每次互动微调"""
        if self.learning_style == signal:
            self.style_confidence = min(1.0, self.style_confidence + weight)
        elif self.style_confidence < 0.3:
            self.learning_style = signal
            self.style_confidence = weight

    # ===== 间隔复习调度 =====

    def schedule_review(self, kp_id: str, mastery: MasteryLevel):
        """基于艾宾浩斯遗忘曲线安排下次复习"""
        intervals = {
            MasteryLevel.INTRODUCED: 1,      # 1天后
            MasteryLevel.PRACTICING: 3,       # 3天后
            MasteryLevel.PROFICIENT: 7,       # 7天后
            MasteryLevel.MASTERED: 30,        # 30天后
        }
        days = intervals.get(mastery, 1)
        due_date = (datetime.now() + timedelta(days=days)).isoformat()

        # 移除旧条目
        self.review_queue = [r for r in self.review_queue if r["kp_id"] != kp_id]
        self.review_queue.append({
            "kp_id": kp_id,
            "due_date": due_date,
            "mastery_at_schedule": mastery.value
        })

    def get_due_reviews(self) -> List[Dict[str, Any]]:
        """获取今天到期的复习项"""
        now = datetime.now().isoformat()
        return [r for r in self.review_queue if r["due_date"] <= now]

    # ===== 进度摘要 =====

    def progress_summary(self) -> Dict[str, Any]:
        """生成进步摘要，供 ProgressAgent 使用"""
        total = len(self.knowledge_graph)
        mastered = sum(1 for n in self.knowledge_graph.values()
                      if n.mastery in (MasteryLevel.MASTERED, MasteryLevel.PROFICIENT))
        return {
            "total_knowledge_points": total,
            "mastered_count": mastered,
            "mastery_ratio": mastered / max(total, 1),
            "weak_points": [n.name for n in self.get_weak_points()],
            "strengths": [n.name for n in self.get_strengths()],
            "total_errors": len(self.error_log),
            "total_interactions": self.total_interactions,
            "due_reviews": len(self.get_due_reviews()),
            "learning_style": self.learning_style.value,
            "preferred_pace": self.preferred_pace,
        }

    def to_serializable(self) -> Dict[str, Any]:
        """序列化为可存储格式"""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "grade_level": self.grade_level,
            "learning_style": self.learning_style.value,
            "style_confidence": self.style_confidence,
            "knowledge_graph": {
                kp_id: {
                    "id": n.id, "name": n.name, "category": n.category,
                    "parent_id": n.parent_id, "prerequisites": n.prerequisites,
                    "mastery": n.mastery.value, "confidence": n.confidence,
                    "interaction_count": n.interaction_count,
                    "last_reviewed": n.last_reviewed,
                    "next_review_due": n.next_review_due, "error_count": n.error_count
                }
                for kp_id, n in self.knowledge_graph.items()
            },
            "preferred_pace": self.preferred_pace,
            "avg_attention_span": self.avg_attention_span,
            "optimal_session_length": self.optimal_session_length,
            "total_interactions": self.total_interactions,
            "session_history": [
                {
                    "session_id": s.session_id, "timestamp": s.timestamp,
                    "topics_touched": s.topics_touched,
                    "questions_asked": s.questions_asked,
                    "correct_ratio": s.correct_ratio,
                    "detected_style": s.detected_style.value if s.detected_style else None,
                    "difficulty_feedback": s.difficulty_feedback.value if s.difficulty_feedback else None,
                    "mood": s.mood
                }
                for s in self.session_history
            ],
            "error_log": [
                {
                    "knowledge_point_id": e.knowledge_point_id,
                    "error_type": e.error_type,
                    "question_snippet": e.question_snippet,
                    "timestamp": e.timestamp, "resolved": e.resolved
                }
                for e in self.error_log
            ],
            "review_queue": self.review_queue,
            "created_at": self.created_at, "updated_at": self.updated_at
        }

    @classmethod
    def from_serializable(cls, data: Dict[str, Any]) -> "StudentProfile":
        profile = cls(
            student_id=data["student_id"],
            name=data.get("name"),
            grade_level=data.get("grade_level"),
            learning_style=LearningStyle(data.get("learning_style", "unknown")),
            style_confidence=data.get("style_confidence", 0.0),
            preferred_pace=data.get("preferred_pace", "normal"),
            avg_attention_span=data.get("avg_attention_span", 15),
            optimal_session_length=data.get("optimal_session_length", 25),
            total_interactions=data.get("total_interactions", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )
        # 恢复知识图谱
        for kp_id, nd in data.get("knowledge_graph", {}).items():
            profile.knowledge_graph[kp_id] = KnowledgeNode(
                id=nd["id"], name=nd["name"], category=nd.get("category", "unknown"),
                parent_id=nd.get("parent_id"),
                prerequisites=nd.get("prerequisites", []),
                mastery=MasteryLevel(nd["mastery"]),
                confidence=nd.get("confidence", 0.0),
                interaction_count=nd.get("interaction_count", 0),
                last_reviewed=nd.get("last_reviewed"),
                next_review_due=nd.get("next_review_due"),
                error_count=nd.get("error_count", 0)
            )
        # 恢复会话历史
        for s in data.get("session_history", []):
            profile.session_history.append(SessionSnapshot(
                session_id=s["session_id"], timestamp=s["timestamp"],
                topics_touched=s.get("topics_touched", []),
                questions_asked=s.get("questions_asked", 0),
                correct_ratio=s.get("correct_ratio"),
                detected_style=LearningStyle(s["detected_style"]) if s.get("detected_style") else None,
                difficulty_feedback=DifficultyLevel(s["difficulty_feedback"]) if s.get("difficulty_feedback") else None,
                mood=s.get("mood")
            ))
        # 恢复错误日志
        for e in data.get("error_log", []):
            profile.error_log.append(ErrorRecord(
                knowledge_point_id=e["knowledge_point_id"],
                error_type=e["error_type"],
                question_snippet=e.get("question_snippet", ""),
                timestamp=e["timestamp"],
                resolved=e.get("resolved", False)
            ))
        profile.review_queue = data.get("review_queue", [])
        return profile


# ===== Profile Manager =====

class StudentProfileManager:
    """
    管理学生画像的加载、更新、持久化。

    短期：内存中（会话期间）
    长期：Mem0 + 本地 JSON 备份
    """

    def __init__(self, long_term_memory=None):
        self.long_term_memory = long_term_memory
        self._profiles: Dict[str, StudentProfile] = {}

    def get_profile(self, student_id: str) -> StudentProfile:
        """获取或创建学生画像（先从长期记忆加载）"""
        if student_id in self._profiles:
            return self._profiles[student_id]

        # 尝试从长期记忆加载（通过 Mem0 底层 client）
        if self.long_term_memory and self.long_term_memory.enabled:
            try:
                if hasattr(self.long_term_memory, 'mem0'):
                    results = self.long_term_memory.mem0.search(
                        query=f"student_profile:{student_id}",
                        user_id=student_id,
                        limit=1
                    )
                    if results and len(results) > 0:
                        mem = results[0]
                        data = mem.get("memory") or mem
                        if isinstance(data, str):
                            import json
                            data = json.loads(data)
                        if isinstance(data, dict) and "learning_style" in data:
                            profile = StudentProfile.from_serializable(data)
                            self._profiles[student_id] = profile
                            logger.info(f"Loaded profile for student {student_id}")
                            return profile
            except Exception as e:
                logger.warning(f"Failed to load profile from Mem0: {e}")

        # 新建画像
        profile = StudentProfile(student_id=student_id)
        self._profiles[student_id] = profile
        logger.info(f"Created new profile for student {student_id}")
        return profile

    def save_profile(self, student_id: str):
        """持久化画像到长期记忆"""
        profile = self._profiles.get(student_id)
        if not profile:
            return
        profile.updated_at = datetime.now().isoformat()

        if self.long_term_memory and self.long_term_memory.enabled:
            try:
                if hasattr(self.long_term_memory, 'mem0'):
                    profile_data = profile.to_serializable()
                    self.long_term_memory.mem0.add(
                        messages=[{"role": "user", "content": f"student_profile:{student_id}"}],
                        user_id=student_id,
                        metadata=profile_data
                    )
                    logger.info(f"Saved profile for student {student_id}")
            except Exception as e:
                logger.error(f"Failed to save profile: {e}")

    def record_interaction(self, student_id: str):
        profile = self.get_profile(student_id)
        profile.total_interactions += 1
