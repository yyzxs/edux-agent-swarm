"""
Personalization Engine — 个性化引擎

三大核心算法：
1. 学习风格检测 — 从互动模式推断学生风格
2. 难度自适应校准 — 根据反馈动态调节
3. 间隔复习调度 — 基于艾宾浩斯遗忘曲线
"""
import re
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger

from memory.student_profile import (
    StudentProfile, LearningStyle, MasteryLevel, DifficultyLevel
)


# ===== 1. 学习风格检测 =====

STYLE_INDICATORS = {
    LearningStyle.VISUAL: [
        r"(画|图|看|图解|示意|可视化|表格|图表|曲线|图像)",
        r"(看不懂|看不明白|看图)",
    ],
    LearningStyle.TEXTUAL: [
        r"(读|文字|定义|课本|书本|笔记|讲义|教材)",
        r"(再看看|再读|再看一遍|记笔记)",
    ],
    LearningStyle.EXAMPLE_DRIVEN: [
        r"(例子|例题|举例|比如|具体|实际|演示|实例|示范)",
        r"(不会做|做不出来|怎么做|求步骤)",
    ],
    LearningStyle.ABSTRACT: [
        r"(为什么|原理|推导|证明|逻辑|本质|根本|原因|公式推导)",
        r"(怎么来的|如何证明|为什么这样)",
    ],
    LearningStyle.HANDS_ON: [
        r"(练习|做题|练习册|试卷|考试|测验|刷题|练一练)",
        r"(给我出题|来几道|做一下|练练)",
    ],
}


def detect_style_from_text(text: str) -> Optional[LearningStyle]:
    """从学生消息中检测学习风格信号"""
    scores = {}
    for style, patterns in STYLE_INDICATORS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            score += len(matches)
        if score > 0:
            scores[style] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def detect_style_from_history(profile: StudentProfile) -> LearningStyle:
    """从历史互动模式推断学习风格"""
    if not profile.session_history:
        return LearningStyle.UNKNOWN

    # 统计各类型错误和学习模式
    visual_match = sum(1 for e in profile.error_log if e.error_type == "概念混淆")
    hands_on_need = sum(1 for e in profile.error_log if e.error_type in ("计算错误", "审题错误"))

    if hands_on_need > visual_match * 2:
        return LearningStyle.HANDS_ON
    elif visual_match > hands_on_need * 2:
        return LearningStyle.VISUAL

    return LearningStyle.UNKNOWN


def choose_teaching_approach(profile: StudentProfile, topic: str) -> Dict[str, Any]:
    """
    根据学生画像选择最佳教学方式

    返回教学方法、例子类型、提问风格、节奏建议
    """
    style = profile.learning_style

    approaches = {
        LearningStyle.VISUAL: {
            "method": "图解优先",
            "description": f"先用图形/表格展示{topic}的核心概念，再辅以文字说明",
            "example_type": "可视化示例（图、表、思维导图）",
            "question_style": "引导观察图形特征",
        },
        LearningStyle.TEXTUAL: {
            "method": "文本精读",
            "description": f"提供清晰的文字定义和分步骤解释{topic}",
            "example_type": "文字推理题",
            "question_style": "引导阅读关键定义",
        },
        LearningStyle.EXAMPLE_DRIVEN: {
            "method": "例题驱动",
            "description": f"从一个简单但典型的{topic}例题开始，逐步抽象出规律",
            "example_type": "递进式例题（易→中→难）",
            "question_style": "从具体例子出发提问",
        },
        LearningStyle.ABSTRACT: {
            "method": "原理推导",
            "description": f"从{topic}的基本原理/公式出发，推导得出结论",
            "example_type": "原理验证题",
            "question_style": "追问'为什么'以深化理解",
        },
        LearningStyle.HANDS_ON: {
            "method": "练习中学习",
            "description": f"给出{topic}相关习题，在练习中发现问题并讲解",
            "example_type": "分步骤练习题",
            "question_style": "在解题过程中提问引导",
        },
        LearningStyle.UNKNOWN: {
            "method": "混合试探",
            "description": f"先用例题引入{topic}，观察学生反应后调整",
            "example_type": "结合例题和概念讲解",
            "question_style": "试探性提问（观察偏好）",
        },
    }
    return approaches.get(style, approaches[LearningStyle.UNKNOWN])


# ===== 2. 难度自适应校准 =====

def calibrate_difficulty(
    profile: StudentProfile,
    feedback: Optional[DifficultyLevel] = None,
    error_ratio: Optional[float] = None
) -> str:
    """
    根据反馈动态调整教学节奏

    返回：pace — "slow" / "normal" / "fast"
    """
    current_pace = profile.preferred_pace

    if feedback:
        if feedback == DifficultyLevel.TOO_EASY and current_pace != "fast":
            return "fast"
        elif feedback == DifficultyLevel.TOO_HARD and current_pace != "slow":
            return "slow"
        elif feedback == DifficultyLevel.JUST_RIGHT:
            return current_pace

    if error_ratio is not None:
        if error_ratio > 0.5:
            return "slow"
        elif error_ratio < 0.15 and current_pace == "slow":
            return "normal"

    return current_pace


def estimate_optimal_session_length(profile: StudentProfile) -> int:
    """
    估算最佳单次学习时长（分钟）

    基于：年级、注意力数据、历史会话时长
    """
    base = profile.avg_attention_span or 15

    # 年级调整
    grade_map = {
        "小学": base - 5, "初一": base, "初二": base + 5,
        "初三": base + 10, "高一": base + 5, "高二": base + 10,
        "高三": base + 15, "大学": base + 20
    }
    for key, val in grade_map.items():
        if profile.grade_level and key in profile.grade_level:
            base = max(10, min(60, val))
            break

    profile.optimal_session_length = base
    return base


# ===== 3. 艾宾浩斯间隔复习调度 =====

EBBINGHAUS_SCHEDULE = {
    # (掌握度, 正确率): (间隔天数, 复习优先级)
    (MasteryLevel.INTRODUCED, "low"):     (1, "high"),
    (MasteryLevel.INTRODUCED, "medium"):  (1, "high"),
    (MasteryLevel.INTRODUCED, "high"):    (2, "high"),
    (MasteryLevel.PRACTICING, "low"):     (2, "high"),
    (MasteryLevel.PRACTICING, "medium"):  (3, "medium"),
    (MasteryLevel.PRACTICING, "high"):    (5, "medium"),
    (MasteryLevel.PROFICIENT, "low"):     (4, "medium"),
    (MasteryLevel.PROFICIENT, "medium"):  (7, "low"),
    (MasteryLevel.PROFICIENT, "high"):    (14, "low"),
    (MasteryLevel.MASTERED, "medium"):    (30, "low"),
    (MasteryLevel.MASTERED, "high"):      (60, "minimal"),
}


def compute_review_schedule(
    profile: StudentProfile,
    kp_id: str
) -> Dict[str, Any]:
    """
    计算单个知识点的复习计划

    返回：下次复习日期、优先级、建议复习方式
    """
    node = profile.knowledge_graph.get(kp_id)
    if not node:
        return {"kp_id": kp_id, "status": "not_in_graph"}

    # 根据掌握度和最近正确率确定间隔
    error_ratio = node.error_count / max(node.interaction_count, 1)
    if error_ratio > 0.5:
        performance = "low"
    elif error_ratio > 0.2:
        performance = "medium"
    else:
        performance = "high"

    key = (node.mastery, performance)
    interval_days, priority = EBBINGHAUS_SCHEDULE.get(
        key, (1, "high")
    )

    next_review = datetime.now() + timedelta(days=interval_days)
    node.next_review_due = next_review.isoformat()

    # 复习方式建议
    review_methods = {
        "high": "active_recall",      # 主动回忆（不给提示，要求学生复述）
        "medium": "guided_practice",  # 引导练习（给提示，做相关题）
        "low": "light_review",        # 轻复习（看一眼公式/定义即可）
        "minimal": "check_in",        # 确认式（问一句"还记得吗"）
    }

    return {
        "kp_id": kp_id,
        "kp_name": node.name,
        "mastery": node.mastery.value,
        "next_review": next_review.isoformat(),
        "interval_days": interval_days,
        "priority": priority,
        "review_method": review_methods.get(priority, "guided_practice"),
    }


def get_todays_review_plan(profile: StudentProfile) -> List[Dict[str, Any]]:
    """获取今天所有需要复习的知识点"""
    due = profile.get_due_reviews()
    plan = []
    for item in due:
        detail = compute_review_schedule(profile, item["kp_id"])
        plan.append(detail)
    return sorted(plan, key=lambda x: {"high": 0, "medium": 1, "low": 2, "minimal": 3}.get(x["priority"], 99))


# ===== 4. 互动节奏控制 =====

def should_ask_question(profile: StudentProfile, turn_count: int) -> bool:
    """决定当前轮是否应该向学生提问（而非继续单向讲解）"""
    # 每3轮互动至少提一个问题
    if turn_count % 3 == 0:
        return True
    # 如果连续2轮都是讲解，下一轮提问
    return False


def choose_question_type(profile: StudentProfile, kp_id: str) -> str:
    """选择提问类型"""
    node = profile.knowledge_graph.get(kp_id)
    if not node:
        return "concept_check"  # 概念确认

    if node.mastery == MasteryLevel.INTRODUCED:
        return "recall"          # 回忆型
    elif node.mastery == MasteryLevel.PRACTICING:
        return "application"     # 应用型
    elif node.mastery in (MasteryLevel.PROFICIENT, MasteryLevel.MASTERED):
        return "extension"       # 拓展型

    return "concept_check"


def detect_emotional_state(text: str) -> Optional[str]:
    """简单的情绪检测"""
    frustration_keywords = ["不会", "太难", "不懂", "完全不明白", "救命", "放弃了", "崩溃"]
    confidence_keywords = ["懂了", "明白了", "简单", "会了", "原来如此", "so easy"]
    engagement_keywords = ["那如果", "还有", "继续", "再来", "下一个"]

    if any(kw in text for kw in frustration_keywords):
        return "frustrated"
    if any(kw in text for kw in confidence_keywords):
        return "confident"
    if any(kw in text for kw in engagement_keywords):
        return "engaged"
    return None
