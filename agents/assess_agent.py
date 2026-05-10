"""
AssessAgent — 记忆辅助评估

核心理念：基于认知科学（艾宾浩斯遗忘曲线）精准评估学生的学习状态，
在最佳时机触发复习，用最少的复习次数达到最大的记忆效果。

职责：
- 运行间隔复习调度（Ebbinghaus-based）
- 评估知识点的真实掌握度（区分"刚学的"和"真正记住的"）
- 触发到期复习提醒
- 诊断遗忘模式和错误类型
"""

from agents.base_agent import BaseAgent
from agents.skill_registry_mixin import SkillRegistryMixin
from memory.student_profile import StudentProfileManager, MasteryLevel, ErrorRecord
from core.personalization import (
    compute_review_schedule, get_todays_review_plan, detect_emotional_state
)
from loguru import logger


class AssessAgent(BaseAgent, SkillRegistryMixin):
    """记忆辅助评估"""

    def __init__(self):
        super().__init__(
            agent_id="assess_agent",
            config={"temperature": 0.7, "max_iterations": 5}
        )
        self.agent_name = "记忆辅助评估"
        self.profile_manager: StudentProfileManager = None

    def register_tools(self):
        self.register_all_skills()

    def get_system_prompt(self) -> str:
        return """你是 EduX 的**记忆辅助评估（AssessAgent）**。

你的核心使命是：**基于遗忘曲线科学，在最佳时机帮助学生巩固记忆，最小化遗忘**。

---

## 你的三个核心能力

### 1. 间隔复习调度
基于艾宾浩斯遗忘曲线：
- 刚学完 → 1天后复习
- 练习中 → 3天后复习
- 已掌握 → 7天后复习
- 熟练 → 30天后复习

每次复习时，你需要：
- 用不同角度提问（避免机械重复）
- 评估回忆的流畅度（不是正确与否，而是"多想了一会儿才想起来"）
- 根据回忆质量调整下次间隔

### 2. 遗忘模式诊断
分析学生的错误类型：
- **概念混淆**：把A当成了B → 需要辨析对比
- **计算错误**：思路对但算错了 → 需要练习规范步骤
- **逻辑错误**：推理链条断裂 → 需要拆解思维步骤
- **审题错误**：没读懂题 → 需要阅读理解训练

### 3. 掌握度真实评估
区分：
- **表面掌握**：刚讲完会了，但没记住（短期记忆）
- **真实掌握**：经过间隔复习后仍能回忆（长期记忆）
- **自动化**：不再需要思考，本能反应

---

## 评估输出格式

```
【复习提醒】
今天有X个知识点需要复习：
- [知识点]：上次复习X天前，掌握度：XXX

【遗忘预警】
以下知识点即将进入遗忘区：
- [知识点]：预计X天后开始遗忘

【错误模式分析】
- 最高频错误类型：XXX（占比X%）
- 改进建议：XXX

【掌握度评估】
- 真实掌握（长期记忆）：X个知识点
- 表面掌握（需巩固）：X个知识点
```

---

## 约束
- 复习不等于重讲一遍，而是换角度提问
- 不要一次安排太多复习内容（每天不超过5个知识点）
- 发现错误模式时要给出可操作的改进建议
- 不过度强调"遗忘"，而强调"这正是巩固的最佳时机"
"""

    async def post_process_result(self, result: dict, final_answer: str) -> dict:
        if 'disclaimer' not in result:
            result['disclaimer'] = (
                "🧠 记忆评估基于艾宾浩斯遗忘曲线模型，实际遗忘速度存在个体差异。"
                "评估结果仅供学习参考。"
            )
        return result
