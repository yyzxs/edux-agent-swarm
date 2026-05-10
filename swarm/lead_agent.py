"""
EduX LeadAgent — 教育版任务分解与结果汇总

将原医疗版的关键词/模式改写为教育领域：
  症状 → 学习问题
  疾病 → 知识点/学科
  就医 → 求助教师
"""
import uuid
from typing import Dict, Any, List, Optional
from loguru import logger

from core.llm_client import LLMClient
from swarm.shared_context import SharedContext, SubTask


class LeadAgent:
    """教育版 Lead Agent — 任务协调者"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.agent_id = "lead_agent"
        self.llm_client = llm_client or LLMClient()

    def _get_system_prompt(self) -> str:
        return """你是 EduX 的 Lead Agent。你的职责是**分析学生的学习需求并分配给最合适的 Agent**。

**核心原则**：
1. **单一知识点讲解 → 1 个 Agent（TutorAgent）**：如"什么是XXX""这道题怎么做"
2. **涉及"分析+给方案" → 2 个 Agent**：如"分析薄弱点+怎么学" → ProgressAgent + TutorAgent
3. **涉及"评估+复习" → 2 个 Agent**：如"我掌握得怎么样+该复习什么" → AssessAgent + ProgressAgent
4. **全面诊断（分析+评估+方案） → 3 个 Agent**：如"为什么学不会+哪里薄弱+怎么规划"
5. **当不确定时，拆成 2 个 Agent 而非 1 个**——多一个 Agent 的分析结果不会有害，但漏掉一个维度会让学生得不到完整帮助
6. 你**只负责分配合适的 Agent**，不决定具体使用哪些工具（各 Agent 自己选择）

---

## 可用的 Agent

### 1. TutorAgent（自适应学习导师）
**擅长**：
- 知识点讲解和概念解释
- 解题思路引导（不是直接给答案）
- 匹配学习风格的教学
- 难度自适应调节

**适用场景**：
- "什么是XXX？"（概念解释）
- "这道题怎么做？"（解题辅导）
- "帮我理解XXX"（知识讲解）
- "XXX和YYY有什么区别？"（概念辨析）

---

### 2. ProgressAgent（学生进步指南）
**擅长**：
- 知识点掌握度分析
- 薄弱环节识别
- 学习路径规划
- 前置依赖分析（"你XXX还不会，所以YYY比较吃力"）

**适用场景**：
- "我学得怎么样了？"
- "还有哪里比较薄弱？"
- "下一步应该学什么？"
- "为什么我总是搞不懂XXX？"

---

### 3. AssessAgent（记忆辅助评估）
**擅长**：
- 遗忘曲线建模和复习调度
- 错误模式分析（概念混淆 / 计算错误 / 逻辑错误）
- 掌握度真实评估（区分短期记忆和长期记忆）

**适用场景**：
- "我该复习什么了？"
- "为什么我总是忘？"
- "上次学的好像又忘了"
- "帮我测试一下我还记不记得"

---

## 任务分配策略

### 策略 1：简单辅导 → 1 个 Agent（TutorAgent）
- 单一知识点讲解
- 解题思路引导
- 概念辨析

### 策略 2：进步评估 → 1-2 个 Agent
- "我学得怎么样" → ProgressAgent
- "该复习什么" → AssessAgent
- "综合评估+建议" → ProgressAgent + AssessAgent

### 策略 3：全面诊断 → 2-3 个 Agent
- 知识体系综合评估
- 学习困难深度分析
- 个性化学习方案制定

---

## 输出格式（JSON）

{{
  "subtasks": [
    {{
      "description": "具体描述该 Agent 需要做什么",
      "assigned_agent": "tutor_agent" | "progress_agent" | "assess_agent"
    }}
  ]
}}
"""

    async def assess_and_decompose(
        self, question: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"学生问题：{question}\n\n背景信息：{context or '无'}"}
        ]

        try:
            content = await self.llm_client.chat(messages)
            logger.debug(f"LeadAgent assessment: {content[:200]}...")

            import json, re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return {
                "subtasks": [{
                    "description": "回答学生的学习问题",
                    "assigned_agent": "tutor_agent"
                }],
                "reason": "无法解析，默认使用 TutorAgent"
            }
        except Exception as e:
            logger.error(f"LeadAgent error: {e}")
            return {"subtasks": [], "reason": f"评估失败：{e}"}

    def create_subtasks(
        self, decomposition_result: Dict[str, Any], shared_context: SharedContext
    ) -> List[SubTask]:
        subtasks_data = decomposition_result.get("subtasks", [])
        subtasks = []
        for data in subtasks_data:
            assigned = data.get("assigned_agent", "tutor_agent")
            subtask = SubTask(
                id=str(uuid.uuid4()),
                type=data.get("type", f"{assigned}_task"),
                description=data["description"],
                assigned_agent=assigned
            )
            shared_context.add_subtask(subtask)
            subtasks.append(subtask)
            logger.info(f"SubTask: {subtask.type} → {subtask.assigned_agent}")
        return subtasks

    async def synthesize_results(
        self, question: str, shared_context: SharedContext,
        timeout_occurred: bool = False
    ) -> str:
        all_contributions = shared_context.get_contributions()

        if not all_contributions:
            if timeout_occurred:
                return """抱歉，系统响应超时，未能完成分析。

【建议】：
- 您的问题可能涉及较多知识点，建议简化后重试
- 或者将问题拆分为多个小问题分别咨询

【联系教师】
如果遇到持续性学习困难，建议与任课老师沟通。"""
            return "抱歉，未能提供有效分析结果。"

        contributions_text = []
        for contrib in all_contributions:
            subtask = shared_context.get_subtask(contrib.subtask_id)
            contributions_text.append(
                f"**{contrib.agent_id}**:\n{contrib.result}"
            )

        timeout_note = ""
        if timeout_occurred:
            timeout_note = "\n\n**注意**：部分分析模块因超时未完成，以下是已完成部分的结果。"

        synthesis_prompt = f"""你是 EduX 的 Lead Agent，汇总多个 Agent 的分析结果。

**学生问题**：{question}

**各 Agent 分析**：
{chr(10).join(contributions_text)}{timeout_note}

**任务**：整合以上分析，生成对学生有实际帮助的综合答复。

**要求**：
1. 整合各 Agent 的视角（教学 + 进步 + 记忆）
2. 给出可执行的下一步建议
3. 保持鼓励性语气
4. 包含【学习建议】或【复习计划】等实用模块
"""

        try:
            response = await self.llm_client.chat([
                {"role": "user", "content": synthesis_prompt}
            ])
            return response
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"汇总结果时出错：{e}"
