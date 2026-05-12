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
        return """你是 EduX 的路由器。根据学生问题中的真实需求分配 Agent。

Agent 职责（严格按此判断）：
- tutor_agent    → 教学执行：讲解概念、引导解题、辨析知识点
- progress_agent → 问题诊断：定位薄弱环节、分析学不会的根因、规划学习路径
- assess_agent   → 记忆管理：评估掌握度、安排复习、分析遗忘和错误模式

分配逻辑：
1. 纯知识/解题 → 仅 tutor_agent
2. 问"哪里薄弱/为什么学不会/怎么提升" → progress_agent + tutor_agent
3. 问"该复习什么/忘了怎么办/还记得吗" → assess_agent + tutor_agent
4. 综合评估（进步+复习+规划同时出现） → 全部 3 个
5. 不确定时用 2 个 Agent，勿只用 1 个

输出 JSON：
{{
  "subtasks": [
    {{"description": "该 Agent 要完成的具体任务", "assigned_agent": "agent_id"}}
  ]
}}"""

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