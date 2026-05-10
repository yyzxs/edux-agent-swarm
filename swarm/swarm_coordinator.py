"""
SwarmCoordinator：Swarm 入口和智能路由

注意：这不是编排器！
- 只负责路由决策：简单问题 → 单 Agent，复杂问题 → Swarm
- 不控制 Agent 执行
- 不编排任务顺序

类比：交通信号灯，决定车辆走哪条路，但不控制车辆如何行驶
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from core import LLMClient
from .shared_context import SharedContext
from .lead_agent import LeadAgent
from .events import Event, EventType
from agents import TutorAgent, ProgressAgent, AssessAgent
from memory import SessionSummaryManager, SessionSummary, ShortTermMemory, LongTermMemory


class SwarmCoordinator:
    """
    Swarm 协调器

    职责：
    1. 智能路由（简单 → 单 Agent，复杂 → Swarm）
    2. 初始化 SharedContext
    3. 启动和监控 Swarm
    4. 生成 SessionSummary

    不做：
    - 不编排 Worker 执行顺序
    - 不直接调用 Worker
    - 不控制任务分配
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm

        # 初始化 Agent
        self.lead_agent = LeadAgent(llm_client=self.llm_client)
        self.tutor_agent = TutorAgent()
        self.progress_agent = ProgressAgent()
        self.assess_agent = AssessAgent()

        # Worker 池
        self.worker_pool: List[Any] = [
            self.tutor_agent,
            self.progress_agent,
            self.assess_agent
        ]

        # 记忆管理器
        self.session_manager = SessionSummaryManager()
        self.short_term_memory = ShortTermMemory(storage_type="memory")  # 或 "redis"
        self.long_term_memory = LongTermMemory()

        # 将短期记忆注入到所有 Worker Agent 的 Loop
        # 注意：LeadAgent 不继承 BaseAgent，没有 loop 属性，不需要注入
        for worker in self.worker_pool:
            if hasattr(worker, 'loop'):
                worker.loop.short_term_memory = self.short_term_memory

        logger.info(f"SwarmCoordinator initialized with {len(self.worker_pool)} workers")
        logger.info(f"Memory system: short_term={self.short_term_memory.storage_type}, long_term={'enabled' if self.long_term_memory.enabled else 'disabled'}")

    def _get_agent_by_id(self, agent_id: str):
        """根据 agent_id 返回对应的 Agent 实例"""
        mapping = {
            "tutor_agent": self.tutor_agent,
            "progress_agent": self.progress_agent,
            "assess_agent": self.assess_agent
        }
        return mapping.get(agent_id)

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        force_swarm: bool = False
    ) -> Dict[str, Any]:
        """
        处理用户问题

        Args:
            question: 用户问题
            context: 额外上下文
            session_id: 会话ID（如果不提供，将自动生成）
            force_swarm: 强制使用多 Agent Swarm 模式
        """
        start_time = datetime.now()
        if session_id is None:
            session_id = f"{start_time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

        logger.info(f"Processing question (session={session_id}): {question[:50]}...")

        # ===== 统一的记忆检索（所有模式都使用）=====
        # 1. 检索短期记忆（当前会话历史）
        recent_history = self.short_term_memory.get_recent_messages(
            session_id=session_id,
            limit=10  # 最近5轮对话（10条消息）
        )

        # 2. 检索长期记忆（相似历史会话）
        similar_memories = self.long_term_memory.search_similar_sessions(
            query=question,
            limit=3
        )

        # 3. 构建增强上下文
        enhanced_context = context or {}

        # 添加短期记忆
        if recent_history:
            enhanced_context["recent_history"] = [
                {"role": msg.get("role", ""), "content": msg.get("content", "")}
                for msg in recent_history
            ]
            logger.info(f"Loaded {len(recent_history)} recent messages from short-term memory")

        # 添加长期记忆
        if similar_memories:
            enhanced_context["historical_cases"] = [
                {
                    "summary": mem["content"],
                    "score": mem["score"]
                }
                for mem in similar_memories
            ]
            logger.info(f"Found {len(similar_memories)} similar historical cases from long-term memory")

        # Step 1: LeadAgent 分解任务
        assessment = await self.lead_agent.assess_and_decompose(question, enhanced_context)
        subtasks = assessment.get("subtasks", [])

        # Step 1.5: 规则兜底 — LLM 返回 1 个任务但问题明显是综合评估
        FORCE_MULTI_PATTERNS = {
            "progress_check": ["进步", "水平", "掌握", "学到", "薄弱", "知识体系", "全局", "全面评估", "综合评估"],
            "review_planning": ["复习", "遗忘", "忘了", "记忆", "记住", "巩固"],
            "comprehensive": ["查漏补缺", "期末", "摸底", "总结"],
        }
        if len(subtasks) == 1 and self.enable_swarm:
            for mode, keywords in FORCE_MULTI_PATTERNS.items():
                if any(kw in question for kw in keywords):
                    if mode == "review_planning":
                        subtasks = [
                            {"description": f"评估'{question}'中涉及知识点的记忆状态和复习优先级", "assigned_agent": "assess_agent"},
                            {"description": f"基于评估结果给出'{question}'的具体复习方案", "assigned_agent": "tutor_agent"},
                        ]
                    elif mode == "comprehensive":
                        subtasks = [
                            {"description": f"分析'{question}'中涉及的薄弱环节和知识差距", "assigned_agent": "progress_agent"},
                            {"description": f"基于分析结果给出综合提升方案和学习路径", "assigned_agent": "tutor_agent"},
                        ]
                    else:  # progress_check
                        subtasks = [
                            {"description": f"分析'{question}'中涉及知识点的掌握程度和薄弱环节", "assigned_agent": "progress_agent"},
                            {"description": f"给出针对性的学习建议和下一步规划", "assigned_agent": "tutor_agent"},
                        ]
                    logger.info(f"Force multi-agent ({mode}): expanded to {len(subtasks)} subtasks by keyword rule")
                    break

        logger.info(f"LeadAgent 分解任务：{len(subtasks)} 个")

        # Step 2: 根据任务数量路由
        final_answer = None
        mode = None

        if len(subtasks) == 1 and not force_swarm:
            # 单任务 → 直接调用对应 Agent
            task = subtasks[0]
            agent_id = task.get("assigned_agent")
            agent = self._get_agent_by_id(agent_id)

            if agent is None:
                logger.warning(f"Unknown agent_id: {agent_id}, fallback to TutorAgent")
                agent = self.tutor_agent

            logger.info(f"Route: Single Agent ({agent_id})")
            mode = "single_agent"
            result = await agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')

            result.update({
                'swarm_enabled': False,
                'session_id': session_id,
                'route_reason': f'单任务路由到 {agent_id}'
            })

            # 确保单Agent模式下也有 disclaimer 字段
            if 'disclaimer' not in result:
                result['disclaimer'] = "📚 以上内容为学习辅导建议。如遇持续性学习困难，建议与学校老师沟通。"

            # 确保单Agent模式下也有 suggestions 字段
            if 'suggestions' not in result:
                result['suggestions'] = []

        if force_swarm and len(subtasks) < 2:
            # 强制多 Agent：将单任务扩展为多 Agent 协作
            logger.info("Force swarm: expanding to multi-agent")
            subtasks = self._force_multi_subtasks(question, subtasks)
            logger.info(f"Force swarm: expanded to {len(subtasks)} subtasks")

        if len(subtasks) >= 2 and self.enable_swarm:
            # 多任务 → 启动 Swarm
            logger.info(f"Route: Swarm (Multi-Agent Collaboration) - {len(subtasks)} tasks")
            mode = "swarm"
            result = await self._process_with_swarm(
                question=question,
                context=enhanced_context,
                assessment=assessment,
                session_id=session_id,
                start_time=start_time
            )
            final_answer = result.get('answer', '')

            # Swarm 模式已经在 _process_with_swarm 中保存了长期记忆，直接返回
            return result

        else:
            # 0个任务或Swarm关闭 → 降级到 TutorAgent
            if len(subtasks) == 0:
                logger.warning("No subtasks generated, fallback to TutorAgent")
                mode = "fallback"
            else:
                logger.info("Swarm disabled, fallback to TutorAgent")
                mode = "disabled_swarm"

            result = await self.tutor_agent.process({
                'question': question,
                'context': enhanced_context,
                'session_id': session_id
            })
            final_answer = result.get('answer', '')
            result.update({
                'swarm_enabled': False,
                'session_id': session_id
            })

        # ===== 统一的记忆保存（非 Swarm 模式）=====
        end_time = datetime.now()

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存

        # 保存到长期记忆
        try:
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                metadata={
                    "mode": mode,
                    "subtasks_count": len(subtasks),
                    "total_time": (end_time - start_time).total_seconds(),
                }
            )
            logger.info(f"Saved to long-term memory (session={session_id}, mode={mode})")
        except Exception as e:
            logger.error(f"Failed to save to long-term memory: {e}")

        return result

    async def _process_with_swarm(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        assessment: Dict[str, Any],
        session_id: str,
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        使用 Swarm 处理复杂问题

        这是群体智能的核心流程

        注意：context 已经包含了长短期记忆（在 process() 中注入）
        """
        # context 已经包含 recent_history 和 historical_cases
        # 无需重复检索

        # 创建 SharedContext
        shared_context = SharedContext(session_id=session_id)

        # 附加 SharedContext 到所有 Worker
        for worker in self.worker_pool:
            worker.attach_shared_context(shared_context)

        # 发布 Swarm 启动事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="swarm_coordinator",
            data={
                "question": question,
                "num_subtasks": len(assessment.get("subtasks", []))
            }
        ))

        # Step 1: LeadAgent 分解任务
        subtasks = self.lead_agent.create_subtasks(assessment, shared_context)
        logger.info(f"Created {len(subtasks)} subtasks")

        # Step 2: Worker 执行分配的任务（并行）
        import time as _time
        tasks = []
        for worker in self.worker_pool:
            task = asyncio.create_task(
                self._worker_execute_with_timing(worker, shared_context, _time)
            )
            tasks.append(task)

        # 等待所有 Worker 完成（或超时）
        timeout_occurred = False
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=180.0  # 3个Agent并行LLM调用需要更长时间
            )
        except asyncio.TimeoutError:
            timeout_occurred = True
            logger.warning("Swarm execution timeout (180s)")
            # 记录哪些 Agent 已完成，哪些未完成
            completed_agents = list(shared_context.agent_contributions.keys())
            claimed_tasks = [
                (subtask.assigned_to, subtask.type)
                for subtask in shared_context.task_decomposition.values()
                if subtask.status.value == "claimed"
            ]
            logger.info(f"Completed agents: {completed_agents}")
            logger.info(f"Timed out tasks: {claimed_tasks}")

        # Step 3: LeadAgent 汇总结果
        # 即使超时，也尝试汇总已完成的部分结果
        final_answer = await self.lead_agent.synthesize_results(
            question=question,
            shared_context=shared_context,
            timeout_occurred=timeout_occurred
        )

        end_time = datetime.now()

        # Step 4: 生成 SessionSummary
        try:
            summary = SessionSummary.from_shared_context(
                session_id=session_id,
                question=question,
                shared_context=shared_context,
                final_answer=final_answer,
                start_time=start_time,
                end_time=end_time
            )
            self.session_manager.save_summary(summary)
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存
        # Agent Loop 保存了完整的对话历史（user + assistant + tool messages）

        # 保存到 Mem0 长期记忆
        try:
            # 保存会话总结
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                metadata={
                    "mode": "swarm",
                    "agents_count": len(shared_context.agent_contributions),
                    "total_time": (end_time - start_time).total_seconds(),
                    "timeout_occurred": timeout_occurred
                }
            )

            logger.info(f"Saved to Mem0 long-term memory (session={session_id})")

        except Exception as e:
            logger.error(f"Failed to save to Mem0: {e}")

        # 发布 Swarm 完成事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="swarm_coordinator",
            data={
                "duration": (end_time - start_time).total_seconds(),
                "agents_count": len(shared_context.agent_contributions)
            }
        ))

        # 返回结果
        completed_agents = list(shared_context.agent_contributions.keys())
        result = {
            'answer': final_answer,
            'swarm_enabled': True,
            'session_id': session_id,
            'agents_involved': completed_agents,
            'subtasks_completed': len(shared_context.get_all_completed_subtasks()),
            'total_time': (end_time - start_time).total_seconds(),
            'swarm_metadata': shared_context.get_summary(),
            'timeout_occurred': timeout_occurred
        }

        # 提取建议和免责声明（简化实现）
        result['suggestions'] = self._extract_suggestions(final_answer)

        # 根据是否超时调整免责声明
        if timeout_occurred and not completed_agents:
            result['disclaimer'] = "由于系统超时，未能提供完整分析。建议简化问题重试。如遇持续性学习困难，建议联系任课老师。"
        elif timeout_occurred:
            result['disclaimer'] = f"以上分析基于 {len(completed_agents)} 位导师的部分协作结果（部分分析模块超时未完成），仅供参考。"
        else:
            result['disclaimer'] = "以上分析基于多位导师的协作，仅供参考。学习效果受多种因素影响，建议结合教师反馈综合判断。"

        return result

    async def _worker_execute_with_timing(self, worker, shared_context, _time):
        t0 = _time.time()
        try:
            await self._worker_execute_assigned_tasks(worker, shared_context)
            elapsed = _time.time() - t0
            logger.info(f"{worker.agent_id}: completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = _time.time() - t0
            logger.error(f"{worker.agent_id}: failed after {elapsed:.1f}s: {e}")

    async def _worker_execute_assigned_tasks(
        self,
        worker: Any,
        shared_context: SharedContext
    ):
        """
        Worker 执行分配给它的任务

        简化后的流程：
        - 查找分配给自己的任务
        - 执行任务
        - 记录结果
        """
        try:
            # 获取分配给该 Agent 的任务
            assigned_tasks = shared_context.get_subtasks_for_agent(worker.agent_id)

            if not assigned_tasks:
                logger.debug(f"{worker.agent_id}: No assigned tasks")
                return

            # 并行执行所有分配的任务
            tasks = []
            for subtask in assigned_tasks:
                logger.info(f"{worker.agent_id}: Starting {subtask.type}")
                shared_context.start_subtask(subtask.id)

                task = asyncio.create_task(
                    self._execute_single_subtask(worker, subtask, shared_context)
                )
                tasks.append(task)

            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"{worker.agent_id}: Error processing subtask: {e}")

    async def _execute_single_subtask(self, worker, subtask, shared_context):
        """执行单个子任务"""
        try:
            result = await worker.process_subtask(subtask)
            shared_context.complete_subtask(subtask.id, worker.agent_id, result)
            logger.info(f"{worker.agent_id}: Completed {subtask.type}")
        except Exception as e:
            logger.error(f"{worker.agent_id}: Error in {subtask.type}: {e}")

    def _force_multi_subtasks(self, question: str, existing: list) -> list:
        """强制扩展为多 Agent 子任务"""
        if not existing:
            return [
                {"description": f"讲解「{question}」的核心概念和解题思路", "assigned_agent": "tutor_agent"},
                {"description": f"分析「{question}」涉及知识点的掌握薄弱点和前置依赖", "assigned_agent": "progress_agent"},
            ]
        # 已有 1 个任务，补充缺失的维度
        has_tutor = any(t.get("assigned_agent") == "tutor_agent" for t in existing)
        has_progress = any(t.get("assigned_agent") == "progress_agent" for t in existing)
        has_assess = any(t.get("assigned_agent") == "assess_agent" for t in existing)

        result = list(existing)
        if not has_tutor:
            result.append({"description": f"讲解「{question}」并提供针对性教学", "assigned_agent": "tutor_agent"})
        if not has_progress:
            result.append({"description": f"分析「{question}」的薄弱环节和学习路径", "assigned_agent": "progress_agent"})
        if not has_assess:
            result.append({"description": f"评估「{question}」相关知识的记忆状态和复习需求", "assigned_agent": "assess_agent"})
        return result

    def _extract_suggestions(self, final_answer: str) -> List[str]:
        """从最终答案中提取建议（简化实现）"""
        suggestions = []

        # 简单的文本匹配
        if "【核心建议】" in final_answer:
            # 提取核心建议部分
            start_idx = final_answer.find("【核心建议】")
            end_idx = final_answer.find("【", start_idx + 1)
            if end_idx == -1:
                end_idx = len(final_answer)

            suggestions_text = final_answer[start_idx:end_idx]

            # 提取编号列表
            import re
            matches = re.findall(r'\d+\.\s*([^\n]+)', suggestions_text)
            suggestions = matches[:5]  # 最多5条

        return suggestions or ["坚持每日练习，保持学习节奏"]

async def process_with_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None,
    force_swarm: bool = False
) -> Dict[str, Any]:
    """
    便捷函数：使用 Swarm 处理问题

    Args:
        question: 用户问题
        context: 额外上下文
        enable_swarm: 是否启用 Swarm（False 则总是用单 Agent）
        session_id: 会话ID
        force_swarm: 强制使用多 Agent 模式
    """
    coordinator = SwarmCoordinator(enable_swarm=enable_swarm)
    return await coordinator.process(question, context, session_id=session_id, force_swarm=force_swarm)
