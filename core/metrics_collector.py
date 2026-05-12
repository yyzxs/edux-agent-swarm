"""
会话级指标收集器

在 AgentLoop、SwarmCoordinator、RAG 检索等关键路径埋点，
一次会话结束后输出完整的指标摘要。
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from loguru import logger


@dataclass
class LLMCallRecord:
    """单次 LLM 调用记录"""
    timestamp: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    has_tool_calls: bool = False
    tool_call_names: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class RAGSearchRecord:
    """单次 RAG 检索记录"""
    timestamp: float = field(default_factory=time.time)
    query: str = ""
    result_count: int = 0
    latency_ms: float = 0.0


@dataclass
class SessionMetrics:
    """一次会话的完整指标"""
    session_id: str = ""
    mode: str = ""  # "single_agent" | "swarm" | "fallback" | "fast_path"

    # LLM 统计
    llm_calls: List[LLMCallRecord] = field(default_factory=list)

    # RAG 统计
    rag_searches: List[RAGSearchRecord] = field(default_factory=list)

    # Agent Loop 统计
    total_iterations: int = 0
    total_tool_calls: int = 0
    tool_call_limit_hit: bool = False
    constraint_blocks: int = 0

    # Swarm 统计
    agents_involved: int = 0
    subtasks_completed: int = 0
    timeout_occurred: bool = False

    # 时间统计
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def total_time_s(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def avg_llm_latency_ms(self) -> float:
        if not self.llm_calls:
            return 0.0
        return sum(c.latency_ms for c in self.llm_calls) / len(self.llm_calls)

    @property
    def rag_hit_rate(self) -> float:
        """RAG 检索命中率（返回结果 > 0）"""
        if not self.rag_searches:
            return 1.0
        hits = sum(1 for r in self.rag_searches if r.result_count > 0)
        return hits / len(self.rag_searches)

    @property
    def avg_rag_latency_ms(self) -> float:
        if not self.rag_searches:
            return 0.0
        return sum(r.latency_ms for r in self.rag_searches) / len(self.rag_searches)

    @property
    def avg_rag_results(self) -> float:
        if not self.rag_searches:
            return 0.0
        return sum(r.result_count for r in self.rag_searches) / len(self.rag_searches)

    @property
    def estimated_cost_usd(self) -> float:
        """按典型 DeepSeek/豆包 价格估算"""
        prompt_price = 0.14 / 1_000_000   # $0.14/1M input tokens
        completion_price = 0.28 / 1_000_000
        return (self.total_prompt_tokens * prompt_price +
                self.total_completion_tokens * completion_price)

    def summary(self) -> str:
        """一行摘要，适合面试引用"""
        return (
            f"[{self.session_id}] mode={self.mode} | "
            f"{self.total_time_s:.1f}s | "
            f"LLM: {len(self.llm_calls)} calls, {self.total_tokens} tokens "
            f"(in:{self.total_prompt_tokens} out:{self.total_completion_tokens}), "
            f"avg {self.avg_llm_latency_ms:.0f}ms/call | "
            f"RAG: {len(self.rag_searches)} searches, hit_rate={self.rag_hit_rate:.0%}, "
            f"avg {self.avg_rag_results:.1f} results, {self.avg_rag_latency_ms:.0f}ms | "
            f"Loop: {self.total_iterations} iters, {self.total_tool_calls} tools"
            + (f", {self.constraint_blocks} blocked" if self.constraint_blocks else "")
            + (f" | Swarm: {self.agents_involved} agents, {self.subtasks_completed} done"
               if self.mode == "swarm" else "")
        )

    def detail(self) -> str:
        """详细报告"""
        lines = [
            "=" * 60,
            f"SESSION METRICS REPORT — {self.session_id}",
            "=" * 60,
            f"  Mode:          {self.mode}",
            f"  Duration:      {self.total_time_s:.2f}s",
            f"  Agents:        {self.agents_involved}",
            f"  Subtasks done: {self.subtasks_completed}",
            f"  Timeout:       {self.timeout_occurred}",
            "",
            "  LLM Usage:",
            f"    Total calls:       {len(self.llm_calls)}",
            f"    Total tokens:      {self.total_tokens:,}",
            f"    Prompt tokens:     {self.total_prompt_tokens:,}",
            f"    Completion tokens: {self.total_completion_tokens:,}",
            f"    Avg latency:       {self.avg_llm_latency_ms:.0f}ms",
            f"    Est. cost:         ${self.estimated_cost_usd:.6f}",
            "",
            "  Agent Loop:",
            f"    Iterations:   {self.total_iterations}",
            f"    Tool calls:   {self.total_tool_calls}",
            f"    Limit hit:    {self.tool_call_limit_hit}",
            f"    Constraints:  {self.constraint_blocks} blocked",
            "",
            "  RAG Retrieval:",
            f"    Searches:     {len(self.rag_searches)}",
            f"    Hit rate:     {self.rag_hit_rate:.1%}",
            f"    Avg results:  {self.avg_rag_results:.1f}",
            f"    Avg latency:  {self.avg_rag_latency_ms:.0f}ms",
            "=" * 60,
        ]
        return "\n".join(lines)


class MetricsCollector:
    """会话级指标收集器（单例）"""

    _instance: Optional["MetricsCollector"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions: List[SessionMetrics] = []
            cls._instance._current: Optional[SessionMetrics] = None
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @property
    def sessions(self) -> List[SessionMetrics]:
        return self._sessions

    @property
    def current(self) -> Optional[SessionMetrics]:
        return self._current

    @property
    def total_sessions(self) -> int:
        return len(self._sessions)

    def start_session(self, session_id: str, mode: str = "") -> SessionMetrics:
        self._current = SessionMetrics(
            session_id=session_id,
            mode=mode,
            start_time=datetime.now(),
        )
        return self._current

    def end_session(self):
        if self._current:
            self._current.end_time = datetime.now()
            self._sessions.append(self._current)
            logger.info(self._current.summary())
            self._current = None

    def set_mode(self, mode: str):
        if self._current:
            self._current.mode = mode

    def record_llm_call(self, record: LLMCallRecord):
        if self._current:
            self._current.llm_calls.append(record)

    def record_rag_search(self, record: RAGSearchRecord):
        if self._current:
            self._current.rag_searches.append(record)

    def record_constraint_block(self):
        if self._current:
            self._current.constraint_blocks += 1

    def set_swarm_stats(self, agents_involved: int, subtasks_completed: int, timeout_occurred: bool = False):
        if self._current:
            self._current.agents_involved = agents_involved
            self._current.subtasks_completed = subtasks_completed
            self._current.timeout_occurred = timeout_occurred

    def set_loop_stats(self, total_iterations: int, total_tool_calls: int, tool_call_limit_hit: bool = False):
        if self._current:
            self._current.total_iterations = total_iterations
            self._current.total_tool_calls = total_tool_calls
            self._current.tool_call_limit_hit = tool_call_limit_hit

    def aggregate_summary(self) -> str:
        """所有会话的聚合摘要 — 面试时可以直接用的数字"""
        if not self._sessions:
            return "No sessions recorded."

        total_time = sum(s.total_time_s for s in self._sessions)
        total_tokens = sum(s.total_tokens for s in self._sessions)
        total_cost = sum(s.estimated_cost_usd for s in self._sessions)
        total_llm_calls = sum(len(s.llm_calls) for s in self._sessions)
        total_rag = sum(len(s.rag_searches) for s in self._sessions)
        avg_iterations = sum(s.total_iterations for s in self._sessions) / len(self._sessions)
        avg_tool_calls = sum(s.total_tool_calls for s in self._sessions) / len(self._sessions)

        swarm_sessions = [s for s in self._sessions if s.mode == "swarm"]
        single_sessions = [s for s in self._sessions if s.mode in ("single_agent", "fast_path")]

        lines = [
            "=" * 60,
            "AGGREGATE METRICS — ALL SESSIONS",
            "=" * 60,
            f"  Total sessions:     {len(self._sessions)}",
            f"  Swarm sessions:     {len(swarm_sessions)}",
            f"  Single-agent:       {len(single_sessions)}",
            f"  Total time:         {total_time:.1f}s",
            f"  Total tokens:       {total_tokens:,}",
            f"  Total cost:         ${total_cost:.6f}",
            f"  Total LLM calls:    {total_llm_calls}",
            f"  Total RAG searches: {total_rag}",
            f"  Avg iterations:     {avg_iterations:.1f}",
            f"  Avg tool calls:     {avg_tool_calls:.1f}",
        ]

        if swarm_sessions:
            avg_swarm_time = sum(s.total_time_s for s in swarm_sessions) / len(swarm_sessions)
            avg_swarm_agents = sum(s.agents_involved for s in swarm_sessions) / len(swarm_sessions)
            lines.extend([
                "",
                "  Swarm avg:",
                f"    Time:        {avg_swarm_time:.1f}s",
                f"    Agents:      {avg_swarm_agents:.1f}",
            ])

        if single_sessions:
            avg_single_time = sum(s.total_time_s for s in single_sessions) / len(single_sessions)
            lines.extend([
                "",
                "  Single-agent avg:",
                f"    Time:        {avg_single_time:.1f}s",
            ])

        if swarm_sessions and single_sessions:
            swarm_avg = sum(s.total_time_s for s in swarm_sessions) / len(swarm_sessions)
            single_avg = sum(s.total_time_s for s in single_sessions) / len(single_sessions)
            ratio = swarm_avg / single_avg if single_avg > 0 else 0
            lines.append(f"    Speedup:     Swarm is {ratio:.1f}x slower than single (expected for multi-agent)")

        # RAG 统计
        all_rag = [r for s in self._sessions for r in s.rag_searches]
        if all_rag:
            hit_rate = sum(1 for r in all_rag if r.result_count > 0) / len(all_rag)
            avg_latency = sum(r.latency_ms for r in all_rag) / len(all_rag)
            avg_results = sum(r.result_count for r in all_rag) / len(all_rag)
            lines.extend([
                "",
                "  RAG aggregate:",
                f"    Hit rate:    {hit_rate:.1%}",
                f"    Avg latency: {avg_latency:.0f}ms",
                f"    Avg results: {avg_results:.1f}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
