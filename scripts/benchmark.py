#!/usr/bin/env python3
"""
性能基准测试脚本

运行一组预定义问题，收集完整的性能指标，输出面试可用的数据摘要。

用法:
    python scripts/benchmark.py              # 运行所有基准问题
    python scripts/benchmark.py --quick      # 只跑 3 个问题（快速验证）
    python scripts/benchmark.py --verbose    # 显示每个会话的详细指标
"""

import asyncio
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm.swarm_coordinator import SwarmCoordinator
from core.metrics_collector import MetricsCollector


# 测试问题集 — 覆盖简单/中等/复杂三类
BENCHMARK_QUESTIONS = [
    # === 简单概念问题（应走快速通道）===
    ("什么是三角函数？", False),
    ("勾股定理的公式是什么？", False),
    ("什么是牛顿第二定律？", False),

    # === 中等复杂问题（应触发单 Agent 或 Swarm）===
    ("三角函数 sin 和 cos 怎么区分？", False),
    ("怎么计算向量的点积？", False),
    ("如何判断二次函数的开口方向？", False),

    # === 复杂综合问题（应触发 Swarm 多 Agent）===
    ("我三角函数学得不好怎么办", True),
    ("帮我查漏补缺，看看我物理力学哪里薄弱", True),
    ("期末快到了，帮我规划数学复习", True),
    ("我的英语语法总是搞混，怎么系统提升", True),
]


async def run_single_question(coordinator, question, force_swarm, idx, total):
    """运行单个问题并返回结果"""
    print(f"\n[{idx}/{total}] 问题: {question}")
    print(f"      模式: {'强制 Swarm' if force_swarm else '自动路由'}")
    print("-" * 60)

    t0 = time.time()
    result = await coordinator.process(
        question=question,
        force_swarm=force_swarm,
    )
    elapsed = time.time() - t0

    # 摘要
    mode = result.get('swarm_enabled', False)
    agents = len(result.get('agents_involved', []))
    answer_len = len(result.get('answer', ''))
    print(f"  → 模式: {'Swarm' if mode else 'Single'}, "
          f"Agent数: {agents}, "
          f"耗时: {elapsed:.1f}s, "
          f"回答长度: {answer_len} 字")
    return result


async def run_benchmark(questions, verbose=False):
    """运行完整基准测试"""
    coordinator = SwarmCoordinator(enable_swarm=True)
    # 注意：不要 reset MetricsCollector，AgentLoop 已在 SwarmCoordinator 构造时持有引用

    print("=" * 60)
    print("EduX 性能基准测试")
    print(f"问题数量: {len(questions)}")
    print("=" * 60)

    t_start = time.time()
    results = []

    for idx, (question, force_swarm) in enumerate(questions, 1):
        result = await run_single_question(coordinator, question, force_swarm, idx, len(questions))
        results.append(result)

        if verbose:
            m = MetricsCollector().current
            if m:
                print(m.detail())

    t_total = time.time() - t_start

    # 聚合报告
    print("\n")
    print(MetricsCollector().aggregate_summary())

    # 补充统计
    print(f"\n📋 补充统计:")
    sessions = MetricsCollector().sessions
    if sessions:
        # 按模式分组
        fast = [s for s in sessions if s.mode == "fast_path"]
        single = [s for s in sessions if s.mode == "single_agent"]
        swarm = [s for s in sessions if s.mode == "swarm"]

        print(f"  Fast path:    {len(fast)}/{len(sessions)} ({len(fast)/len(sessions)*100:.0f}%)")
        if fast:
            avg_t = sum(s.total_time_s for s in fast) / len(fast)
            print(f"    Avg time:   {avg_t:.1f}s")
        print(f"  Single agent: {len(single)}/{len(sessions)}")
        if single:
            avg_t = sum(s.total_time_s for s in single) / len(single)
            print(f"    Avg time:   {avg_t:.1f}s")
        print(f"  Swarm:        {len(swarm)}/{len(sessions)} ({len(swarm)/len(sessions)*100:.0f}%)")
        if swarm:
            avg_t = sum(s.total_time_s for s in swarm) / len(swarm)
            avg_a = sum(s.agents_involved for s in swarm) / len(swarm)
            print(f"    Avg time:   {avg_t:.1f}s")
            print(f"    Avg agents: {avg_a:.1f}")

        # Token 统计
        all_calls = [c for s in sessions for c in s.llm_calls]
        if all_calls:
            total_tokens = sum(c.total_tokens for c in all_calls)
            total_prompt = sum(c.prompt_tokens for c in all_calls)
            total_completion = sum(c.completion_tokens for c in all_calls)
            avg_latency = sum(c.latency_ms for c in all_calls) / len(all_calls)
            print(f"\n  Total LLM calls:      {len(all_calls)}")
            print(f"  Total tokens:         {total_tokens:,}")
            print(f"  Total prompt tokens:  {total_prompt:,}")
            print(f"  Total completion:     {total_completion:,}")
            print(f"  Avg LLM latency:      {avg_latency:.0f}ms")
            print(f"  Avg tokens/call:      {total_tokens/len(all_calls):.0f}")

        # RAG 统计
        all_rag = [r for s in sessions for r in s.rag_searches]
        if all_rag:
            hit_rate = sum(1 for r in all_rag if r.result_count > 0) / len(all_rag)
            avg_lat = sum(r.latency_ms for r in all_rag) / len(all_rag)
            avg_res = sum(r.result_count for r in all_rag) / len(all_rag)
            print(f"\n  Total RAG searches:   {len(all_rag)}")
            print(f"  RAG hit rate:         {hit_rate:.1%}")
            print(f"  RAG avg latency:      {avg_lat:.0f}ms")
            print(f"  RAG avg results:      {avg_res:.1f}")

        # 循环统计
        avg_iter = sum(s.total_iterations for s in sessions) / len(sessions)
        avg_tool = sum(s.total_tool_calls for s in sessions) / len(sessions)
        blocks = sum(s.constraint_blocks for s in sessions)
        print(f"\n  Avg iterations/session:  {avg_iter:.1f}")
        print(f"  Avg tool calls/session:  {avg_tool:.1f}")
        print(f"  Total constraint blocks: {blocks}")

    print(f"\n⏱️  总耗时: {t_total:.1f}s")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="EduX Benchmark")
    parser.add_argument("--quick", action="store_true", help="只跑 3 个问题")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示每个会话的详细指标")
    args = parser.parse_args()

    questions = BENCHMARK_QUESTIONS[:3] if args.quick else BENCHMARK_QUESTIONS
    asyncio.run(run_benchmark(questions, verbose=args.verbose))


if __name__ == "__main__":
    main()
