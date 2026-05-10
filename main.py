#!/usr/bin/env python3
"""
EduX 多智能体教育助手 — 主入口

三大核心功能：
  - 自适应学习导师    → 根据学习风格和节奏个性化教学
  - 学生进步指南      → 追踪知识图谱，规划学习路径
  - 记忆辅助评估      → 基于遗忘曲线科学安排复习

交互式对话；可选 -v / --verbose 开启详细日志
"""
import asyncio
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from loguru import logger

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from swarm import process_with_swarm


def setup_logger(verbose: bool = False):
    logger.remove()
    if verbose:
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="DEBUG"
        )
    else:
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )


STUDENT_WELCOME = """
📚 {sep}
{pad}EduX 多智能体教育助手
{pad}个性化学习 · 自适应辅导 · 科学记忆
📚 {sep}

💡 使用说明：
  - 直接输入你的学习问题
  - 系统会自动调用合适的导师
  - 输入 'exit' 或 'quit' 退出
  - 输入 'progress' 查看学习进度
  - 输入 'review' 查看今日复习计划
  - 输入 'help' 查看帮助

"""

STUDENT_INTRO = """
第一次使用？请简单告诉我：
  - 你的年级（如"高二"）
  - 你想学什么（如"三角函数"）

这样我能更好地为你制定学习方案。
"""


async def interactive_mode():
    sep = "📚 " * 20
    pad = " " * 15

    print("\n" + sep)
    print(pad + "EduX 多智能体教育助手")
    print(pad + "个性化学习 · 自适应辅导 · 科学记忆")
    print(sep + "\n")

    print("💡 使用说明：")
    print("  - 直接输入你的学习问题")
    print("  - 系统会自动判断使用单导师还是多导师协作")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'progress' 查看学习进度")
    print("  - 输入 'review' 查看今日复习计划")
    print("  - 输入 'help' 查看帮助")
    print("\n" + "-" * 60 + "\n")

    conversation_count = 0
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
    student_id = f"student_{str(uuid.uuid4())[:8]}"

    logger.info(f"Session started: {session_id}, student: {student_id}")

    # 初始化学生画像
    try:
        from memory.student_profile import StudentProfileManager
        from memory.long_term import LongTermMemory
        long_term = LongTermMemory()
        profile_manager = StudentProfileManager(long_term_memory=long_term)
        profile = profile_manager.get_profile(student_id)
    except Exception as e:
        logger.warning(f"Profile init failed, continuing without: {e}")

    while True:
        try:
            user_input = input("💬 你的问题：").strip()

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ['exit', 'quit', 'q']:
                print("\n👋 学习是一辈子的事，下次见！\n")
                break

            if cmd == 'clear':
                print("\033[2J\033[H")
                continue

            if cmd == 'help':
                print("\n📖 EduX 帮助：")
                print("  直接提问：如 '三角函数怎么学' '这道力学题不会'")
                print("  progress  - 查看学习进度")
                print("  review    - 查看今日复习计划")
                print("  swarm <问题> - 强制多导师协作模式")
                print("  clear     - 清屏")
                print("  exit      - 退出\n")
                continue

            # 强制 Swarm 模式：swarm <问题>
            force_swarm = False
            if cmd.startswith('swarm '):
                user_input = user_input[6:].strip()
                force_swarm = True
                if not user_input:
                    print("⚠️ 用法：swarm <你的问题>")
                    continue

            if cmd == 'progress':
                try:
                    p = profile_manager.get_profile(student_id)
                    summary = p.progress_summary()
                    print(f"\n📈 你的学习概览：")
                    print(f"  知识点总数：{summary['total_knowledge_points']}")
                    print(f"  已掌握：{summary['mastered_count']} ({summary['mastery_ratio']:.0%})")
                    print(f"  学习风格：{summary['learning_style']}")
                    if summary['weak_points']:
                        print(f"  薄弱环节：{', '.join(summary['weak_points'][:3])}")
                    if summary['due_reviews']:
                        print(f"  待复习：{summary['due_reviews']} 个知识点\n")
                    else:
                        print()
                except Exception as e:
                    print(f"\n⚠️ 进度查询失败：{e}\n")
                continue

            if cmd == 'review':
                try:
                    from core.personalization import get_todays_review_plan
                    p = profile_manager.get_profile(student_id)
                    plan = get_todays_review_plan(p)
                    if plan:
                        print(f"\n🧠 今日复习计划（{len(plan)}项）：")
                        for item in plan:
                            print(f"  • {item.get('kp_name', item['kp_id'])} — {item.get('review_method', 'review')}（掌握度：{item.get('mastery', 'unknown')}）")
                        print()
                    else:
                        print("\n✅ 今天没有需要复习的内容，继续学新知识吧！\n")
                except Exception as e:
                    print(f"\n⚠️ 复习计划查询失败：{e}\n")
                continue

            # 处理学习问题
            conversation_count += 1
            print(f"\n🤖 智能协作系统启动中... (第 {conversation_count} 次辅导)\n")

            start_time = time.time()

            # 学习风格检测
            try:
                from core.personalization import detect_style_from_text
                style_signal = detect_style_from_text(user_input)
                if style_signal:
                    p = profile_manager.get_profile(student_id)
                    p.update_style_signal(style_signal)
            except Exception:
                pass

            context = {
                "student_id": student_id,
                "conversation_count": conversation_count,
            }

            result = await process_with_swarm(
                user_input,
                context=context,
                session_id=session_id,
                force_swarm=force_swarm
            )
            end_time = time.time()
            execution_time = end_time - start_time

            # 显示模式
            if result.get('swarm_enabled'):
                agents_count = len(result.get('agents_involved', []))
                timeout = result.get('timeout_occurred', False)
                if timeout and agents_count == 0:
                    print(f"⚠️  多导师模式：系统超时，所有导师未完成")
                elif timeout:
                    print(f"⚠️  多导师模式：{agents_count} 位导师完成（部分超时）")
                else:
                    agent_names = [a.replace('_', ' ').title() for a in result.get('agents_involved', [])]
                    print(f"🐝 多导师协作：{' + '.join(agent_names)}")
            else:
                print(f"🤖 单导师模式")

            print(f"⏱️  耗时：{execution_time:.2f} 秒")

            # 显示回答
            print("\n📋 回答：")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)

            if result.get('suggestions'):
                print(f"\n💡 建议：")
                for i, s in enumerate(result['suggestions'], 1):
                    print(f"  {i}. {s}")

            print(f"\n{result.get('disclaimer', '')}")
            print("\n" + "=" * 60 + "\n")

            # 更新学生画像
            try:
                p = profile_manager.get_profile(student_id)
                p.total_interactions += 1
            except Exception:
                pass

        except KeyboardInterrupt:
            print("\n\n👋 暂停学习，下次继续！\n")
            break
        except Exception as e:
            logger.error(f"处理请求时出错: {e}")
            print(f"\n❌ 抱歉，处理时出现错误：{e}\n")

    # 会话结束时保存画像
    try:
        profile_manager.save_profile(student_id)
        logger.info(f"Profile saved for {student_id}")
    except Exception as e:
        logger.error(f"Failed to save profile: {e}")


def main():
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    setup_logger(verbose)

    try:
        asyncio.run(interactive_mode())
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
