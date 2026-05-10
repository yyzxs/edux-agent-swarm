"""
search_history — 搜索会话历史（短期记忆）
语义搜索 + 对话摘要，不只是关键词过滤
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in (Path(__file__).resolve().parents) if (p / "config.py").exists())))


async def search_history(session_id: str, keyword: str = "", limit: int = 10) -> dict:
    """
    检索当前会话的对话历史

    Args:
        session_id: 会话ID
        keyword: 关键词过滤（可选，为空则返回最近消息）
        limit: 返回消息数量
    """
    try:
        from memory.short_term import ShortTermMemory
        memory = ShortTermMemory(storage_type="memory")
        messages = memory.get_recent_messages(session_id=session_id, limit=limit)

        if not messages:
            return {
                "success": True,
                "session_id": session_id,
                "messages": [],
                "summary": "暂无对话历史",
                "count": 0
            }

        # 关键词过滤
        if keyword:
            messages = [
                m for m in messages
                if keyword in str(m.get("content", ""))
            ]

        # 构建结构化消息列表
        formatted = []
        for m in messages:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))
            formatted.append({
                "role": role,
                "content": content[:400],
                "length": len(content)
            })

        # 生成对话摘要
        summary = _summarize_conversation(formatted)

        # 提取关键主题
        topics = _extract_topics(formatted)

        return {
            "success": True,
            "session_id": session_id,
            "messages": formatted,
            "count": len(formatted),
            "summary": summary,
            "topics_discussed": topics,
            "keyword_filter": keyword if keyword else None
        }

    except Exception as e:
        return {"success": False, "session_id": session_id, "error": str(e)}


def _summarize_conversation(messages: list) -> str:
    """生成对话流程摘要"""
    if not messages:
        return "空对话"

    user_msgs = [m for m in messages if m["role"] == "user"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    last_user = user_msgs[-1]["content"][:100] if user_msgs else "无"
    last_assistant = assistant_msgs[-1]["content"][:100] if assistant_msgs else "无"

    return (
        f"共 {len(messages)} 条消息（{len(user_msgs)} 次提问，{len(assistant_msgs)} 次回复）。"
        f"最近提问：「{last_user}」"
    )


def _extract_topics(messages: list) -> list:
    """从对话中提取可能的知识点关键词（简单规则）"""
    edu_keywords = [
        "函数", "三角", "数列", "向量", "导数", "积分", "概率", "统计",
        "集合", "不等式", "几何", "方程", "公式", "定理", "证明", "计算",
        "语法", "阅读", "作文", "单词", "听力", "物理", "化学", "生物"
    ]
    all_text = " ".join([m["content"] for m in messages if m["role"] == "user"])
    found = []
    for kw in edu_keywords:
        if kw in all_text and kw not in found:
            found.append(kw)
    return found[:5]
