"""
自动修复器
根据约束违规自动修复输出

基于 Harness Engineering 原则：
- 自动检测问题
- 自动修复（在可能的情况下）
- 保持 Agent 输出质量
"""
from typing import Dict, Any, List
from loguru import logger


class AutoFixer:
    """自动修复器"""

    def fix_output(
        self,
        output: str,
        auto_fixable: List[str]
    ) -> str:
        """
        自动修复输出

        Args:
            output: 原始输出
            auto_fixable: 可修复的违规列表

        Returns:
            修复后的输出
        """
        fixed_output = output

        for fix_type in auto_fixable:
            if fix_type == "add_disclaimer":
                fixed_output = self.fix_missing_disclaimer(fixed_output)
            elif fix_type == "add_stuck_guidance":
                fixed_output = self.fix_high_risk_warning(fixed_output)

        if fixed_output != output:
            logger.info("🔧 输出已自动修复")

        return fixed_output

    def fix_missing_disclaimer(self, output: str) -> str:
        """
        自动添加免责声明

        Args:
            output: 原始输出

        Returns:
            添加免责声明后的输出
        """
        if "免责" not in output and "仅供参考" not in output:
            disclaimer = "\n\n【免责声明】\n以上内容为学习辅导建议，仅供参考。如遇持续性学习困难，建议与学校老师沟通，找到最适合的学习方案。"
            logger.debug("+ 自动添加免责声明")
            return output + disclaimer
        return output

    def fix_high_risk_warning(self, output: str) -> str:
        """
        自动添加学习困难提醒

        Args:
            output: 原始输出

        Returns:
            添加提醒后的输出
        """
        stuck_keywords = ["完全听不懂", "实在太难了", "学不会", "放弃了", "不想学了"]

        # 检查是否包含学习困难信号且未建议求助
        if any(kw in output for kw in stuck_keywords):
            if "老师" not in output and "求助" not in output and "同学" not in output:
                warning = "💡 **学习建议**：遇到持续困难是正常的，建议主动向老师或同学寻求帮助，不要独自硬撑。\n\n"
                logger.debug("+ 自动添加学习困难引导")
                return warning + output

        return output

    def fix_excessive_length(self, output: str, max_length: int) -> str:
        """
        截断过长的输出

        Args:
            output: 原始输出
            max_length: 最大长度

        Returns:
            截断后的输出
        """
        if len(output) > max_length:
            logger.warning(f"输出过长（{len(output)} > {max_length}），自动截断")
            truncated = output[:max_length - 50]  # 保留50字空间添加提示
            truncated += "\n\n[回答内容较长，已截断。如需完整信息，建议与学校老师沟通]"
            return truncated

        return output

    def remove_diagnosis_statements(self, output: str) -> str:
        """
        移除越界的确定性断言（高级功能，需要 LLM 辅助）

        Args:
            output: 原始输出

        Returns:
            修复后的输出
        """
        # 简单替换确定性断言为建议性表述
        output = output.replace("你肯定是", "你可能存在")
        output = output.replace("一定是", "这种表现可能意味着")

        return output
