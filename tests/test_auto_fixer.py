"""
AutoFixer 纯文本修复测试

所有测试只依赖字符串输入输出，不依赖 LLM 或外部服务。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.auto_fixer import AutoFixer


fixer = AutoFixer()


# ── 免责声明 ────────────────────────────────────────

class TestDisclaimer:
    def test_adds_disclaimer_when_missing(self):
        output = "这道题的思路是……"
        result = fixer.fix_missing_disclaimer(output)
        assert "仅供参考" in result
        assert "学习辅导建议" in result

    def test_no_duplicate_when_already_present(self):
        output = "这道题的思路是……仅供参考，建议与老师沟通。"
        result = fixer.fix_missing_disclaimer(output)
        # 已含 "仅供参考"，不应追加
        assert result == output

    def test_no_duplicate_when_disclaimer_present(self):
        output = "答案如下。\n\n【免责声明】\n仅供参考。"
        result = fixer.fix_missing_disclaimer(output)
        assert result == output


# ── 学习困难提醒 ────────────────────────────────────

class TestStuckGuidance:
    def test_adds_guidance_for_stuck_signal(self):
        output = "三角函数实在太难了，完全听不懂"
        result = fixer.fix_high_risk_warning(output)
        assert "老师" in result or "求助" in result or "同学" in result

    def test_no_duplicate_when_guidance_present(self):
        output = "实在太难了，建议找老师问问"
        result = fixer.fix_high_risk_warning(output)
        assert result == output

    def test_no_guidance_for_normal_content(self):
        output = "三角函数的核心是单位圆的定义。"
        result = fixer.fix_high_risk_warning(output)
        assert result == output


# ── 长度截断 ───────────────────────────────────────

class TestLengthTruncation:
    def test_truncates_over_limit(self):
        output = "x" * 500
        result = fixer.fix_excessive_length(output, max_length=200)
        assert len(result) <= 200

    def test_preserves_short_output(self):
        output = "short answer"
        result = fixer.fix_excessive_length(output, max_length=200)
        assert result == output


# ── 断言替换 ───────────────────────────────────────

class TestDiagnosisRemoval:
    def test_replaces_definitive_statements(self):
        output = "你肯定是审题不仔细。"
        result = fixer.remove_diagnosis_statements(output)
        assert "你肯定是" not in result
        assert "可能" in result

    def test_preserves_normal_text(self):
        output = "我们来分析一下这道题的思路。"
        result = fixer.remove_diagnosis_statements(output)
        assert result == output


# ── fix_output 调度 ─────────────────────────────────

class TestFixOutputDispatcher:
    def test_dispatches_to_correct_fixer(self):
        output = "三角函数实在太难了，完全听不懂"
        result = fixer.fix_output(output, auto_fixable=["add_stuck_guidance"])
        assert len(result) > len(output)  # 被添加了内容

    def test_dispatches_multiple_fixes(self):
        output = "三角函数实在太难了，学不会"
        result = fixer.fix_output(output, auto_fixable=["add_stuck_guidance", "add_disclaimer"])
        assert "老师" in result or "求助" in result or "同学" in result
        assert "仅供参考" in result

    def test_unknown_fix_type_preserves_output(self):
        output = "hello"
        result = fixer.fix_output(output, auto_fixable=["nonexistent_fix"])
        assert result == output
