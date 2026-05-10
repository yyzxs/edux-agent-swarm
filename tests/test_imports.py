"""
模块导入完整性测试

确保所有 __init__.py 中声明的导出都存在，防止 `LearningRecord` 这种事再发生。
"""
from pathlib import Path
import sys
import importlib

sys.path.insert(0, str(Path(__file__).parent.parent))


def _exports_of(module_name: str) -> list[str]:
    """读取模块 __init__.py 中 __all__ 列表"""
    mod = importlib.import_module(module_name)
    return getattr(mod, "__all__", [])


def _assert_all_exportable(module_name: str):
    """验证 __all__ 中的每个名字都能被 import"""
    mod = importlib.import_module(module_name)
    missing = []
    for name in _exports_of(module_name):
        try:
            getattr(mod, name)
        except AttributeError:
            missing.append(name)
    assert missing == [], f"{module_name} 导出不存在: {missing}"


class TestCoreImports:
    def test_core_all_exports_valid(self):
        _assert_all_exportable("core")

    def test_core_imports_llmclient(self):
        from core import LLMClient
        assert LLMClient is not None

    def test_core_imports_agentloop(self):
        from core import AgentLoop
        assert AgentLoop is not None

    def test_core_imports_skillregistry(self):
        from core import SkillRegistry
        assert SkillRegistry is not None


class TestMemoryImports:
    def test_memory_all_exports_valid(self):
        _assert_all_exportable("memory")

    def test_memory_imports_short_term(self):
        from memory import ShortTermMemory
        assert ShortTermMemory is not None

    def test_memory_imports_long_term(self):
        from memory import LongTermMemory
        assert LongTermMemory is not None

    def test_memory_imports_student_profile(self):
        from memory.student_profile import StudentProfile, StudentProfileManager
        assert StudentProfile is not None
        assert StudentProfileManager is not None


class TestSwarmImports:
    def test_swarm_all_exports_valid(self):
        _assert_all_exportable("swarm")

    def test_swarm_imports_coordinator(self):
        from swarm import SwarmCoordinator, process_with_swarm
        assert SwarmCoordinator is not None
        assert callable(process_with_swarm)


class TestAgentsImports:
    def test_agents_can_be_instantiated(self):
        from agents.tutor_agent import TutorAgent
        from agents.progress_agent import ProgressAgent
        from agents.assess_agent import AssessAgent

        tutor = TutorAgent()
        progress = ProgressAgent()
        assess = AssessAgent()

        assert tutor.agent_id == "tutor_agent"
        assert progress.agent_id == "progress_agent"
        assert assess.agent_id == "assess_agent"

    def test_agents_have_system_prompt(self):
        from agents.tutor_agent import TutorAgent
        from agents.progress_agent import ProgressAgent
        from agents.assess_agent import AssessAgent

        for agent_cls in [TutorAgent, ProgressAgent, AssessAgent]:
            agent = agent_cls()
            prompt = agent.get_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 50, f"{agent.agent_id} 的 system prompt 过短"

    def test_agents_have_skills_registered(self):
        from agents.tutor_agent import TutorAgent
        from agents.progress_agent import ProgressAgent
        from agents.assess_agent import AssessAgent

        for agent_cls in [TutorAgent, ProgressAgent, AssessAgent]:
            agent = agent_cls()
            tools = agent.get_tools_for_llm()
            assert len(tools) > 0, f"{agent.agent_id} 没有注册任何 skill"


class TestConstraintsImports:
    def test_constraints_validator_loads(self):
        from constraints.validator import ConstraintValidator
        v = ConstraintValidator()
        assert v.agent_constraints is not None
        assert v.swarm_constraints is not None

    def test_agent_constraints_yaml_parsable(self):
        import yaml
        from pathlib import Path
        path = Path(__file__).parent.parent / "constraints" / "agent_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "agents" in data
        for agent_id in ["tutor_agent", "progress_agent", "assess_agent"]:
            assert agent_id in data["agents"], f"{agent_id} 不在约束配置中"

    def test_swarm_constraints_yaml_parsable(self):
        import yaml
        from pathlib import Path
        path = Path(__file__).parent.parent / "constraints" / "swarm_constraints.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "swarm" in data


class TestValidationImports:
    def test_auto_fixer_loads(self):
        from validation.auto_fixer import AutoFixer
        f = AutoFixer()
        assert f is not None
