# EduX Agent Swarm 架构教程

## 一、项目概览

EduX 是一个面向教育的**多智能体协作系统（Agent Swarm）**，模仿蚁群的信息素机制实现去中心化的多 Agent 协作。核心目标是为学生提供个性化学习辅导：自适应教学、进步追踪、记忆评估。

**技术栈：** Python asyncio / DeepSeek V4 (OpenAI Compatible API) / Milvus Lite (向量检索) / Mem0 (长期记忆)

---

## 二、整体架构：6 层设计

```
用户层 (User Layer)
  └─ 交互层 (Interface Layer)          main.py CLI 交互
      └─ 编排层 (Orchestration Layer)   SwarmCoordinator + LeadAgent
          └─ Agent 层 (Multi-Agent Layer)  Tutor / Progress / Assess
              └─ 核心引擎层 (Core Engine)   AgentLoop / SkillRegistry / Personalization
                  ├─ 记忆系统 (Memory)     短期/长期/学生画像
                  ├─ 约束系统 (Constraints)  YAML规则 + 运行时验证 + 自动修复
                  └─ 知识库 (Knowledge)     Milvus 向量检索 + 知识图谱
```

### 各层职责

| 层 | 核心类 | 职责 |
|---|---|---|
| 交互层 | `main.py` | CLI 对话、风格检测、结果展示 |
| 编排层 | `SwarmCoordinator` | 智能路由（单/多Agent决策）、记忆注入、并行调度 |
| 编排层 | `LeadAgent` | 任务拆解、结果汇总（LLM 调用） |
| Agent 层 | `TutorAgent` / `ProgressAgent` / `AssessAgent` | 各自领域分析 |
| 核心引擎 | `AgentLoop` | Think-Act-Observe 循环引擎 |
| 核心引擎 | `SkillRegistry` | Skill → OpenAI Function Calling 格式转换 |
| 核心引擎 | `Personalization` | 学习风格检测、难度校准、艾宾浩斯复习调度 |

---

## 三、事件驱动机制（核心亮点）

### 3.1 设计哲学：去中心化 = 蚁群信息素系统

传统多 Agent 方案由中心节点指挥每个 Agent 做什么。EduX 反其道而行：**没有中心控制器，Agent 通过共享环境（SharedContext）间接通信。**

```
类比：
  蚁群：蚂蚁通过地面信息素痕迹感知环境 → 自主决定行动
  EduX：Agent 通过 SharedContext 读写数据 → 自主决定行动
```

### 3.2 核心组件

**Event（事件数据类）** — `swarm/events.py`

```python
@dataclass
class Event:
    type: EventType          # 8 种事件类型
    source_agent: str        # 发布者
    data: Dict[str, Any]     # 附加数据
    target_agents: List[str] # None = 广播给所有 Agent
```

**8 种事件类型：**

| 事件 | 触发时机 | 发布者 |
|---|---|---|
| `TASK_DECOMPOSED` | LeadAgent 分解任务后 | LeadAgent |
| `SUBTASK_STARTED` | Agent 开始执行子任务 | WorkerAgent |
| `SUBTASK_COMPLETED` | Agent 完成子任务 | WorkerAgent |
| `CONTEXT_UPDATED` | 共享数据被修改 | System |
| `AGENT_QUESTION` | Agent 向其他 Agent 提问 | WorkerAgent |
| `AGENT_ANSWER` | Agent 回答其他 Agent | WorkerAgent |
| `SWARM_STARTED` | Swarm 开始处理 | Coordinator |
| `SWARM_COMPLETED` | Swarm 完成处理 | Coordinator |

**SharedContext（共享环境）** — `swarm/shared_context.py`

```python
class SharedContext:
    data: Dict[str, Any]           # 共享数据
    events: List[Event]            # 事件流（按时间有序）
    task_decomposition: Dict[str, SubTask]  # 任务分解
    agent_contributions: Dict[str, List[Contribution]]  # Agent 贡献
    memory_pool: Dict[str, Any]    # 工作记忆
```

### 3.3 完整数据流

```
学生提问 "诱导公式总是搞不懂"
  │
  ├── 1. 学习风格检测  detect_style_from_text()
  │     关键词匹配 → 贝叶斯更新 → visual style
  │
  ├── 2. 记忆检索  SwarmCoordinator.process()
  │     短期记忆：最近对话历史
  │     长期记忆：Mem0 向量相似度搜索
  │
  ├── 3. LeadAgent 任务拆解  1 次 LLM 调用
  │     subtask_1 → TutorAgent: 讲解诱导公式
  │     subtask_2 → ProgressAgent: 分析薄弱链
  │     subtask_3 → AssessAgent: 评估记忆状态
  │
  ├── 4. 并行执行  asyncio.gather + 180s 超时
  │     ┌──────────┐  ┌──────────┐  ┌──────────┐
  │     │TutorAgent│  │Progress  │  │Assess    │
  │     │AgentLoop │  │AgentLoop │  │AgentLoop │
  │     └────┬─────┘  └────┬─────┘  └────┬─────┘
  │          │              │              │
  │     ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
  │     │ Think    │  │ Think    │  │ Think    │
  │     │ "查概念"  │  │ "分析图谱"│  │ "评估回答"│
  │     ├──────────┤  ├──────────┤  ├──────────┤
  │     │ Act      │  │ Act      │  │ Act      │
  │     │search_   │  │analyze_  │  │assess_   │
  │     │content() │  │gaps()    │  │level()   │
  │     ├──────────┤  ├──────────┤  ├──────────┤
  │     │ Observe  │  │ Observe  │  │ Observe  │
  │     │ "单位圆   │  │ "根因:    │  │ "概念混淆  │
  │     │ 是前置   │  │ 单位圆    │  │ 型错误"   │
  │     │ 依赖"    │  │ 未掌握"   │  │          │
  │     └────┬─────┘  └────┬─────┘  └────┬─────┘
  │          └──────────────┼──────────────┘
  │                  SharedContext 共享
  │
  ├── 5. LeadAgent 结果汇总  1 次 LLM 调用
  │     整合 3 个 Agent 视角 → 综合答复
  │
  ├── 6. 约束验证  ConstraintValidator
  │     检查：无代写答案 ✓  有免责声明 ✓  有鼓励 ✓
  │
  └── 7. 记忆持久化
        短期记忆追加 / 长期记忆 Mem0 保存 / 学生画像更新
```

---

## 四、Agent Loop：Think-Act-Observe 循环引擎

这是每个 Agent 运行的核心状态机，位于 `core/agent_loop.py`。

```
        ┌──────────┐       ┌──────────┐       ┌──────────┐
        │  THINK   │ ───→  │   ACT    │ ───→  │ OBSERVE  │
        │ LLM 决定  │       │ 调用Skill │       │ 分析结果  │
        │ 下一步    │       │ 获取数据  │       │ 决定继续  │
        └──────────┘       └──────────┘       └──────────┘
             ↑                                      │
             └──────────────────────────────────────┘
                    (循环直到完成或达到上限)
```

### 关键参数

- **max_iterations=10**：最大 10 轮循环，防止无限调用
- **max_tool_calls=10**：最大 Skill 调用次数，超过后强制生成最终答案
- **180s 超时**：Swarm 并行模式总超时

### 执行细节

1. **初始化消息**：系统 prompt + 短期记忆历史 + 用户输入
2. **LLM 调用**：`chat_with_tools()` 传入 Skill 的 OpenAI Function Calling 定义
3. **分叉判断**：
   - LLM 返回 `tool_calls` → 执行 Skill → 将结果追加到消息列表 → 继续循环
   - LLM 返回纯文本 → 任务完成，输出结果
4. **约束 Hook**：每次工具调用前验证合法性，最终输出前验证合规性
5. **强制降级**：达到最大迭代数或最大 tool 调用数时，强制 LLM 生成兜底答案

---

## 五、3 个 Agent 的实现详解

### 5.1 继承体系

```
BaseAgent (ABC)                    ← agents/base_agent.py
  ├── 持有 AgentLoop 实例
  ├── 持有 SkillRegistry 实例
  ├── 定义 get_system_prompt() / register_tools() 抽象方法
  └── 提供 process() / process_subtask() / run_loop()

SkillRegistryMixin (Mixin)         ← agents/skill_registry_mixin.py
  └── register_all_skills(): 自动扫描 .claude/skills/ 目录

TutorAgent(BaseAgent, SkillRegistryMixin)
ProgressAgent(BaseAgent, SkillRegistryMixin)
AssessAgent(BaseAgent, SkillRegistryMixin)
```

### 5.2 TutorAgent — 自适应学习导师

**文件：** `agents/tutor_agent.py`  
**配置：** `temperature=0.7, max_iterations=3`  
**定位：** 教学执行者，负责知识传递

**核心能力：**
1. **学习风格匹配** — 调用 `detect_style_from_text()` 从学生消息中识别风格信号，然后 `choose_teaching_approach()` 匹配 6 种教学策略（图形/文本/例题/推理/实践/未知）
2. **难度自适应** — `calibrate_difficulty()` 根据反馈动态调节 slow/normal/fast 三档节奏
3. **讲-问-练循环** — `should_ask_question()` 每 3 轮强制互动

**约束：** 禁止直接给答案（`give_answer_directly`），禁止代写作业（`do_homework_for_student`）

### 5.3 ProgressAgent — 学生进步指南

**文件：** `agents/progress_agent.py`  
**配置：** `temperature=0.7, max_iterations=5`  
**定位：** 学习分析师，负责诊断和规划

**核心能力：**
1. **知识图谱分析** — 读取 `StudentProfile.knowledge_graph`，分析掌握度分布
2. **薄弱点识别** — `get_weak_points()` 按错误率排序
3. **学习路径规划** — 结合前置依赖关系和当前掌握度推荐下一步
4. **进步趋势** — 对比历史数据生成进步曲线

**约束：** 不能横向比较学生，不能制造焦虑，必须先肯定进步再指出不足

### 5.4 AssessAgent — 记忆辅助评估

**文件：** `agents/assess_agent.py`  
**配置：** `temperature=0.7, max_iterations=5`  
**定位：** 记忆科学家，负责遗忘管理

**核心能力：**
1. **间隔复习调度** — 基于艾宾浩斯遗忘曲线，调用 `compute_review_schedule()`
2. **错误模式分类** — 概念混淆 / 计算错误 / 逻辑错误 / 审题错误
3. **掌握度真实评估** — 区分短期记忆（表面掌握）和长期记忆（真实掌握）
4. **遗忘预警** — 自动标记即将进入遗忘区的知识点

**约束：** 不能贴标签，单次复习不超过 5 个知识点

---

## 六、智能路由决策

`SwarmCoordinator.process()` 是系统的核心路由入口，决策逻辑如下：

```python
# 1. LeadAgent 分析问题 → 返回 subtasks[]
assessment = await lead_agent.assess_and_decompose(question, context)
subtasks = assessment.get("subtasks", [])

# 2. 规则兜底：LLM 误判为 1 个任务但问题明显复杂
if len(subtasks) == 1 and 问题包含 ["进步","薄弱","复习","遗忘","期末"] 等关键词:
    强制扩展为 2 个 subtask

# 3. 路由
if len(subtasks) == 1 and not force_swarm:
    → 单 Agent 模式，直接调用对应 Agent
elif len(subtasks) >= 2 and enable_swarm:
    → Swarm 模式，并行执行所有 Worker + 汇总
else:
    → 降级到 TutorAgent
```

---

## 七、Skill 自动发现机制

项目中有 7 个预定义 Skill（位于 `.claude/skills/`），通过自动发现机制注册：

```
.claude/skills/
├── search-content/      ← 搜教材/教辅/题库
├── analyze-gaps/        ← 薄弱环节分析
├── assess-level/        ← 掌握水平评估
├── knowledge-map/       ← 知识图谱查询
├── personalize-path/    ← 个性化学习路径
├── curriculum-standard/ ← 课程标准检索
├── search-history/      ← 对话历史检索
├── search-similar-cases/← 相似案例检索
└── deep-research/       ← 深度研究
```

**发现流程（`core/skill_loader.py`）：**

1. 扫描 `.claude/skills/*/` 子目录
2. 解析每个目录下的 `SKILL.md`（YAML frontmatter）
3. 动态 `importlib` 加载 `script/*.py`
4. 自动推断函数签名 → 生成 OpenAI Function Calling 参数定义
5. 注册到 `SkillRegistry` → 转换为 `tools` 传给 LLM

---

## 八、个性化引擎

**文件：** `core/personalization.py`

### 学习风格检测
通过正则匹配中文关键词推断 6 种风格（visual / textual / example / abstract / hands_on / unknown），采用贝叶斯式更新：每次互动微调 `style_confidence`。

### 艾宾浩斯复习调度
根据掌握度 × 历史正确率查表决定复习间隔（1~60 天）和复习方式（active_recall / guided_practice / light_review / check_in）。

```python
EBBINGHAUS_SCHEDULE = {
    (INTRODUCED, "low"):    (1, "high"),    # 1天后高优先级复习
    (PROFICIENT, "high"):   (14, "low"),    # 14天后低优先级
    (MASTERED, "high"):     (60, "minimal"),# 60天后确认式
}
```

---

## 九、约束系统（Harness Engineering）

**约束定义：** `constraints/agent_constraints.yaml`  
**运行时验证：** `constraints/validator.py`  
**自动修复：** `validation/auto_fixer.py`

### 工作流程

```
Agent 输出文本
  → ConstraintValidator.validate_output()
    检查：禁止直接给答案、禁止代写、必须含免责声明、不能贴标签等
  → 如果违规：
    → AutoFixer.fix_output()
       自动添加免责声明 / 添加求助引导 / 替换确定性断言
  → 输出修复后文本
```

### 示例约束（YAML）

```yaml
tutor_agent:
  forbidden_actions:
    - do_homework_for_student     # 不能替学生写作业
    - give_answer_directly        # 不能直接给答案
  output_constraints:
    - must_guide_not_solve        # 引导思路，不给解题步骤
    - max_response_length: 2000   # 回答最长 2000 字
```

---

## 十、记忆系统三层架构

| 层级 | 类 | 后端 | 生命周期 | 用途 |
|---|---|---|---|---|
| 短期记忆 | `ShortTermMemory` | 内存 Dict / Redis | 会话级（60min） | 当前对话历史 |
| 长期记忆 | `LongTermMemory` | Mem0 Cloud | 跨会话持久化 | 向量相似度搜索 |
| 学生画像 | `StudentProfile` | Mem0 + JSON | 永久 | 知识图谱/错题/复习队列 |

记忆在每次请求中形成闭环：**检索历史 → 注入上下文 → Agent 推理 → 追加新记忆**。

---

## 十一、关键设计模式总结

1. **去中心化协作**：Agent 通过 SharedContext 间接通信，而非相互调用
2. **事件驱动**：所有状态变化发布为 Event，Agent 按需订阅
3. **声明式约束**：YAML 定义行为边界，运行时自动验证和修复
4. **自动发现**：Skill 从目录自动扫描注册，无需手动维护列表
5. **优雅降级**：LLM 失败 → 默认 TutorAgent；超时 → 部分结果汇总；约束违规 → 自动修复

---

## 十二、快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API（编辑 config.py）
LLM_CONFIG = {
    "api_key": "your-key",
    "base_url": "https://api.deepseek.com/v1",
    "model_name": "deepseek-chat",
}

# 启动
python main.py

# 带详细日志
python main.py -v
```

交互命令：

| 命令 | 功能 |
|---|---|
| 直接输入问题 | 自动路由到合适的Agent |
| `swarm <问题>` | 强制多导师协作模式 |
| `progress` | 查看学习进度概览 |
| `review` | 查看今日复习计划 |
| `help` | 帮助 |
| `exit` | 退出 |
