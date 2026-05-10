# EduX 多智能体教育助手

基于 Skills-Agent 两层架构的多智能体协作教育系统，融合 Agent Loop、Agent Swarm、学生画像和 Milvus 知识库。

## 概述

EduX 采用 **Skills-Agent 两层架构**，通过 7 个自包含的原子 Skills 和 3 个专业教育 Agent 协同工作，将一刀切的教育转化为真正的个性化学习。

### 三大核心功能

| 功能 | Agent | 一句话 |
|------|-------|--------|
| **自适应学习导师** | TutorAgent | 根据学生的学习风格和节奏，实时调整教学方式 |
| **学生进步指南** | ProgressAgent | 追踪知识图谱，告诉学生"在哪、去哪、怎么去" |
| **记忆辅助评估** | AssessAgent | 基于艾宾浩斯遗忘曲线，在最佳时机触发复习 |

### 核心特性

- **自适应教学**: 识别 5 种学习风格（图形型/文本型/例题型/推理型/实践型），匹配最佳讲解方式
- **学生画像系统**: 持久化知识图谱、学习轨迹、错误模式、遗忘曲线，驱动个性化
- **Agent Loop**: LLM 驱动的 Skill 调用循环，Agent 自主规划并完成任务
- **Agent Swarm**: 群体智能（去中心化协作，并行执行），复杂问题多 Agent 协同
- **Milvus 知识库**: 语义检索教育内容（教材、课标、题库），支持模糊查询
- **约束系统**: 教育边界约束（不代写作业、不保证分数、不制造焦虑），运行时自动验证
- **记忆系统**: 短期（会话级对话历史）+ 长期（Mem0 跨会话学生画像）

---

## 快速上手

### 1. 环境准备

```bash
conda create -n edux python=3.12 -y
conda activate edux
cd edux-agent-swarm
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API

编辑 `config.py`：

```python
LLM_CONFIG = {
    "api_key": "your-api-key",
    "model_name": "your-model-name",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}

MEM0_CONFIG = {"api_key": "m0-your-api-key-here"}
```

### 4. 初始化知识库

```bash
python knowledge/scripts/import_edu_data.py
```

### 5. 开始使用

```bash
python main.py
```

交互命令：

| 命令 | 功能 |
|------|------|
| 直接输入问题 | 触发智能辅导 |
| `progress` | 查看学习进度概览 |
| `review` | 查看今日复习计划 |
| `help` | 显示帮助 |
| `exit` | 退出 |

---

## 项目结构

```
edux-agent-swarm/
├── main.py                          # 主入口（交互式对话）
├── config.py                        # LLM/Mem0 配置
│
├── agents/                          # 教育 Agent
│   ├── tutor_agent.py               #   自适应学习导师
│   ├── progress_agent.py            #   学生进步指南
│   ├── assess_agent.py              #   记忆辅助评估
│   ├── base_agent.py                #   Agent 基类
│   └── skill_registry_mixin.py      #   Skill 注册混入
│
├── core/                            # 核心引擎
│   ├── agent_loop.py                #   Agent 循环（Think-Act-Observe）
│   ├── personalization.py           #   个性化引擎（风格检测/难度校准/复习调度）
│   ├── skill_registry.py            #   Skill 注册表（→ OpenAI function calling）
│   ├── skill_loader.py              #   动态加载 Skills
│   ├── llm_client.py                #   LLM 客户端
│   └── state_manager.py             #   状态管理
│
├── memory/                          # 记忆管理
│   ├── student_profile.py           #   学生画像（学习风格/知识图谱/错题轨迹/复习队列）
│   ├── short_term.py                #   短期记忆（会话级对话历史）
│   ├── long_term.py                 #   长期记忆（Mem0 跨会话）
│   ├── session_summary.py           #   会话总结
│   ├── entropy_manager.py           #   熵管理（去重和压缩）
│   └── agent_identity.py            #   Agent 身份管理
│
├── swarm/                           # 群体协作
│   ├── lead_agent.py                #   任务分解与结果汇总
│   ├── swarm_coordinator.py         #   智能路由（单/多 Agent 决策）
│   ├── shared_context.py            #   共享环境
│   └── events.py                    #   事件驱动通信
│
├── constraints/                     # 约束系统
│   ├── agent_constraints.yaml       #   Agent 能力边界
│   ├── swarm_constraints.yaml       #   Swarm 协作规则
│   └── validator.py                 #   运行时约束验证器
│
├── knowledge/                       # Milvus 知识库
│   ├── milvus_kb.py                 #   知识库封装
│   ├── data/documents/              #   教育知识文档（txt）
│   └── scripts/                     #   数据导入脚本
│
├── research/                        # DeepResearch 模块
│   ├── deep_research_workflow.py    #   深度研究工作流
│   ├── evidence_synthesizer.py      #   证据综合器
│   └── web_search.py                #   网络搜索
│
└── .claude/skills/                  # 7 个教育 Skills
    ├── search-content/              #   搜索教育内容
    ├── assess-level/                #   学力水平诊断
    ├── analyze-gaps/                #   薄弱点分析
    ├── personalize-path/            #   个性化学习路径
    ├── knowledge-map/               #   知识点关系图谱
    ├── curriculum-standard/         #   课程标准检索
    └── deep-research/               #   深度教育研究
```

---

## Skills 和 Agent

### 7 个教育 Skills

| Skill | 功能 | 数据源 |
|-------|------|--------|
| `search_content` | 搜索教材/教辅/题库内容 | Milvus 语义检索 |
| `assess_level` | 诊断当前知识点的掌握水平 | 规则引擎 + 学生画像 |
| `analyze_gaps` | 分析知识薄弱环节，按紧急度排序 | 学生画像知识图谱 |
| `personalize_path` | 生成个性化学习路径 | 学生画像 + Milvus |
| `knowledge_map` | 查询知识点前置依赖和延伸关系 | Milvus |
| `curriculum_standard` | 检索课程标准/考纲要求 | Milvus |
| `deep_research` | 深度教育研究（多来源证据综合） | 网络搜索 + 知识库 |

### 3 个教育 Agent

#### TutorAgent — 自适应学习导师

根据学生的学习风格和掌握度动态调整教学：

- 识别 5 种学习风格（图形/文本/例题/推理/实践），匹配最佳讲解方式
- 3 档节奏调节（慢/正常/快），根据学生反馈实时切换
- 讲-问-练循环（单向讲解不超过 3 轮）
- 检测情绪信号（困惑/掌握/投入），即时响应

常用 Skills: `search_content`, `personalize_path`, `knowledge_map`

#### ProgressAgent — 学生进步指南

基于知识图谱的进步追踪：

- 分析知识点的掌握度分布和薄弱环节
- 识别前置依赖链（"你 A 还不会，所以 B 比较吃力"）
- 规划下一步学习路径（先学什么、为什么）
- 生成可量化的短期目标

常用 Skills: `analyze_gaps`, `knowledge_map`, `curriculum_standard`

#### AssessAgent — 记忆辅助评估

基于艾宾浩斯遗忘曲线的科学复习：

- 间隔复习调度（刚学→1天/练习中→3天/已掌握→7天/熟练→30天）
- 区分表面掌握和真实掌握（短期记忆 vs 长期记忆）
- 错误模式分类（概念混淆/计算错误/逻辑错误/审题错误）
- 回忆质量评估（不是对错，而是"多想了一会儿才想起来"）

常用 Skills: `assess_level`, `search_content`, `knowledge_map`

---

## Student Profile — 学生画像

EduX 与医疗版最大的架构差异。每个学生拥有持久化画像：

```
Student Profile
├── 学习风格标签     visual / textual / example-driven / abstract / hands-on
├── 知识图谱快照     每个知识点的掌握度 + 置信度 + 前置依赖
├── 互动历史摘要     最近 N 次会话的难度、情绪、耗时
├── 错题轨迹         高频错误知识点、错误类型分布
└── 复习队列         基于艾宾浩斯曲线的间隔复习调度
```

画像在每次会话开始时由 AssessAgent 加载，TutorAgent 消费用于自适应教学，ProgressAgent 在会话结束时更新。

学习风格通过贝叶斯式更新逐步收敛：每次互动检测到风格信号后微调权重，3-5 次会话后达到高置信度。

---

## 技术架构

### Skills 直达架构

```
Skills (函数) → 直接转换 → OpenAI function calling format → LLM 调用
         ↓
    学生画像 / Milvus / 业务逻辑
```

### 三 Agent 协作流程

```
学生提问
   ↓
学习风格检测 (detect_style_from_text)
   ↓
SwarmCoordinator 智能路由
   ↓
┌───────────────┴───────────────┐
│ 简单问题 → 单 Agent            │  复杂问题 → 多 Agent Swarm
│                               │
│ TutorAgent                    │  LeadAgent 分解任务
│ (自适应讲解)                   │       ↓
│                               │  ┌──────┼──────┐
│                               │  ↓      ↓      ↓
│                               │ Tutor Progress Assess
│                               │  └──────┼──────┘
│                               │       ↓
│                               │  LeadAgent 汇总
└───────────────────────────────┘
   ↓
更新 StudentProfile
   ↓
持久化到 Mem0 长期记忆
```

### Agent Loop (Think-Act-Observe)

```
┌─────────┐     ┌────────┐     ┌──────────┐
│  Think  │ ──> │  Act   │ ──> │  Observe │
└─────────┘     └────────┘     └──────────┘
     ↑                               │
     └───────────────────────────────┘
```

### 记忆系统

```
┌────────────────────────────────────┐
│  短期记忆（会话级，内存/Redis）      │
│  - 当前会话对话历史                 │
│  - 保留时间：60 分钟                │
└────────────────────────────────────┘
           ↕ 会话结束时
┌────────────────────────────────────┐
│  长期记忆（跨会话，Mem0）           │
│  - Student Profile（学生画像）      │
│  - 会话总结                        │
│  - 相似历史会话检索                 │
└────────────────────────────────────┘
```

---

## Harness Engineering

"人类设计约束，AI 代理执行"——让 AI 在明确约束下自主工作、自我修正。

| 原则 | EduX 实现 |
|------|----------|
| **约束驱动** | YAML 定义教育边界（不代写作业、不保证分数、不制造焦虑），运行时验证 |
| **自动修复** | 输出违规自动添加辅导声明、教师求助建议 |
| **熵管理** | 记忆自动去重和压缩，防止系统膨胀 |

核心约束：
- 不能替学生完成作业（只能引导思路）
- 不能直接给答案（必须引导学生自己得出）
- 不能保证考试分数或升学结果
- 持续学习困难必须建议联系教师
- 不能制造焦虑或贴标签

---

## 个人化闭环示例

```
第 1 次会话 — 学生："三角函数我总是记不住公式"
  → 风格检测：倾向 visual（提到了"记不住"，偏好图形）
  → TutorAgent：用单位圆图解 + 对比记忆法
  → AssessAgent：安排 1 天后复习正弦、余弦、正切定义
  → StudentProfile：记录首次互动，风格 confidence=0.3

第 2 次会话（3天后）— 学生："上次那个单位圆挺好的，再讲讲诱导公式"
  → 风格确认：visual 信号加强，confidence→0.5
  → AssessAgent：上次内容已进入遗忘区，触发复习提醒
  → TutorAgent：先快速复习单位圆，再讲诱导公式（继续用图形）
  → ProgressAgent：三角-单位圆→PROFICIENT，三角-诱导公式→INTRODUCED

第 3 次会话（7天后）— 学生："和二倍角公式结合起来还是有点乱"
  → 风格高置信度：visual（0.75）
  → AssessAgent：诱导公式需复习（7天间隙），错误模式：概念混淆
  → ProgressAgent：知识图谱显示需先巩固和差公式
  → TutorAgent：用函数图像对比和差公式和二倍角的关系
```

---

## 配置

```python
# config.py
LLM_CONFIG = {
    "api_key": "your-api-key",
    "model_name": "your-model",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}

MEM0_CONFIG = {"api_key": "m0-your-api-key-here"}  # https://app.mem0.ai
```

未配置 Mem0 时系统优雅降级，使用短期记忆继续工作（学生画像仅存在于当前会话）。

---

## 知识库

- **向量数据库**: Milvus Lite（本地文件，无需服务器）
- **Embedding 模型**: BAAI/bge-small-zh-v1.5（中文，512 维）
- **数据存储**: `knowledge/data/documents/`（txt 文档）
- **初始化**: `python knowledge/scripts/import_edu_data.py`

---

## 免责声明

EduX 是学习辅导工具，旨在帮助学生理解知识点和规划学习路径。不能替代专业教师的教学。如遇持续性学习困难，建议与学校老师沟通。

## 许可证

MIT License
