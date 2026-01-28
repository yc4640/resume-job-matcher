# LM Match Service

**简体中文** | **[English](README.md)**

## 项目简介

LM Match Service 是一个基于 FastAPI 的求职简历匹配服务。本项目目前处于 M7 阶段,在可解释排序、RAG 解释、评估体系和 Streamlit 交互界面的基础上,新增了完整的 Learning to Rank (LTR) 系统,通过机器学习优化排序效果,提供更精准的职位推荐。

### 当前功能

#### M1:基础匹配功能
- ✅ 健康检查接口 (`/health`)
- ✅ 职位-简历匹配接口 (`/match`) - 返回结构化匹配结果
- ✅ 使用 Pydantic 定义数据模型(JobPosting、Resume、MatchResponse)
- ✅ 基于技能集合的匹配算法(不使用 LLM)
- ✅ 提供匹配分数、匹配技能、技能差距和学习建议

#### M2:语义推荐功能
- ✅ 职位推荐接口 (`/recommend_jobs`) - 基于语义相似度的 Top-K 推荐
- ✅ 使用 sentence-transformers 本地模型进行文本嵌入
- ✅ 余弦相似度计算和排序
- ✅ 批量职位数据集(jobs.jsonl)和简历数据集(resumes.jsonl)
- ✅ 完全本地运行,无需付费 API

#### M3:可解释排序功能
- ✅ 轻量排序层 - 在 embedding 召回基础上引入多维度打分
- ✅ 技能词表 (180+ 技能) - 标准化技能匹配
- ✅ YAML 配置 - 无需修改代码即可调整排序权重
- ✅ 多维度特征:
  - `embedding`: 语义相似度(embedding score)
  - `skill_overlap`: 技能覆盖率
  - `keyword_bonus`: 关键字命中加分
  - `gap_penalty`: 缺失关键技能惩罚
- ✅ 可解释性 - 自动生成排名第一的详细解释

#### M4:RAG 可解释层
- ✅ 证据构建 - 从职位和简历中提取结构化证据
- ✅ 智能检索 - 基于语义相似度选择最相关的证据片段
- ✅ LLM 生成 - 使用大语言模型生成基于证据的解释
- ✅ 三维分析 - 为每个推荐职位提供:
  - `explanation`: 为什么这个岗位适合候选人
  - `gap_analysis`: 候选人缺少哪些关键技能或资质
  - `improvement_suggestions`: 具体可行的提升建议
- ✅ 防止幻觉 - 严格基于证据生成,LLM 仅用于解释层,不参与排序
- ✅ **技能自动提取与合并** - 从简历文本(education/projects/experience)中自动提取技能,避免过度严格的匹配
- ✅ **软技能过滤** - 软技能(如 Communication、Leadership)缺失不计入 gap_penalty

#### M5:评估与弱监督标签生成（旧版，部分已被 M7 替代）
- ✅ 数据 ID 对齐 - jobs.jsonl 和 resumes.jsonl 添加 job_id、resume_id
- ⚠️ LLM 辅助标签生成 - 使用 GPT-4o-mini 为 Top-15 推荐生成 0-3 分级标签（**M7 已升级为全量 1-5 标签**）
- ✅ 弱监督标签(Weak Labels) - 快速生成大规模标注数据
- ✅ 评估指标实现:
  - Precision@K - 衡量推荐精准度
  - NDCG@K - 衡量排序质量(考虑位置权重)
- ⚠️ 评估方法 - 简单的标签验证（**M7 已升级为 LOOCV + Ablation Study**）
- ❌ **已弃用文件**: labels_final.csv（人工校正模板）、run_eval.py（评估脚本）、eval_results.json（评估结果）

#### M6:Streamlit 交互界面
- ✅ Streamlit Web 界面 - 轻量级交互式前端
- ✅ 多种简历输入方式 - 文本框输入或上传 TXT 文件
- ✅ 职位选择 - 从 jobs.jsonl 数据库选择
- ✅ Top-K 参数配置 - 灵活调整推荐数量
- ✅ 一键匹配 - 调用后端 `/recommend_jobs` 接口
- ✅ 可视化结果展示 - 职位信息、匹配分数、技能对比
- ✅ 详细解释生成 - 点击按钮调用 `/explain` 接口
- ✅ 后端状态监控 - 实时检查后端服务可用性

#### M7:Learning to Rank (LTR) 系统
- ✅ 全量 Weak Labels(1-5 scale) - 覆盖所有 resume×job 组合(750 pairs: 15 resumes × 50 jobs)
- ✅ Pairwise LTR 训练 - 基于 Logistic Regression 的排序模型
- ✅ LOOCV 评估 - Leave-One-Out Cross-Validation(小数据必备)
- ✅ Ablation Study - 对比 embedding_only、heuristic、LTR 三种排序方法
- ✅ 评估指标 - NDCG@5/10、Precision@5/10
- ✅ FastAPI use_ltr 开关 - 前端可切换是否启用 LTR 排序
- ✅ Streamlit LTR 切换 - UI 上一键开启/关闭 LTR 功能
- ✅ 模型持久化 - joblib 保存/加载 LTR 模型

#### 通用特性
- ✅ RESTful API 设计
- ✅ 自动生成的 API 文档(Swagger UI / ReDoc)

## 项目结构

```
lm/
├── backend/
│   ├── main.py              # FastAPI 主应用文件
│   ├── schemas.py           # Pydantic 数据模型定义
│   ├── test_match.py        # 匹配接口测试文件
│   ├── requirements.txt     # Python 依赖
│   ├── .env.example         # 环境变量配置示例 (M4)
│   ├── services/            # 业务逻辑服务
│   │   ├── __init__.py         # 服务包初始化
│   │   ├── embedding.py        # 文本嵌入服务 (M2)
│   │   ├── retrieval.py        # 检索和排序服务 (M2)
│   │   ├── ranking.py          # 可解释排序服务 (M3)
│   │   ├── rag.py              # RAG 可解释层服务 (M4)
│   │   └── utils.py            # 工具函数(技能提取与合并) (M4.1)
│   ├── src/                 # LTR 源码模块 (M7 新增)
│   │   └── ranking/
│   │       ├── __init__.py     # 排序包初始化
│   │       ├── features.py     # 特征提取与向量化
│   │       ├── pairwise.py     # Pairwise 训练数据构造
│   │       └── ltr_logreg.py   # Pairwise Logistic Regression 模型
│   ├── scripts/             # 脚本目录 (M7 新增)
│   │   └── eval_ablation.py    # LOOCV + Ablation 评估脚本
│   ├── models/              # 模型保存目录 (M7 新增)
│   │   └── ltr_logreg.joblib   # 训练好的 LTR 模型
│   ├── results/             # 评估结果目录 (M7 新增)
│   │   └── ablation_results.json  # Ablation study 结果
│   ├── config/              # 配置文件 (M3 新增)
│   │   └── ranking_config.yaml # 排序权重配置
│   ├── eval/                # 评估模块 (M5/M7 更新)
│   │   ├── generate_labels.py  # 全量 1-5 weak labels 生成脚本 (M7 更新)
│   │   ├── labels_suggested.jsonl  # 全量 1-5 标签 (M7: 750 pairs)
│   │   ├── labels_final.csv    # 人工校正模板(已弃用)
│   │   ├── metrics.py          # 评估指标(Precision@K, NDCG@K)
│   │   ├── run_eval.py         # 评估运行脚本(已弃用,改用 scripts/eval_ablation.py)
│   │   ├── eval_results.json   # 评估结果(已弃用)
│   │   └── eval_report.md      # 评估报告 (M7 更新)
│   └── data/
│       ├── sample_job.json        # 示例职位数据
│       ├── sample_resume.json     # 示例简历数据
│       ├── jobs.jsonl             # 批量职位数据(50条,含 job_id) (M5/M7)
│       ├── resumes.jsonl          # 批量简历数据(15条,含 resume_id) (M5/M7)
│       └── skills_vocabulary.txt  # 技能词表(180+ 技能) (M3)
├── frontend/                # 前端界面 (M6 新增)
│   ├── streamlit_app.py     # Streamlit 交互界面
│   └── requirements.txt     # 前端依赖(Streamlit, requests)
├── .gitignore               # Git 忽略文件配置
└── README.md                # 项目说明文档
```

## 如何运行

### 1. 环境要求

- Python 3.8+
- pip

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 4. 配置环境变量(M4 新增)

为了使用 RAG 可解释层功能,需要配置 OpenAI API Key:

```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑 .env 文件,填入你的 OpenAI API Key
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

**获取 OpenAI API Key:**
1. 访问 https://platform.openai.com/api-keys
2. 登录或注册 OpenAI 账号
3. 创建新的 API Key
4. 将 API Key 填入 `.env` 文件

**注意:** 如果不配置 API Key,推荐接口仍可正常工作,但每个推荐职位的 `explanation`、`gap_analysis` 和 `improvement_suggestions` 字段将为 `null`。

### 5. 启动服务

```bash
# 方式一:使用 uvicorn 命令
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 方式二:直接运行 main.py
python main.py
```

服务启动后,访问 http://localhost:8000

### 6. 查看 API 文档

FastAPI 自动生成交互式 API 文档:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 数据模型说明

### JobPosting(职位信息)

```json
{
  "title": "职位名称",
  "responsibilities": "岗位职责描述",
  "requirements_text": "任职要求描述",
  "skills": ["技能1", "技能2", "..."],
  "company": "公司名称(可选)",
  "location": "工作地点(可选)",
  "level": "职位级别(可选)"
}
```

### Resume(简历信息)

```json
{
  "education": "教育背景",
  "projects": "项目经历",
  "skills": ["技能1", "技能2", "..."],
  "experience": "工作经验"
}
```

### MatchResponse(匹配结果)

```json
{
  "match_score": 57,
  "matched_skills": ["Python", "FastAPI", "Docker"],
  "gaps": ["PostgreSQL", "Kubernetes", "Redis", "AWS"],
  "suggestions": [
    "Consider learning PostgreSQL to better match this position",
    "Consider learning Kubernetes to better match this position",
    "..."
  ]
}
```

## 示例数据

### 示例职位数据 (backend/data/sample_job.json)

```json
{
  "title": "Senior Backend Engineer",
  "responsibilities": "Design and implement scalable backend services, lead technical architecture decisions, mentor junior developers, and collaborate with cross-functional teams to deliver high-quality software solutions.",
  "requirements_text": "5+ years of backend development experience, strong knowledge of Python and web frameworks, experience with databases and cloud platforms, excellent problem-solving skills.",
  "skills": [
    "Python",
    "FastAPI",
    "PostgreSQL",
    "Docker",
    "Kubernetes",
    "Redis",
    "AWS"
  ],
  "company": "TechCorp Inc.",
  "location": "San Francisco, CA / Remote",
  "level": "Senior"
}
```

### 示例简历数据 (backend/data/sample_resume.json)

```json
{
  "education": "Bachelor of Science in Computer Science, Stanford University, 2015-2019. Relevant coursework: Data Structures, Algorithms, Database Systems, Distributed Systems.",
  "projects": "1) E-commerce Platform - Built a scalable e-commerce backend using Python and FastAPI, serving 100k+ daily users. Implemented RESTful APIs, payment integration, and order management system. 2) Real-time Chat Application - Developed a real-time messaging system using WebSocket, Redis pub/sub, and MongoDB for message persistence. 3) DevOps Automation - Created CI/CD pipelines using Docker and GitHub Actions to automate deployment processes.",
  "skills": [
    "Python",
    "FastAPI",
    "Django",
    "Docker",
    "MongoDB",
    "Git",
    "Linux"
  ],
  "experience": "Software Engineer at StartupXYZ (2019-2023): Developed and maintained backend services using Python and FastAPI. Designed database schemas and optimized query performance. Collaborated with frontend team to integrate APIs. Implemented automated testing and deployment pipelines using Docker. Mentored 2 junior developers."
}
```

### 数据集(JSONL 格式)

为了支持后续的 top-k 推荐功能,我们提供了两个 JSON Lines 格式的数据集:

#### backend/data/jobs.jsonl
- 包含 50 条真实的职位信息(M7 扩展)
- 涵盖技能领域:推荐系统、搜索、NLP、LLM、CV、数据工程、后端开发、机器学习等
- 每行一个 JSON 对象,符合 `JobPosting` schema

#### backend/data/resumes.jsonl
- 包含 15 条不同背景的简历(M7 扩展)
- 技能与职位数据有不同程度的重叠,适合测试匹配算法
- 每行一个 JSON 对象,符合 `Resume` schema


#### 预期使用场景

这些 JSONL 数据集将在后续 Milestone 中用于:
1. **批量匹配测试**:测试系统处理多个职位和简历的性能
2. **Top-k 推荐**:为给定简历推荐最匹配的 k 个职位(或反向推荐)
3. **排序算法验证**:验证基于匹配分数的排序逻辑
4. **性能基准测试**:测试大规模匹配的响应时间和准确性

## 如何测试接口

### 测试健康检查接口

**使用 curl:**
```bash
curl http://localhost:8000/health
```

**预期响应:**
```json
{
  "status": "ok",
  "message": "Service is healthy and running"
}
```

### 测试匹配接口

#### 方式一:使用 Swagger UI(推荐)

1. 访问 http://localhost:8000/docs
2. 找到 `POST /match` 接口
3. 点击 **"Try it out"** 按钮
4. 在 Request body 中粘贴以下 JSON:

```json
{
  "job": {
    "title": "Senior Backend Engineer",
    "responsibilities": "Design and implement scalable backend services, lead technical architecture decisions, mentor junior developers, and collaborate with cross-functional teams to deliver high-quality software solutions.",
    "requirements_text": "5+ years of backend development experience, strong knowledge of Python and web frameworks, experience with databases and cloud platforms, excellent problem-solving skills.",
    "skills": [
      "Python",
      "FastAPI",
      "PostgreSQL",
      "Docker",
      "Kubernetes",
      "Redis",
      "AWS"
    ],
    "company": "TechCorp Inc.",
    "location": "San Francisco, CA / Remote",
    "level": "Senior"
  },
  "resume": {
    "education": "Bachelor of Science in Computer Science, Stanford University, 2015-2019. Relevant coursework: Data Structures, Algorithms, Database Systems, Distributed Systems.",
    "projects": "1) E-commerce Platform - Built a scalable e-commerce backend using Python and FastAPI, serving 100k+ daily users. Implemented RESTful APIs, payment integration, and order management system. 2) Real-time Chat Application - Developed a real-time messaging system using WebSocket, Redis pub/sub, and MongoDB for message persistence. 3) DevOps Automation - Created CI/CD pipelines using Docker and GitHub Actions to automate deployment processes.",
    "skills": [
      "Python",
      "FastAPI",
      "Django",
      "Docker",
      "MongoDB",
      "Git",
      "Linux"
    ],
    "experience": "Software Engineer at StartupXYZ (2019-2023): Developed and maintained backend services using Python and FastAPI. Designed database schemas and optimized query performance. Collaborated with frontend team to integrate APIs. Implemented automated testing and deployment pipelines using Docker. Mentored 2 junior developers."
  }
}
```

5. 点击 **"Execute"** 按钮执行请求
6. 查看响应结果

**预期响应示例:**
```json
{
  "match_score": 42,
  "matched_skills": [
    "Python",
    "FastAPI",
    "Docker"
  ],
  "gaps": [
    "PostgreSQL",
    "Kubernetes",
    "Redis",
    "AWS"
  ],
  "suggestions": [
    "Consider learning PostgreSQL to better match this position",
    "Consider learning Kubernetes to better match this position",
    "Consider learning Redis to better match this position",
    "Consider learning AWS to better match this position"
  ]
}
```

#### 方式二:使用 curl

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "job": {
      "title": "Senior Backend Engineer",
      "responsibilities": "Design and implement scalable backend services",
      "requirements_text": "5+ years of backend development experience",
      "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
      "company": "TechCorp Inc.",
      "location": "Remote",
      "level": "Senior"
    },
    "resume": {
      "education": "BS Computer Science",
      "projects": "Built e-commerce platform",
      "skills": ["Python", "FastAPI", "MongoDB"],
      "experience": "4 years backend development"
    }
  }'
```

#### 方式三:使用 Python 测试脚本

运行项目自带的测试脚本:

```bash
cd backend
python test_match.py
```

该脚本包含多个测试用例,涵盖完全匹配、部分匹配、不匹配等场景。

### 测试职位推荐接口(M2 新增)

#### 方式一:使用 Swagger UI(推荐)

1. 访问 http://localhost:8000/docs
2. 找到 `POST /recommend_jobs` 接口
3. 点击 **"Try it out"** 按钮
4. 在 Request body 中粘贴以下 JSON(使用示例简历数据):

```json
{
  "resume": {
    "education": "Master of Science in Natural Language Processing, Carnegie Mellon University, 2019-2021. Bachelor of Science in Linguistics and Computer Science, University of Washington, 2015-2019. Relevant coursework: Deep Learning for NLP, Statistical NLP, Computational Semantics, Machine Translation.",
    "projects": "1) Conversational AI System - Built chatbot using GPT-4 and RAG, serving 500K+ users with 90% satisfaction rate. Implemented custom fine-tuning pipeline and prompt engineering framework. 2) Multilingual NER System - Developed named entity recognition system supporting 15 languages using BERT and mBERT. 3) Text Summarization Tool - Created abstractive summarization model fine-tuned on domain-specific data, deployed to production with FastAPI backend. 4) LLM Evaluation Framework - Built comprehensive evaluation pipeline for testing LLM outputs across multiple dimensions.",
    "skills": [
      "NLP",
      "LLM",
      "Transformers",
      "BERT",
      "GPT",
      "Claude",
      "Prompt Engineering",
      "RAG",
      "Fine-tuning",
      "Python",
      "spaCy",
      "Langchain",
      "PyTorch",
      "FastAPI"
    ],
    "experience": "NLP Engineer at AI Startup (2021-2024): Built LLM-powered products, implemented RAG systems, fine-tuned models for domain adaptation. NLP Research Intern at Microsoft (Summer 2020): Worked on transformer models for multilingual understanding, contributed to internal NLP libraries."
  },
  "top_k": 3
}
```

5. 点击 **"Execute"** 按钮执行请求
6. 查看响应结果

**预期响应示例:**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "title": "NLP Engineer - Conversational AI",
      "company": "ChatBot Solutions",
      "location": "Austin, TX",
      "level": "Mid-level",
      "similarity_score": 0.682073712348938,
      "matched_skills": [
        "Python",
        "Transformers",
        "Prompt Engineering",
        "NLP",
        "LLM",
        "BERT",
        "spaCy"
      ],
      "gap_skills": [],
      "features": {
        "embedding": 0.682073712348938,
        "skill_overlap": 1,
        "keyword_bonus": 0.85,
        "gap_penalty": 0,
        "final_score": 0.7428294849395752
      }
    },
    {
      "rank": 2,
      "title": "LLM Engineer",
      "company": "AI Startup",
      "location": "Remote",
      "level": null,
      "similarity_score": 0.6174665093421936,
      "matched_skills": [
        "Python",
        "Prompt Engineering",
        "LLM",
        "Claude",
        "RAG",
        "GPT",
        "Fine-tuning",
        "Langchain"
      ],
      "gap_skills": [
        "Vector Databases"
      ],
      "features": {
        "embedding": 0.6174665093421936,
        "skill_overlap": 0.8888888888888888,
        "keyword_bonus": 0.9,
        "gap_penalty": 0.1,
        "final_score": 0.6836532704035442
      }
    },
    {
      "rank": 3,
      "title": "NLP Research Scientist",
      "company": "AI Research Lab",
      "location": "Remote",
      "level": "Senior",
      "similarity_score": 0.6600039005279541,
      "matched_skills": [
        "Python",
        "PyTorch",
        "Transformers",
        "NLP",
        "GPT",
        "BERT"
      ],
      "gap_skills": [
        "Deep Learning",
        "Research"
      ],
      "features": {
        "embedding": 0.6600039005279541,
        "skill_overlap": 0.75,
        "keyword_bonus": 0.7,
        "gap_penalty": 0.2,
        "final_score": 0.6090015602111816
      }
    }
  ],
  "total_jobs_searched": 50,
}
```

**说明(M4 更新):**
- `similarity_score`:基于语义嵌入的余弦相似度(0-1之间,等同于 embedding_score)
- `matched_skills`:简历技能与职位要求技能的交集(基于标准化技能词表)
- `gap_skills`:职位要求但简历缺失的技能(M3 新增)
- `features`:可解释的排序特征(M3 新增)
  - `embedding`:语义相似度(0-1)
  - `skill_overlap`:技能覆盖率(0-1)
  - `keyword_bonus`:关键词加分(0-1)
  - `gap_penalty`:缺失惩罚(0-1)
  - `final_score`:综合得分(加权计算)
- `explanation`:排名第一职位的详细解释(M3 新增)
- **M4 新增字段(每个推荐职位):**
  - `explanation`:为什么这个岗位适合候选人(基于证据的解释)
  - `gap_analysis`:候选人缺少哪些关键技能或资质
  - `improvement_suggestions`:具体可行的提升建议
- `total_jobs_searched`:从 jobs.jsonl 加载的总职位数量

**M4 返回示例(单个推荐职位):**
```json
{
  "rank": 1,
  "title": "NLP Engineer - Conversational AI",
  "company": "ChatBot Solutions",
  "location": "Austin, TX",
  "level": "Mid-level",
  "similarity_score": 0.682,
  "matched_skills": ["Python", "Transformers", "NLP", "LLM"],
  "gap_skills": [],
  "features": {
    "embedding": 0.682,
    "skill_overlap": 1.0,
    "keyword_bonus": 0.85,
    "gap_penalty": 0.0,
    "final_score": 0.743
  },
  "explanation": "这个职位非常适合你,因为你构建对话式 AI 系统(使用 GPT-4 和 RAG)的经验直接符合该岗位的核心要求。你的项目展示了 NLP 和 LLM 应用的实践专长,特别是在处理大规模用户交互(50万+用户)方面。",
  "gap_analysis": "虽然你拥有扎实的 NLP 基础,但该职位要求对话系统和意图识别框架的经验,这些在你的简历中没有明确提及。此外,使用特定聊天机器人框架的生产级部署经验将增强你的竞争力。",
  "improvement_suggestions": "- 使用 Rasa 或类似框架构建对话管理系统,以展示意图识别能力\n- 完成一个专注于多轮对话处理和上下文管理的项目\n- 记录你在生产聊天机器人环境中进行 A/B 测试和性能优化的经验"
}
```

#### 方式二:使用 curl

```bash
curl -X POST http://localhost:8000/recommend_jobs \
  -H "Content-Type: application/json" \
  -d '{
    "resume": {
      "education": "BS Computer Science",
      "projects": "Built recommendation systems and ML models",
      "skills": ["Python", "Machine Learning", "TensorFlow", "Recommendation Systems"],
      "experience": "3 years as ML Engineer"
    },
    "top_k": 3
  }'
```

#### 推荐接口特点(M3 增强)

- **语义匹配 (M2)**:使用 sentence-transformers 本地模型(all-MiniLM-L6-v2)进行文本嵌入
- **多维度排序 (M3)**:结合语义相似度、技能覆盖率、关键词加分、缺失惩罚的综合打分
- **可解释性 (M3)**:自动生成排名第一职位的详细解释,说明为什么它最匹配
- **灵活配置 (M3)**:通过 YAML 配置文件调整排序权重,无需修改代码
- **标准化技能 (M3)**:基于 180+ 技能词表进行标准化匹配
- **无需付费 API**:完全本地运行,无需调用外部 API
- **技能重叠信息**:提供精确的匹配技能和缺失技能列表

## 技术栈

- **FastAPI**: 现代、高性能的 Python Web 框架
- **Pydantic**: 数据验证和设置管理
- **Uvicorn**: ASGI 服务器
- **Sentence-Transformers**: 本地文本嵌入模型(M2)
- **NumPy**: 向量计算和相似度计算(M2)
- **PyYAML**: 配置文件管理(M3)
- **OpenAI API**: LLM 生成解释文本(M4)

## 匹配算法说明

### M1:基于技能集合的精确匹配

使用集合运算进行技能匹配:

1. **匹配技能** (matched_skills):求职者技能与职位要求技能的交集
2. **技能差距** (gaps):职位要求技能中求职者不具备的技能
3. **匹配分数** (match_score):匹配技能数量占职位要求技能总数的百分比
   - 公式:`match_score = (len(matched_skills) / len(job.skills)) * 100`
   - 如果职位没有技能要求,则返回 0
4. **学习建议** (suggestions):针对每个技能差距提供学习建议

### M2:基于语义嵌入的推荐系统

使用 sentence-transformers 进行语义相似度匹配:

1. **文本嵌入**:
   - 模型:all-MiniLM-L6-v2(384维向量,本地运行)
   - 职位文本:拼接 title + responsibilities + requirements_text + skills
   - 简历文本:拼接 education + projects + experience + skills

2. **相似度计算**:
   - 使用余弦相似度(Cosine Similarity)计算简历与职位的语义相似度
   - 相似度范围:0-1,越接近1表示越相似

3. **Top-K 推荐**:
   - 根据相似度分数降序排序
   - 返回最匹配的 top-k 个职位
   - 附带精确的技能重叠信息(复用 M1 逻辑)

### M3:可解释的轻量排序层

在 M2 embedding 召回基础上,引入多维度打分机制:

#### 1. 排序特征

- **embedding (语义相似度)**:
  - 来自 M2 的文本嵌入余弦相似度
  - 范围:0-1

- **skill_overlap (技能覆盖率)**:
  - 基于标准化技能词表(180+ 技能)的匹配率
  - 公式:`matched_skills / job_required_skills`
  - 范围:0-1

- **keyword_bonus (关键词加分)**:
  - 高优先级技能匹配加分(如 Python、Machine Learning、LLM 等)
  - 高优先级技能权重 1.5x
  - 归一化到 0-1 范围

- **gap_penalty (缺失惩罚)**:
  - 缺失关键技能的惩罚(如 Python、SQL 等核心技能)
  - 关键技能缺失权重 2.0x
  - 归一化到 0-1 范围

#### 2. 打分公式

```
final_score = w1 * embedding
            + w2 * skill_overlap
            + w3 * keyword_bonus
            - w4 * gap_penalty
```

默认权重配置(可通过 YAML 调整):
- `w1 (embedding)`: 0.4
- `w2 (skill_overlap)`: 0.3
- `w3 (keyword_bonus)`: 0.2
- `w4 (gap_penalty)`: 0.1

#### 3. 配置文件

排序权重通过 `config/ranking_config.yaml` 配置,支持:
- 调整各特征权重
- 定义高优先级关键词列表
- 定义关键技能列表
- 调整奖惩倍数
- **无需修改代码即可调整排序策略**

#### 4. 可解释性

系统自动生成排名第一职位的详细解释,包括:
- 各维度特征分数
- 匹配技能列表
- 缺失技能列表
- 综合得分计算过程

示例解释输出:
```
【NLP Engineer - Conversational AI】排名第一的原因:

1. 语义相似度: 0.682 (权重: 0.4)
   - 职位描述与简历内容高度匹配

2. 技能覆盖率: 0.875 (权重: 0.3)
   - 匹配技能 (7个): NLP, Prompt Engineering, Python, ...
   - 缺失技能 (1个): Dialogue Systems

3. 关键词加分: 0.650 (权重: 0.2)
   - 匹配高优先级技能

4. 缺失惩罚: 0.100 (权重: 0.1)
   - 缺失关键技能的惩罚

综合得分: 0.723
```

### M4.1:技能自动提取与合并(Skills Auto-Extract & Merge)

#### 问题背景

在传统的技能匹配中,系统仅依赖用户在 `resume.skills` 列表中明确列出的技能。这会导致以下问题:

1. **过度严格的匹配**:很多技能实际上在简历的 `experience`、`projects` 或 `education` 中提到,但未在 `skills` 列表中列出
2. **误判技能缺口**:例如简历中提到 "conducted NER research" 或 "published papers on entity extraction",但因为 `skills` 列表没写 "NER" 或 "Entity Extraction",就被判定为缺失技能

#### 解决方案

系统自动从简历文本中提取技能,并与用户提供的技能列表合并:

**核心逻辑:**
```
merged_skills = union(
    user_provided_resume.skills,
    extracted_skills_from_resume_text
)
```

**提取流程:**
1. **文本组装**:将 `resume.education`、`resume.projects`、`resume.experience` 组合成一段文本
2. **词汇匹配**:基于 `skills_vocabulary.txt`(包含 180+ 技能词)进行匹配
3. **智能边界检测**:使用正则表达式的词边界(`\b`),避免误匹配(例如 "C" 不会匹配 "Cloud", "React" 不会匹配 "Reactivity")
4. **特殊字符处理**:正确处理 "C++"、"C#"、".NET" 等包含特殊字符的技能
5. **大小写规范化**:匹配时忽略大小写,但保留词汇表中的原始大小写
6. **去重合并**:将提取的技能与用户提供的技能合并,去重后返回

**示例:**
```python
# 用户提供的技能
resume.skills = ["Python", "Machine Learning"]

# 简历文本中提到的内容
resume.projects = "Conducted research on NER and entity extraction..."
resume.experience = "Published papers on Named Entity Recognition..."

# 自动提取的技能
extracted_skills = ["NER", "Entity Extraction", "Research", "Publication"]

# 最终合并后的技能(用于匹配)
merged_skills = ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication"]
```

#### 软技能过滤

为了避免对候选人过度惩罚,系统在计算 `gap_penalty` 时会**过滤掉软技能**:

**软技能列表**(不计入缺失惩罚):
- Communication(沟通)
- Leadership(领导力)
- Collaboration(协作)
- Teamwork(团队合作)
- Problem Solving(问题解决)
- Critical Thinking(批判性思维)
- Time Management(时间管理)
- Adaptability(适应性)
- 等等...

**为什么过滤软技能?**
- 软技能很重要,但缺失不应该像技术技能那样被严重扣分
- 软技能难以在简历中量化,容易被遗漏
- 软技能更多是在面试中评估,而非简历筛选阶段的硬性要求

**注意:** 软技能仍然会:
- ✅ 出现在 `matched_skills` 中(如果匹配)
- ✅ 出现在 `gap_skills` 中(如果缺失)
- ✅ 可用于 `keyword_bonus` 加分
- ✅ 出现在 RAG 解释的 evidence 中
- ❌ **不会**计入 `gap_penalty` 扣分

#### 实现位置

**新增文件:** `backend/services/utils.py`
- `extract_skills_from_text(text, vocab)` - 从文本中提取技能
- `merge_resume_skills(resume, vocab)` - 合并用户技能与提取技能
- `filter_soft_skills(skills)` - 过滤软技能
- `SOFT_SKILLS` - 软技能常量集合

**调用位置:** `backend/services/ranking.py` 的 `rank_jobs_with_features` 函数
```python
# === SKILLS AUTO-EXTRACT & MERGE ===
# Line 247-255
vocab_list = list(vocab)
merged_skills = merge_resume_skills(resume, vocab_list)
resume_skills_normalized = normalize_skills(merged_skills, vocab)
```

**使用位置:**
- ✅ `matched_skills` 计算 - 使用 merged skills
- ✅ `gap_skills` 计算 - 使用 merged skills
- ✅ `skill_overlap` 计算 - 使用 merged skills
- ✅ `keyword_bonus` 计算 - 使用 merged skills
- ✅ `gap_penalty` 计算 - 使用 merged skills(过滤软技能后)

#### 验收示例

**场景:** 简历中提到了 NER 研究,但未在 skills 列表中列出

```json
{
  "resume": {
    "skills": ["Python", "Machine Learning"],
    "projects": "Built NER system for entity extraction in medical texts",
    "experience": "Conducted research on Named Entity Recognition, published 2 papers",
    "education": "Thesis: Literature review of state-of-the-art NER methods"
  }
}
```

**旧行为(问题):**
- `matched_skills`: ["Python", "Machine Learning"]
- `gap_skills`: ["NER", "Entity Extraction", "Research", "Publication"]  ❌ 误判为缺失

**新行为(修复):**
- `merged_skills`: ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication", "Literature Review"]
- `matched_skills`: ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication"]
- `gap_skills`: []  ✅ 正确识别

### M4:RAG 可解释层架构

#### RAG 在系统中的位置

RAG(Retrieval-Augmented Generation)层是 **纯解释层**,位于排序之后,**不参与职位排序逻辑**。整个推荐流程如下:

```
1. [M2 语义检索] 使用 embedding 计算所有职位与简历的相似度
           ↓
2. [M3 可解释排序] 基于多维度特征(embedding + skill + keyword + gap)计算最终得分并排序
           ↓
3. [M3 Top-K 选择] 选出排名前 K 的职位(排序已确定,不再改变)
           ↓
4. [M4 RAG 解释层] 对每个 Top-K 职位生成基于证据的解释
   ├─ 证据构建:提取职位和简历的结构化证据
   ├─ 智能检索:选择最相关的证据片段
   └─ LLM 生成:基于证据生成 explanation / gap_analysis / improvement_suggestions
           ↓
5. [返回结果] 包含排序、特征、RAG 解释的完整推荐结果
```

**关键约束:**
- M4 的 RAG 层 **仅用于生成解释文本**
- **不改变** M3 的 `final_score` 和排序顺序
- LLM 输出必须基于证据,禁止幻觉

#### RAG 的检索对象

RAG 检索的对象是 **职位和简历的文本片段(chunks)**,具体包括:

**职位证据(Job Evidence):**
- `title`:职位名称
- `responsibilities`:岗位职责
- `requirements_text`:任职要求
- `skills`:要求技能列表

**简历证据(Resume Evidence):**
- `education`:教育背景
- `projects`:项目经历
- `experience`:工作经验
- `skills`:技能列表

**检索流程:**
1. **文本分块(Chunking)**:将职位描述和简历内容按句子切分成小片段(约 200 字符)
2. **语义嵌入**:使用 sentence-transformers 模型对所有 chunks 计算向量表示
3. **相似度计算**:计算职位 chunks 与简历 chunks 之间的交叉相似度
4. **Top-K 选择**:选出最相关的 3 个职位 chunks 和 3 个简历 chunks 作为证据

**示例:**
- 职位 chunk: `[responsibilities] Design and implement scalable NLP systems for production chatbots.`
- 简历 chunk: `[projects] Built chatbot using GPT-4 and RAG, serving 500K+ users with 90% satisfaction rate.`
- 这两个 chunks 语义相似度高,会被选为证据传递给 LLM

#### LLM 在系统中的角色

LLM(大语言模型)**仅承担"解释生成"角色**,不参与任何排序或推荐决策:

**LLM 的职责:**
1. **阅读证据**:接收检索出的最相关职位和简历片段
2. **生成解释**:基于证据回答"为什么这个职位适合候选人"
3. **分析差距**:基于证据指出候选人缺少的关键技能
4. **提供建议**:给出具体可行的提升建议

**LLM 不做的事:**
- ❌ 不计算匹配分数(由 M3 ranking 层完成)
- ❌ 不决定职位排序(由 M3 final_score 决定)
- ❌ 不检索职位(由 M2 embedding 完成)
- ❌ 不评估技能匹配(由 M3 skill_overlap 完成)

**使用的 LLM 模型:**
- 默认:`gpt-4o-mini`(OpenAI)
- 优势:成本低、速度快、适合生成简短解释
- 温度设置:0.3(低温度保证输出稳定、事实性强)

#### 如何避免 LLM 编造内容

为了防止 LLM 幻觉(hallucination),我们采取了多层防护措施:

**1. 证据约束(Evidence Grounding)**
- LLM 只能看到通过检索选出的证据片段
- Prompt 明确要求:"Based ONLY on the evidence provided below"
- 禁止 LLM 添加未在证据中出现的信息

**2. 结构化 Prompt**
- 提供清晰的职位证据和简历证据
- 明确列出 `matched_skills` 和 `gap_skills`(由 M3 计算得出)
- 要求 LLM 引用具体证据内容

**3. 低温度生成**
- 设置 `temperature=0.3`(默认是 1.0)
- 低温度使输出更确定性、更贴近事实
- 减少创造性发挥,增强事实准确性

**4. 格式化输出**
- 要求 LLM 按照固定格式输出(EXPLANATION / GAP_ANALYSIS / IMPROVEMENT_SUGGESTIONS)
- 自动解析和验证输出格式
- 失败时回退到基于规则的简单解释

**5. 检索质量保证**
- 使用与 M2 相同的 sentence-transformers 模型进行检索
- 基于余弦相似度选择最相关的证据
- 确保传递给 LLM 的证据与职位-简历匹配度高

**Prompt 示例片段:**
```
CRITICAL RULES:
- Base your analysis ONLY on the evidence provided above
- Reference specific details from the job and resume evidence
- Do not make assumptions or add information not present in the evidence
- Keep each section concise and focused
```

**后备机制:**
如果 LLM API 调用失败(网络问题、API key 未设置等),系统会回退到基于规则的简单解释:
```python
{
    "explanation": "此职位匹配您的 4 项技能: Python, NLP, LLM, Transformers。总体兼容性得分为 0.68。",
    "gap_analysis": "您可能需要发展这些技能: Dialogue Systems, Intent Recognition。",
    "improvement_suggestions": "- 仔细审查职位要求\n- 考虑参加缺失技能的在线课程"
}
```

## M3 配置说明

### 排序权重配置

编辑 `backend/config/ranking_config.yaml` 调整排序策略:

```yaml
weights:
  embedding: 0.4        # 语义相似度权重
  skill_overlap: 0.3    # 技能覆盖率权重
  keyword_bonus: 0.2    # 关键词加分权重
  gap_penalty: 0.1      # 缺失惩罚权重

keywords:
  high_priority:        # 高优先级关键词
    - "Python"
    - "Machine Learning"
    - "LLM"
    # ... 更多
  high_priority_multiplier: 1.5  # 加分倍数

gap_penalty:
  critical_skills:      # 关键技能
    - "Python"
    - "SQL"
  critical_penalty_multiplier: 2.0  # 惩罚倍数
```

### 技能词表

`backend/data/skills_vocabulary.txt` 包含 200+ 标准化技能,涵盖:
- 编程语言(Python, Java, JavaScript, ...)
- Web 框架(FastAPI, Django, React, ...)
- ML/AI(Machine Learning, Deep Learning, TensorFlow, ...)
- NLP/LLM(Transformers, BERT, GPT, RAG, ...)
- 推荐/搜索(Recommendation Systems, Elasticsearch, ...)
- 数据工程(Spark, Airflow, ETL, ...)
- 云/基础设施(AWS, Docker, Kubernetes, ...)

可根据需要添加新技能到词表。

## M5 评估说明（旧版，已被 M7 替代）

> ⚠️ **注意**：M5 是初版评估方法，主要功能已被 M7 的 Learning to Rank (LTR) 系统替代。M7 使用全量 1-5 标签（750 对）和 LOOCV + Ablation 评估，比 M5 的 Top-15 部分标签（105 对，0-3 scale）更全面和严格。以下内容仅供参考历史实现。

### 评估目标（M5 旧版）

M5 引入了初版评估体系,用于量化职位推荐系统的性能:
- **数据对齐**:为 jobs.jsonl 和 resumes.jsonl 添加唯一 ID(job_id, resume_id)
- **弱监督标签**:使用 LLM(GPT-4o-mini)为 Top-15 推荐生成 0-3 分级标签（**已被 M7 的全量 1-5 标签替代**）
- **量化指标**:Precision@K 和 NDCG@K 衡量推荐质量
- **人工校正**:支持人工审核和修正 LLM 生成的标签（**M7 已弃用**）

### 标签体系(0-3 分级)（M5 旧版，M7 已改为 1-5 scale）

> ⚠️ **已过时**：M7 使用 1-5 标签体系替代此 0-3 体系。

| 标签 | 名称 | 定义 |
|------|------|------|
| **0** | 不匹配 | 明显不相关或方向不一致 |
| **1** | 弱匹配 | 有少量相关点,但缺少关键技能或方向偏差 |
| **2** | 中等匹配 | 方向一致,部分技能满足,存在一些技能差距 |
| **3** | 强匹配 | 方向高度一致,关键技能覆盖率高,技能差距少 |

**相关性阈值**:标签 ≥ 2(中等匹配或强匹配)被视为"相关职位"

### 评估指标

**Precision@K**(精确率):
- 定义:Top-K 推荐中相关职位的比例
- 公式:`Precision@K = (Top-K 中相关职位数) / K`
- 值域:0.0 - 1.0,越高越好

**NDCG@K**(归一化折损累积增益):
- 定义:考虑排序位置的质量评分
- 公式:`NDCG@K = DCG@K / IDCG@K`
- 值域:0.0 - 1.0,越高越好
- 特点:排在前面的职位权重更高

### 如何运行评估（M5 旧版，已弃用）

> ⚠️ **已弃用**：以下 M5 评估流程已被 M7 的 LOOCV + Ablation Study 替代。请参考 **M7: Learning to Rank (LTR) 完整 Pipeline** 部分了解新的评估方法。

#### 1. 生成 LLM 标签（已弃用，M7 使用 generate_labels.py 生成全量 1-5 标签）

```bash
cd backend/eval
python generate_labels.py
```

~~这将生成:
- `labels_suggested.jsonl` - LLM 生成的标签(JSONL 格式)~~（**M7 已覆盖为 750 对 1-5 标签**）
- ~~`labels_final.csv` - 人工校正模板(CSV 格式)~~（**M7 已弃用**）

#### 2. 人工校正（已弃用，M7 不再需要）

~~打开 `backend/eval/labels_final.csv`,在 `final_label` 列填入校正后的标签~~（**M7 已弃用此文件**）

#### 3. 运行评估（已弃用，M7 使用 scripts/eval_ablation.py）

```bash
cd backend/eval
python run_eval.py  # 已弃用
```

~~评估结果将保存到:
- `eval_results.json` - 详细结果(JSON 格式)~~（**M7 已弃用，改用 results/ablation_results.json**）
- ~~控制台输出汇总指标~~

#### 4. 查看评估报告（M7 仍然保留，但内容已更新）

```bash
cat backend/eval/eval_report.md
```

~~报告包含~~（**M7 已更新报告内容**）:
- 数据规模与分布（M7: 750 对 vs M5: 105 对）
- 标签体系说明（M7: 1-5 scale vs M5: 0-3 scale）
- 评估指标定义（相同）
- 结果解读指南（M7: LOOCV + Ablation vs M5: 简单验证）
- Weak Labels 说明与改进建议

### 评估数据规模

当前评估基于:
- **M5 旧版(已弃用)**:7 份简历 × Top-15 职位 = 105 个标注对(0-3 scale)
- **M7 新版(当前)**:15 份简历 × 50 个职位 = **750 个标注对**(1-5 scale)
- **标签来源**:LLM(GPT-4o-mini)独立生成(无信息泄漏)
- **覆盖率**:全量覆盖(所有 resume×job 组合)

### 评估公正性保证

**防止评估偏置(Label Leakage Prevention)**:

为避免评估偏置,LLM 标注阶段不暴露任何系统排序或打分信息,所有标签均基于原始 JD 与 Resume 独立生成。

具体措施:
- ✅ LLM 仅接收原始简历和职位描述文本
- ✅ 不提供系统计算的 matched_skills、gap_skills、final_score
- ✅ LLM 被明确告知其角色是"独立的人工评估者"
- ✅ 确保标签反映真实判断,而非系统输出的复述

### Weak Labels 说明

**什么是 Weak Labels?**
- LLM 自动生成的标签,非人工标注的金标准
- 优势:快速、低成本、可扩展
- 局限:准确性不如人工,建议抽查并修正

**推荐流程:**
1. LLM 快速生成 suggested_label(已完成)
2. 人工抽查 20-30% 并修正 final_label
3. 重新运行评估获得更准确的结果

### 数据 ID 说明

**为什么添加 job_id 和 resume_id?**
- 仅用于评估对齐,不影响推荐逻辑
- job_id: job_001, job_002, ..., job_022
- resume_id: resume_001, resume_002, ..., resume_007
- 在 `/recommend_jobs` 接口返回的 JobRecommendation 中包含 job_id

## M6:一键运行 Demo(Streamlit 交互界面)

### 功能概述

M6 提供了一个基于 Streamlit 的交互式 Web 界面,让您无需手动编写代码即可体验完整的职位匹配功能:
- 📄 多种简历输入方式(文本框输入或上传 TXT 文件)
- 💼 职位选择(从 jobs.jsonl 数据库选择)
- 🎯 Top-K 参数配置(推荐职位数量)
- 🚀 一键匹配并展示结果(包括匹配分数、匹配技能、技能差距)
- 💡 详细解释(点击按钮查看 RAG 生成的匹配解释、差距分析、提升建议)

### 一键运行步骤

#### 前置条件

确保已完成环境配置和依赖安装(参考上文"如何运行"部分)。

#### 安装 Streamlit

```bash
# 方式一:使用 requirements.txt(推荐)
pip install -r frontend/requirements.txt

# 方式二:手动安装
pip install streamlit requests
```

#### 启动后端服务

在**第一个终端**中启动 FastAPI 后端:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后,API 将运行在 http://localhost:8000

#### 启动前端界面

在**第二个终端**中启动 Streamlit 前端:

```bash
# 确保在项目根目录
streamlit run frontend/streamlit_app.py
```

前端启动后,会自动打开浏览器,访问地址:http://localhost:8501

**如果浏览器没有自动打开**,请手动访问 http://localhost:8501

### 使用指南

#### 1. 输入简历

**方式一:手动输入**
- 选择"Manual Text Input"
- 在文本框中输入简历内容
- 建议按照以下格式组织(系统会自动解析):
  ```
  Education
  Bachelor of Science in Computer Science, MIT, 2020

  Projects
  Built a recommendation system using collaborative filtering and deep learning

  Skills
  Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, NLP

  Experience
  Software Engineer at Tech Corp (2020-2023)
  - Developed ML models for user personalization
  - Improved recommendation accuracy by 25%
  ```

**方式二:上传文件**
- 选择"Upload TXT File"
- 点击"Browse files"上传 TXT 格式的简历文件

#### 2. 选择职位(可选)

- 从下拉列表中选择职位
  - 列表显示格式:`job_id: 职位名称`
  - 选择"-- None (match all jobs) --"表示匹配所有职位
  - 点击"View Job Details"可查看职位详情

#### 3. 设置匹配参数

- 使用滑块调整 **Top-K**(推荐职位数量)
- 范围:1-20,默认值:5

#### 4. 运行匹配

- 点击 **"🚀 Run Match"** 按钮
- 系统将:
  1. 解析简历内容
  2. 调用后端 `/recommend_jobs` 接口
  3. 展示 Top-K 匹配职位

#### 5. 查看结果

匹配结果将显示每个职位的:
- **职位信息**:标题、公司、地点、级别
- **匹配技能**:简历与职位要求的技能交集
- **技能差距**:职位要求但简历缺失的技能

#### 6. 查看详细解释

- 点击任意职位下的 **"💡 Explain Match"** 按钮
- 系统将调用 `/explain` 接口生成详细解释
- 展开的解释包含:
  - **Why this job matches**:基于证据的匹配原因
  - **Gap Analysis**:详细的技能差距分析
  - **Improvement Suggestions**:可行的提升建议

### 界面功能说明

#### 侧边栏

- **About**:系统简介和使用说明
- **Backend Status**:实时检查后端服务状态
  - 绿色:后端正常运行
  - 红色:后端未启动(请先启动后端服务)

#### 主界面布局

- **左侧列**:简历输入区域
- **右侧列**:职位选择区域(可选)
- **底部**:匹配参数和运行按钮
- **结果区**:Top-K 职位卡片(按匹配分数排序)

### 示例数据

您可以使用以下示例数据快速测试:

**示例简历(NLP 方向)**:
```
Education
Master of Science in Natural Language Processing, Carnegie Mellon University, 2019-2021

Projects
Built conversational AI system using GPT-4 and RAG, serving 500K+ users
Developed multilingual NER system supporting 15 languages using BERT

Skills
NLP, LLM, Transformers, BERT, GPT, Claude, Prompt Engineering, RAG, Fine-tuning, Python, PyTorch, FastAPI

Experience
NLP Engineer at AI Startup (2021-2024): Built LLM-powered products, implemented RAG systems, fine-tuned models for domain adaptation
```

然后:
1. 设置 Top-K = 5
2. 点击"Run Match"
3. 查看推荐的 NLP 相关职位(如"NLP Engineer - Conversational AI"、"LLM Engineer"等)
4. 点击"Explain Match"查看详细匹配解释

### 技术栈

- **前端框架**:Streamlit(轻量级 Python Web 框架)
- **HTTP 客户端**:requests
- **后端 API**:FastAPI(详见 M1-M5)

### 故障排除

**问题:点击"Run Match"后提示"Backend is not running"**
- 解决:确保后端服务已启动(`uvicorn main:app --reload`)
- 检查后端是否运行在 http://localhost:8000
- 查看侧边栏"Backend Status"状态

**问题:解释生成失败**
- 原因:可能是 OpenAI API Key 未配置或 RAG 服务异常
- 解决:检查 `.env` 文件中的 `OPENAI_API_KEY` 配置(参考 M4 配置说明)
- 说明:即使 RAG 失败,匹配功能仍可正常使用

**问题:简历解析不准确**
- 解决:建议在简历中明确使用"Education"、"Projects"、"Skills"、"Experience"等节标题
- 技能建议使用逗号分隔(如"Python, Machine Learning, NLP")

**问题:找不到 jobs.jsonl 文件**
- 解决:确保 `backend/data/jobs.jsonl` 文件存在
- 检查 Streamlit 是否从项目根目录运行(`streamlit run frontend/streamlit_app.py`)

## M7:Learning to Rank (LTR) 完整 Pipeline

### 概述

M7 引入了完整的 Learning to Rank(LTR)系统,相比 M3 的启发式排序(heuristic),LTR 通过**学习**来优化排序效果。

**核心改进:**
1. **全量 Weak Labels(1-5 scale)**:覆盖所有 resume×job 组合(15×50=750 pairs),替代旧版只标注 top-15 的 0-3 标签
2. **Pairwise Learning to Rank**:使用 Logistic Regression 学习排序,而非固定权重
3. **LOOCV + Ablation**:严格的小数据评估,对比三种排序方法(embedding_only / heuristic / LTR)
4. **前端一键切换**:Streamlit UI 支持开启/关闭 LTR,实时对比效果

### 三步完整流程

#### 步骤 1:生成全量 1-5 Weak Labels

```bash
# 设置环境变量(需要 OpenAI API Key)
export OPENAI_API_KEY=sk-your-actual-api-key-here

# 生成标签(覆盖所有 resume×job 组合)
cd backend/eval
python generate_labels.py
```

**功能说明:**
- 遍历所有 15×50=750 个 resume-job 组合
- LLM 独立打分(1-5),**不泄露**系统排序信息
- 校验覆盖率(缺失配对会报错)

**输出文件:**
- `backend/eval/labels_suggested.jsonl` - 全量 750 对标签(1-5 scale)

**标签定义(1-5 scale):**

| 标签 | 名称 | 定义 |
|------|------|------|
| **1** | Not a match | 明显不相关或方向不一致 |
| **2** | Weak match | 有少量相关点,但缺少关键技能 |
| **3** | Partial match | 方向一致,部分技能满足,有差距 |
| **4** | Good match | 方向对齐好,技能覆盖率高,轻微差距 |
| **5** | Strong match | 高度匹配,技能覆盖优秀,差距极少 |

**覆盖率校验:**
脚本会自动验证是否覆盖所有配对:
```
✅ Coverage validation PASSED: All 750 pairs are labeled!
```
如有遗漏,会打印缺失的 (resume_id, job_id) 并报错。

---

#### 步骤 2:运行 LOOCV + Ablation 评估

```bash
# 运行评估(训练 LTR 模型 + 计算指标)
cd backend
python scripts/eval_ablation.py
```

**评估方法:**
- **LOOCV(Leave-One-Out Cross-Validation)**:
  - 每次留 1 个 resume 做测试,其余 14 个做训练
  - 共 15 折,确保每个 resume 都被测试
  - 适合小数据集(15 resumes),避免过拟合
- **测试集评估范围**:
  - 对测试 resume 的**所有 50 个 jobs** 进行排序评估
  - **不是只评估 top-15**(避免偏置)

## Ablation 对比方法

| 方法 | 说明 |
| --- | --- |
| embedding_only | 仅使用语义相似度排序（M2 baseline） |
| heuristic | M3 启发式加权（embedding + skill_overlap + keyword_bonus - gap_penalty） |
| ltr_logreg | M7 Pairwise Logistic Regression（2个特征: embedding + keyword_bonus） |

**评估指标:**
- **NDCG@5 / NDCG@10**:排序质量(考虑位置权重,0-1 越高越好)
- **Precision@5 / Precision@10**:相关职位比例(阈值:label ≥ 4,0-1 越高越好)

**输出文件:**
- `backend/results/ablation_results.json` - 详细结果(per-fold + aggregated)
- `backend/eval/eval_report.md` - 可读性评估报告
- 终端输出汇总表格

**示例输出:**
```
================================================================
Summary
================================================================

embedding_only:
  ndcg@5          0.723 ± 0.045
  ndcg@10         0.801 ± 0.032
  precision@5     0.657 ± 0.089
  precision@10    0.571 ± 0.067

heuristic:
  ndcg@5          0.756 ± 0.041
  ndcg@10         0.825 ± 0.029
  precision@5     0.714 ± 0.082
  precision@10    0.600 ± 0.061

ltr_logreg:
  ndcg@5          0.782 ± 0.038
  ndcg@10         0.845 ± 0.026
  precision@5     0.743 ± 0.075
  precision@10    0.629 ± 0.058
```

**模型保存:**
评估过程中,每个 fold 会训练一个 LTR 模型。若需在生产环境使用,需单独用**全部数据**训练最终模型:
```bash
# 训练最终模型(全量数据)
cd backend
python scripts/train_ltr_model.py `
  --resumes_path data/resumes.jsonl `
  --jds_path data/jobs.jsonl `
  --labels_path eval/labels_suggested.jsonl `
  --min_rel_diff 2 `
  --random_state 42

# 默认输出: models/ltr_logreg.joblib
```

**输出示例:**
```
================================================================================
LTR Model Training for Production
================================================================================

[1/6] Loading data...
  Loaded: 15 resumes, 50 jobs, 750 labels

[2/6] Validating data...
  ✅ Full coverage: 750/750 pairs labeled

[3/6] Building feature cache...
  [OK] Cached 750 embedding scores
  [OK] Built 750 feature vectors
  Feature dimension: 2
  Feature names: ['embedding', 'keyword_bonus']

[4/6] Constructing pairwise training data...
  [OK] Created 5700 pairwise training samples

[5/6] Training LTR model...
  [OK] Model trained successfully

  Learned feature weights:
    embedding            +3.4061
    keyword_bonus        +2.2702

[6/6] Saving model...
  [OK] Model saved to: models/ltr_logreg.joblib

Training Complete!
```

---

#### 步骤 3:在 Demo 中启用 LTR

**后端 API 支持:**

`/recommend_jobs` 接口新增参数:
```json
{
  "resume": { ... },
  "top_k": 5,
  "use_ltr": true  // 新增:启用 LTR 排序
}
```

**响应新增字段:**
```json
{
  "recommendations": [ ... ],
  "total_jobs_searched": 50,
  "ranker": "ltr_logreg"  // 新增:使用的排序器
}
```

**ranker 字段可能的值:**
- `"heuristic"` - 使用 M3 启发式排序(默认,use_ltr=false)
- `"ltr_logreg"` - 使用 LTR 模型排序(use_ltr=true 且模型存在)
- `"heuristic_fallback"` - LTR 失败回退到启发式(模型不存在或加载失败)

**Streamlit 前端使用:**

1. 启动后端:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. 启动前端:
```bash
streamlit run frontend/streamlit_app.py
```

3. 在 UI 上勾选 **"Enable LTR re-ranking (use_ltr)"** 复选框
4. 点击 **"Run Match"** 运行匹配
5. 查看结果顶部的 ranker 标识(🤖 LTR 或 🔧 Heuristic)

**效果对比:**
- 不勾选:使用 M3 启发式排序(固定权重)
- 勾选:使用 LTR 学习的排序(如果模型存在)

---

### 关键设计约束

**防止标签泄漏(Label Leakage Prevention):**
- ✅ LLM 生成标签时**不接收**任何系统排序信息(matched_skills、gap_skills、scores、topK)
- ✅ LLM 仅基于原始 resume 和 job 文本打分
- ✅ Prompt 明确告知 LLM 其角色是"独立评估者"

**LTR 特征（多重共线性感知）:**
- ✅ LTR 使用 2 个特征: **embedding** 和 **keyword_bonus**（避免多重共线性）
- ✅ 移除的特征: skill_overlap 和 gap_penalty（相关性 r>0.95，导致权重学习不稳定）
- ✅ L2 正则化（C=0.1）稳定训练，尽管仍存在相关性（r=0.89）
- ✅ 与 M3 的区别: M3 使用固定权重（全部 4 个特征），LTR 从数据学习权重（2 个特征）

**Pairwise 训练与 Mirrored Pairs:**
- 默认 `min_rel_diff=2`:只有当 `label_i ≥ label_j + 2` 时才构造训练对
- 例如:(label=5, label=3) → 构造训练对;(label=4, label=3) → 不构造
- 如果某个 resume 的 labels 方差太小(所有 jobs 标签都接近),可能无法构造足够的 pairs

**为什么需要 Mirrored Pairs(镜像对)?**

Pairwise LTR 使用 Logistic Regression 进行二分类:
- `y=1` 表示"第一个职位优于第二个职位"
- `y=0` 表示"第一个职位不优于第二个职位"

**关键约束**:sklearn 的 LogisticRegression **要求训练数据至少包含 2 个类别**。如果 `y_pairs` 只包含一个类别(全是 1),训练会失败。

**解决方案**:对每个正向配对生成镜像负样本:
```
原始 pair:   (winner - loser, y=1)  # 表示 winner 优于 loser
镜像 pair:   (loser - winner, y=0)  # 表示 loser 不优于 winner
```

由于 `loser - winner = -(winner - loser)`,镜像 pair 使用相反的特征差向量,确保模型学习到对称的排序关系。

**实现细节:**
- `construct_pairwise_data()` 函数的 `add_mirror` 参数**默认为 True**
- 训练脚本会自动检查 `y_pairs` 的类别数:
  - 如果只有 1 个类别 → 自动用 `add_mirror=True` 重新构造
  - 如果仍然失败 → 报错退出
- 这样确保 LogisticRegression 总能接收到有效的训练数据

**为什么默认启用 add_mirror?**
- 保证训练稳定性(避免单类别错误)
- 增加训练样本数量(约 2x)
- 提供更平衡的类别分布(通常接近 50%-50%)
- 对小数据集尤其重要(如本项目的 15 resumes)

**回退机制:**
- 如果某个 fold 的 pairwise pairs < 10,LTR 训练会失败,自动回退到 heuristic
- 如果 FastAPI 找不到 `models/ltr_logreg.joblib`,自动回退到 heuristic,ranker 返回 `"heuristic_fallback"`

---

### 文件说明

**新增/修改文件列表:**

| 文件路径 | 说明 | 类型 |
|----------|------|------|
| `backend/eval/generate_labels.py` | 全量 1-5 weak labels 生成(覆盖旧版 0-3 top-15) | 修改 |
| `backend/src/ranking/features.py` | 特征提取与向量化(FEATURE_NAMES 固定顺序) | 新增 |
| `backend/src/ranking/pairwise.py` | Pairwise 训练数据构造(含 mirror pairs 支持) | 新增 |
| `backend/src/ranking/ltr_logreg.py` | Pairwise Logistic Regression 模型(含 save/load) | 新增 |
| `backend/scripts/eval_ablation.py` | LOOCV + Ablation 评估脚本 | 新增 |
| `backend/scripts/train_ltr_model.py` | 生产环境 LTR 模型训练脚本(含自动 mirror pairs 回退) | 新增 |
| `backend/main.py` | FastAPI:新增 use_ltr 参数、ranker 字段 | 修改 |
| `frontend/streamlit_app.py` | Streamlit:新增 LTR 切换 checkbox、ranker 显示 | 修改 |
| `backend/data/resumes.jsonl` | 扩展到 15 条简历 | 修改 |
| `backend/data/jobs.jsonl` | 扩展到 50 条职位 | 修改 |
| `backend/eval/labels_suggested.jsonl` | 全量 750 对标签(1-5 scale) | 覆盖 |
| `backend/models/ltr_logreg.joblib` | 训练好的 LTR 模型 | 新增(需运行训练脚本) |
| `backend/results/ablation_results.json` | Ablation study 结果 | 新增 |

---

### 快速命令汇总

```bash
# ====== 步骤 1:生成全量 1-5 weak labels ======
export OPENAI_API_KEY=sk-your-key
cd backend/eval
python generate_labels.py

# ====== 步骤 2:运行 LOOCV + Ablation 评估 ======
cd backend
python scripts/eval_ablation.py

# ====== 步骤 3:训练最终 LTR 模型(用于生产) ======
cd backend
python scripts/train_ltr_model.py `
  --resumes_path data/resumes.jsonl `
  --jds_path data/jobs.jsonl `
  --labels_path eval/labels_suggested.jsonl `
  --min_rel_diff 2 `
  --random_state 42

# ====== 步骤 4:启动 Demo ======
# 终端 1:后端
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 终端 2:前端
streamlit run frontend/streamlit_app.py

# ====== 覆盖率校验(可选) ======
# 训练脚本会自动验证标签覆盖率,无需单独运行
# 查看验证结果:运行训练脚本即可看到 [2/6] Validating data 步骤的输出
```

---

### 常见问题

**Q1:为什么要覆盖旧的 labels_suggested.jsonl?**
- 旧版只标注 top-15(105 对:7×15),且使用 0-3 scale
- 新版覆盖全量(750 对:15×50),使用 1-5 scale
- 旧文件会自动备份,不会丢失

**Q2:LTR 模型保存在哪里?**
- 评估脚本(`scripts/eval_ablation.py`)在每个 fold 中训练模型,但不保存
- 需要单独训练全量模型并保存到 `models/ltr_logreg.joblib`(见步骤 2 的代码片段)
- 也可以修改评估脚本,在最后一个 fold 结束后保存模型

**Q3:FastAPI 如何使用 LTR 模型?**
- 如果 `use_ltr=true` 且 `models/ltr_logreg.joblib` 存在,加载模型并排序
- 如果模型不存在或加载失败,自动回退到 heuristic,ranker 返回 `"heuristic_fallback"`

**Q4:LOOCV 每个 fold 的训练数据是否足够?**
- 15 个 resumes,每个 fold 用 14 个训练
- 每个 resume 有 50 个 jobs,理论上可以构造很多 pairwise pairs
- 但如果某个 resume 的 labels 方差太小,pairs 可能不足,会回退到 heuristic

**Q5:如何查看 LTR 学到的特征权重?**
```python
from src.ranking.ltr_logreg import PairwiseLTRModel
model = PairwiseLTRModel.load('models/ltr_logreg.joblib')
weights = model.get_feature_weights()
print(weights)
# 输出示例:{'embedding': 3.41, 'keyword_bonus': 2.27}
```

或使用提供的脚本:
```bash
cd backend
python view_ltr_weights.py
```

**Q6:如何添加新特征?**
1. 在 `src/ranking/features.py` 的 `FEATURE_NAMES` 列表末尾添加新特征名
2. 在 `build_features()` 函数中计算新特征值
3. 重新生成 labels 和训练模型(特征顺序变化会导致旧模型不兼容)

---

## 下一步计划

后续 Milestone 将实现:
- ✅ ~~基于向量嵌入的语义匹配~~(M2 已完成)
- ✅ ~~批量匹配和排序功能~~(M2 已完成)
- ✅ ~~可解释的轻量排序层~~(M3 已完成)
- ✅ ~~集成 LLM 进行更智能的匹配分析和个性化建议~~(M4 已完成)
- ✅ ~~评估体系与弱监督标签生成~~(M5 已完成)
- ✅ ~~Streamlit 交互界面 Demo~~(M6 已完成)
- ✅ ~~Learning to Rank 完整 Pipeline~~(M7 已完成)
- 更多推荐算法(混合推荐、协同过滤等)

## 许可证

MIT
