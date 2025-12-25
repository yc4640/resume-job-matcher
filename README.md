# LM Match Service

## 项目简介

LM Match Service 是一个基于 FastAPI 的求职简历匹配服务。本项目目前处于 M4 阶段，在可解释排序的基础上，增加了 RAG（检索增强生成）可解释层，为每个推荐职位生成基于证据的匹配分析。

### 当前功能

#### M1：基础匹配功能
- ✅ 健康检查接口 (`/health`)
- ✅ 职位-简历匹配接口 (`/match`) - 返回结构化匹配结果
- ✅ 使用 Pydantic 定义数据模型（JobPosting、Resume、MatchResponse）
- ✅ 基于技能集合的匹配算法（不使用 LLM）
- ✅ 提供匹配分数、匹配技能、技能差距和学习建议

#### M2：语义推荐功能
- ✅ 职位推荐接口 (`/recommend_jobs`) - 基于语义相似度的 Top-K 推荐
- ✅ 使用 sentence-transformers 本地模型进行文本嵌入
- ✅ 余弦相似度计算和排序
- ✅ 批量职位数据集（jobs.jsonl）和简历数据集（resumes.jsonl）
- ✅ 完全本地运行，无需付费 API

#### M3：可解释排序功能
- ✅ 轻量排序层 - 在 embedding 召回基础上引入多维度打分
- ✅ 技能词表 (180+ 技能) - 标准化技能匹配
- ✅ YAML 配置 - 无需修改代码即可调整排序权重
- ✅ 多维度特征：
  - `embedding_score`: 语义相似度
  - `skill_overlap`: 技能覆盖率
  - `keyword_bonus`: 关键字命中加分
  - `gap_penalty`: 缺失关键技能惩罚
- ✅ 可解释性 - 自动生成排名第一的详细解释

#### M4：RAG 可解释层
- ✅ 证据构建 - 从职位和简历中提取结构化证据
- ✅ 智能检索 - 基于语义相似度选择最相关的证据片段
- ✅ LLM 生成 - 使用大语言模型生成基于证据的解释
- ✅ 三维分析 - 为每个推荐职位提供：
  - `explanation`: 为什么这个岗位适合候选人
  - `gap_analysis`: 候选人缺少哪些关键技能或资质
  - `improvement_suggestions`: 具体可行的提升建议
- ✅ 防止幻觉 - 严格基于证据生成，LLM 仅用于解释层，不参与排序
- ✅ **技能自动提取与合并** - 从简历文本（education/projects/experience）中自动提取技能，避免过度严格的匹配
- ✅ **软技能过滤** - 软技能（如 Communication、Leadership）缺失不计入 gap_penalty

#### 通用特性
- ✅ RESTful API 设计
- ✅ 自动生成的 API 文档（Swagger UI / ReDoc）

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
│   │   └── utils.py            # 工具函数（技能提取与合并）(M4.1)
│   ├── config/              # 配置文件 (M3 新增)
│   │   └── ranking_config.yaml # 排序权重配置
│   └── data/
│       ├── sample_job.json        # 示例职位数据
│       ├── sample_resume.json     # 示例简历数据
│       ├── jobs.jsonl             # 批量职位数据（22条）
│       ├── resumes.jsonl          # 批量简历数据（7条）
│       └── skills_vocabulary.txt  # 技能词表（180+ 技能）(M3)
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

### 4. 配置环境变量（M4 新增）

为了使用 RAG 可解释层功能，需要配置 OpenAI API Key：

```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑 .env 文件，填入你的 OpenAI API Key
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

**获取 OpenAI API Key：**
1. 访问 https://platform.openai.com/api-keys
2. 登录或注册 OpenAI 账号
3. 创建新的 API Key
4. 将 API Key 填入 `.env` 文件

**注意：** 如果不配置 API Key，推荐接口仍可正常工作，但每个推荐职位的 `explanation`、`gap_analysis` 和 `improvement_suggestions` 字段将为 `null`。

### 5. 启动服务

```bash
# 方式一：使用 uvicorn 命令
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 方式二：直接运行 main.py
python main.py
```

服务启动后，访问 http://localhost:8000

### 6. 查看 API 文档

FastAPI 自动生成交互式 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 数据模型说明

### JobPosting（职位信息）

```json
{
  "title": "职位名称",
  "responsibilities": "岗位职责描述",
  "requirements_text": "任职要求描述",
  "skills": ["技能1", "技能2", "..."],
  "company": "公司名称（可选）",
  "location": "工作地点（可选）",
  "level": "职位级别（可选）"
}
```

### Resume（简历信息）

```json
{
  "education": "教育背景",
  "projects": "项目经历",
  "skills": ["技能1", "技能2", "..."],
  "experience": "工作经验"
}
```

### MatchResponse（匹配结果）

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

### 批量测试数据集（JSONL 格式）

为了支持后续的 top-k 推荐功能测试，我们提供了两个 JSON Lines 格式的数据集：

#### backend/data/jobs.jsonl
- 包含 22 条真实的职位信息
- 涵盖技能领域：推荐系统、搜索、NLP、LLM、数据工程、后端开发、机器学习等
- 每行一个 JSON 对象，符合 `JobPosting` schema

#### backend/data/resumes.jsonl
- 包含 7 条不同背景的简历
- 技能与职位数据有不同程度的重叠，适合测试匹配算法
- 每行一个 JSON 对象，符合 `Resume` schema

#### 如何加载 JSONL 文件

在 Python 中加载这些文件用于测试：

```python
import json
from schemas import JobPosting, Resume

# 加载所有职位
jobs = []
with open('backend/data/jobs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        job_data = json.loads(line)
        jobs.append(JobPosting(**job_data))

print(f"加载了 {len(jobs)} 个职位")

# 加载所有简历
resumes = []
with open('backend/data/resumes.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        resume_data = json.loads(line)
        resumes.append(Resume(**resume_data))

print(f"加载了 {len(resumes)} 份简历")
```

#### 用于 top-k 推荐测试示例

```python
# 示例：为一份简历找到最匹配的 top-5 职位
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# 加载第一份简历（推荐系统背景）
with open('backend/data/resumes.jsonl', 'r', encoding='utf-8') as f:
    resume_data = json.loads(f.readline())

# 加载所有职位并计算匹配分数
matches = []
with open('backend/data/jobs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        job_data = json.loads(line)

        # 调用 /match 接口
        response = client.post("/match", json={
            "job": job_data,
            "resume": resume_data
        })

        result = response.json()
        matches.append({
            "job_title": job_data["title"],
            "match_score": result["match_score"],
            "matched_skills": result["matched_skills"],
            "gaps": result["gaps"]
        })

# 按匹配分数排序，取 top-5
top_5 = sorted(matches, key=lambda x: x["match_score"], reverse=True)[:5]

print("\nTop 5 最匹配的职位：")
for i, match in enumerate(top_5, 1):
    print(f"{i}. {match['job_title']} - 匹配度: {match['match_score']}%")
    print(f"   匹配技能: {', '.join(match['matched_skills'])}")
    print(f"   技能差距: {', '.join(match['gaps'])}\n")
```

#### 预期使用场景

这些 JSONL 数据集将在后续 Milestone 中用于：
1. **批量匹配测试**：测试系统处理多个职位和简历的性能
2. **Top-k 推荐**：为给定简历推荐最匹配的 k 个职位（或反向推荐）
3. **排序算法验证**：验证基于匹配分数的排序逻辑
4. **性能基准测试**：测试大规模匹配的响应时间和准确性

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

#### 方式一：使用 Swagger UI（推荐）

1. 访问 http://localhost:8000/docs
2. 找到 `POST /match` 接口
3. 点击 **"Try it out"** 按钮
4. 在 Request body 中粘贴以下 JSON：

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

#### 方式二：使用 curl

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

#### 方式三：使用 Python 测试脚本

运行项目自带的测试脚本：

```bash
cd backend
python test_match.py
```

该脚本包含多个测试用例，涵盖完全匹配、部分匹配、不匹配等场景。

### 测试职位推荐接口（M2 新增）

#### 方式一：使用 Swagger UI（推荐）

1. 访问 http://localhost:8000/docs
2. 找到 `POST /recommend_jobs` 接口
3. 点击 **"Try it out"** 按钮
4. 在 Request body 中粘贴以下 JSON（使用示例简历数据）：

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
        "embedding_score": 0.682073712348938,
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
        "embedding_score": 0.6174665093421936,
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
        "embedding_score": 0.6600039005279541,
        "skill_overlap": 0.75,
        "keyword_bonus": 0.7,
        "gap_penalty": 0.2,
        "final_score": 0.6090015602111816
      }
    }
  ],
  "total_jobs_searched": 22,
  "explanation": "【NLP Engineer - Conversational AI】Ranked #1 for the following reasons:\n\n1. Semantic Similarity: 0.682 (Weight: 0.4)\n   - The job description is highly semantically aligned with the resume content\n\n2. Skill Coverage: 1.000 (Weight: 0.3)\n   - Matched skills (7): Python, Transformers, Prompt Engineering, NLP, LLM\n   - Missing skills (0): None\n\n3. Keyword Bonus: 0.850 (Weight: 0.2)\n   - Matches high-priority skills\n\n4. Gap Penalty: 0.000 (Weight: 0.1)\n   - Penalty applied for missing critical skills\n\nOverall Score: 0.743"
}
```

**说明（M4 更新）：**
- `similarity_score`：基于语义嵌入的余弦相似度（0-1之间，等同于 embedding_score）
- `matched_skills`：简历技能与职位要求技能的交集（基于标准化技能词表）
- `gap_skills`：职位要求但简历缺失的技能（M3 新增）
- `features`：可解释的排序特征（M3 新增）
  - `embedding_score`：语义相似度（0-1）
  - `skill_overlap`：技能覆盖率（0-1）
  - `keyword_bonus`：关键词加分（0-1）
  - `gap_penalty`：缺失惩罚（0-1）
  - `final_score`：综合得分（加权计算）
- `explanation`：排名第一职位的详细解释（M3 新增）
- **M4 新增字段（每个推荐职位）：**
  - `explanation`：为什么这个岗位适合候选人（基于证据的解释）
  - `gap_analysis`：候选人缺少哪些关键技能或资质
  - `improvement_suggestions`：具体可行的提升建议
- `total_jobs_searched`：从 jobs.jsonl 加载的总职位数量

**M4 返回示例（单个推荐职位）：**
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
    "embedding_score": 0.682,
    "skill_overlap": 1.0,
    "keyword_bonus": 0.85,
    "gap_penalty": 0.0,
    "final_score": 0.743
  },
  "explanation": "This position is an excellent fit for you because your experience building conversational AI systems with GPT-4 and RAG directly aligns with the job's core requirements. Your projects demonstrate practical expertise in NLP and LLM applications, particularly in handling large-scale user interactions (500K+ users).",
  "gap_analysis": "While you have strong NLP fundamentals, the position requires experience with dialogue systems and intent recognition frameworks which are not explicitly mentioned in your resume. Additionally, production-scale deployment experience with specific chatbot frameworks would strengthen your candidacy.",
  "improvement_suggestions": "- Build a dialogue management system using Rasa or similar frameworks to demonstrate intent recognition capabilities\n- Complete a project focusing on multi-turn conversation handling and context management\n- Document your experience with A/B testing and performance optimization in production chatbot environments"
}
```

#### 方式二：使用 curl

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

#### 推荐接口特点（M3 增强）

- **语义匹配 (M2)**：使用 sentence-transformers 本地模型（all-MiniLM-L6-v2）进行文本嵌入
- **多维度排序 (M3)**：结合语义相似度、技能覆盖率、关键词加分、缺失惩罚的综合打分
- **可解释性 (M3)**：自动生成排名第一职位的详细解释，说明为什么它最匹配
- **灵活配置 (M3)**：通过 YAML 配置文件调整排序权重，无需修改代码
- **标准化技能 (M3)**：基于 180+ 技能词表进行标准化匹配
- **无需付费 API**：完全本地运行，无需调用外部 API
- **技能重叠信息**：提供精确的匹配技能和缺失技能列表

## 技术栈

- **FastAPI**: 现代、高性能的 Python Web 框架
- **Pydantic**: 数据验证和设置管理
- **Uvicorn**: ASGI 服务器
- **Sentence-Transformers**: 本地文本嵌入模型（M2）
- **NumPy**: 向量计算和相似度计算（M2）
- **PyYAML**: 配置文件管理（M3）
- **OpenAI API**: LLM 生成解释文本（M4）

## 匹配算法说明

### M1：基于技能集合的精确匹配

使用集合运算进行技能匹配：

1. **匹配技能** (matched_skills)：求职者技能与职位要求技能的交集
2. **技能差距** (gaps)：职位要求技能中求职者不具备的技能
3. **匹配分数** (match_score)：匹配技能数量占职位要求技能总数的百分比
   - 公式：`match_score = (len(matched_skills) / len(job.skills)) * 100`
   - 如果职位没有技能要求，则返回 0
4. **学习建议** (suggestions)：针对每个技能差距提供学习建议

### M2：基于语义嵌入的推荐系统

使用 sentence-transformers 进行语义相似度匹配：

1. **文本嵌入**：
   - 模型：all-MiniLM-L6-v2（384维向量，本地运行）
   - 职位文本：拼接 title + responsibilities + requirements_text + skills
   - 简历文本：拼接 education + projects + experience + skills

2. **相似度计算**：
   - 使用余弦相似度（Cosine Similarity）计算简历与职位的语义相似度
   - 相似度范围：0-1，越接近1表示越相似

3. **Top-K 推荐**：
   - 根据相似度分数降序排序
   - 返回最匹配的 top-k 个职位
   - 附带精确的技能重叠信息（复用 M1 逻辑）

### M3：可解释的轻量排序层

在 M2 embedding 召回基础上，引入多维度打分机制：

#### 1. 排序特征

- **embedding_score (语义相似度)**：
  - 来自 M2 的文本嵌入余弦相似度
  - 范围：0-1

- **skill_overlap (技能覆盖率)**：
  - 基于标准化技能词表（180+ 技能）的匹配率
  - 公式：`matched_skills / job_required_skills`
  - 范围：0-1

- **keyword_bonus (关键词加分)**：
  - 高优先级技能匹配加分（如 Python、Machine Learning、LLM 等）
  - 高优先级技能权重 1.5x
  - 归一化到 0-1 范围

- **gap_penalty (缺失惩罚)**：
  - 缺失关键技能的惩罚（如 Python、SQL 等核心技能）
  - 关键技能缺失权重 2.0x
  - 归一化到 0-1 范围

#### 2. 打分公式

```
final_score = w1 * embedding_score
            + w2 * skill_overlap
            + w3 * keyword_bonus
            - w4 * gap_penalty
```

默认权重配置（可通过 YAML 调整）：
- `w1 (embedding)`: 0.4
- `w2 (skill_overlap)`: 0.3
- `w3 (keyword_bonus)`: 0.2
- `w4 (gap_penalty)`: 0.1

#### 3. 配置文件

排序权重通过 `config/ranking_config.yaml` 配置，支持：
- 调整各特征权重
- 定义高优先级关键词列表
- 定义关键技能列表
- 调整奖惩倍数
- **无需修改代码即可调整排序策略**

#### 4. 可解释性

系统自动生成排名第一职位的详细解释，包括：
- 各维度特征分数
- 匹配技能列表
- 缺失技能列表
- 综合得分计算过程

示例解释输出：
```
【NLP Engineer - Conversational AI】排名第一的原因：

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

### M4.1：技能自动提取与合并（Skills Auto-Extract & Merge）

#### 问题背景

在传统的技能匹配中，系统仅依赖用户在 `resume.skills` 列表中明确列出的技能。这会导致以下问题：

1. **过度严格的匹配**：很多技能实际上在简历的 `experience`、`projects` 或 `education` 中提到，但未在 `skills` 列表中列出
2. **误判技能缺口**：例如简历中提到 "conducted NER research" 或 "published papers on entity extraction"，但因为 `skills` 列表没写 "NER" 或 "Entity Extraction"，就被判定为缺失技能

#### 解决方案

系统自动从简历文本中提取技能，并与用户提供的技能列表合并：

**核心逻辑：**
```
merged_skills = union(
    user_provided_resume.skills,
    extracted_skills_from_resume_text
)
```

**提取流程：**
1. **文本组装**：将 `resume.education`、`resume.projects`、`resume.experience` 组合成一段文本
2. **词汇匹配**：基于 `skills_vocabulary.txt`（包含 180+ 技能词）进行匹配
3. **智能边界检测**：使用正则表达式的词边界（`\b`），避免误匹配（例如 "C" 不会匹配 "Cloud", "React" 不会匹配 "Reactivity"）
4. **特殊字符处理**：正确处理 "C++"、"C#"、".NET" 等包含特殊字符的技能
5. **大小写规范化**：匹配时忽略大小写，但保留词汇表中的原始大小写
6. **去重合并**：将提取的技能与用户提供的技能合并，去重后返回

**示例：**
```python
# 用户提供的技能
resume.skills = ["Python", "Machine Learning"]

# 简历文本中提到的内容
resume.projects = "Conducted research on NER and entity extraction..."
resume.experience = "Published papers on Named Entity Recognition..."

# 自动提取的技能
extracted_skills = ["NER", "Entity Extraction", "Research", "Publication"]

# 最终合并后的技能（用于匹配）
merged_skills = ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication"]
```

#### 软技能过滤

为了避免对候选人过度惩罚，系统在计算 `gap_penalty` 时会**过滤掉软技能**：

**软技能列表**（不计入缺失惩罚）：
- Communication（沟通）
- Leadership（领导力）
- Collaboration（协作）
- Teamwork（团队合作）
- Problem Solving（问题解决）
- Critical Thinking（批判性思维）
- Time Management（时间管理）
- Adaptability（适应性）
- 等等...

**为什么过滤软技能？**
- 软技能很重要，但缺失不应该像技术技能那样被严重扣分
- 软技能难以在简历中量化，容易被遗漏
- 软技能更多是在面试中评估，而非简历筛选阶段的硬性要求

**注意：** 软技能仍然会：
- ✅ 出现在 `matched_skills` 中（如果匹配）
- ✅ 出现在 `gap_skills` 中（如果缺失）
- ✅ 可用于 `keyword_bonus` 加分
- ✅ 出现在 RAG 解释的 evidence 中
- ❌ **不会**计入 `gap_penalty` 扣分

#### 实现位置

**新增文件：** `backend/services/utils.py`
- `extract_skills_from_text(text, vocab)` - 从文本中提取技能
- `merge_resume_skills(resume, vocab)` - 合并用户技能与提取技能
- `filter_soft_skills(skills)` - 过滤软技能
- `SOFT_SKILLS` - 软技能常量集合

**调用位置：** `backend/services/ranking.py` 的 `rank_jobs_with_features` 函数
```python
# === SKILLS AUTO-EXTRACT & MERGE ===
# Line 247-255
vocab_list = list(vocab)
merged_skills = merge_resume_skills(resume, vocab_list)
resume_skills_normalized = normalize_skills(merged_skills, vocab)
```

**使用位置：**
- ✅ `matched_skills` 计算 - 使用 merged skills
- ✅ `gap_skills` 计算 - 使用 merged skills
- ✅ `skill_overlap` 计算 - 使用 merged skills
- ✅ `keyword_bonus` 计算 - 使用 merged skills
- ✅ `gap_penalty` 计算 - 使用 merged skills（过滤软技能后）

#### 验收示例

**场景：** 简历中提到了 NER 研究，但未在 skills 列表中列出

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

**旧行为（问题）：**
- `matched_skills`: ["Python", "Machine Learning"]
- `gap_skills`: ["NER", "Entity Extraction", "Research", "Publication"]  ❌ 误判为缺失

**新行为（修复）：**
- `merged_skills`: ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication", "Literature Review"]
- `matched_skills`: ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication"]
- `gap_skills`: []  ✅ 正确识别

### M4：RAG 可解释层架构

#### RAG 在系统中的位置

RAG（Retrieval-Augmented Generation）层是 **纯解释层**，位于排序之后，**不参与职位排序逻辑**。整个推荐流程如下：

```
1. [M2 语义检索] 使用 embedding 计算所有职位与简历的相似度
           ↓
2. [M3 可解释排序] 基于多维度特征（embedding + skill + keyword + gap）计算最终得分并排序
           ↓
3. [M3 Top-K 选择] 选出排名前 K 的职位（排序已确定，不再改变）
           ↓
4. [M4 RAG 解释层] 对每个 Top-K 职位生成基于证据的解释
   ├─ 证据构建：提取职位和简历的结构化证据
   ├─ 智能检索：选择最相关的证据片段
   └─ LLM 生成：基于证据生成 explanation / gap_analysis / improvement_suggestions
           ↓
5. [返回结果] 包含排序、特征、RAG 解释的完整推荐结果
```

**关键约束：**
- M4 的 RAG 层 **仅用于生成解释文本**
- **不改变** M3 的 `final_score` 和排序顺序
- LLM 输出必须基于证据，禁止幻觉

#### RAG 的检索对象

RAG 检索的对象是 **职位和简历的文本片段（chunks）**，具体包括：

**职位证据（Job Evidence）：**
- `title`：职位名称
- `responsibilities`：岗位职责
- `requirements_text`：任职要求
- `skills`：要求技能列表

**简历证据（Resume Evidence）：**
- `education`：教育背景
- `projects`：项目经历
- `experience`：工作经验
- `skills`：技能列表

**检索流程：**
1. **文本分块（Chunking）**：将职位描述和简历内容按句子切分成小片段（约 200 字符）
2. **语义嵌入**：使用 sentence-transformers 模型对所有 chunks 计算向量表示
3. **相似度计算**：计算职位 chunks 与简历 chunks 之间的交叉相似度
4. **Top-K 选择**：选出最相关的 3 个职位 chunks 和 3 个简历 chunks 作为证据

**示例：**
- 职位 chunk: `[responsibilities] Design and implement scalable NLP systems for production chatbots.`
- 简历 chunk: `[projects] Built chatbot using GPT-4 and RAG, serving 500K+ users with 90% satisfaction rate.`
- 这两个 chunks 语义相似度高，会被选为证据传递给 LLM

#### LLM 在系统中的角色

LLM（大语言模型）**仅承担"解释生成"角色**，不参与任何排序或推荐决策：

**LLM 的职责：**
1. **阅读证据**：接收检索出的最相关职位和简历片段
2. **生成解释**：基于证据回答"为什么这个职位适合候选人"
3. **分析差距**：基于证据指出候选人缺少的关键技能
4. **提供建议**：给出具体可行的提升建议

**LLM 不做的事：**
- ❌ 不计算匹配分数（由 M3 ranking 层完成）
- ❌ 不决定职位排序（由 M3 final_score 决定）
- ❌ 不检索职位（由 M2 embedding 完成）
- ❌ 不评估技能匹配（由 M3 skill_overlap 完成）

**使用的 LLM 模型：**
- 默认：`gpt-4o-mini`（OpenAI）
- 优势：成本低、速度快、适合生成简短解释
- 温度设置：0.3（低温度保证输出稳定、事实性强）

#### 如何避免 LLM 编造内容

为了防止 LLM 幻觉（hallucination），我们采取了多层防护措施：

**1. 证据约束（Evidence Grounding）**
- LLM 只能看到通过检索选出的证据片段
- Prompt 明确要求："Based ONLY on the evidence provided below"
- 禁止 LLM 添加未在证据中出现的信息

**2. 结构化 Prompt**
- 提供清晰的职位证据和简历证据
- 明确列出 `matched_skills` 和 `gap_skills`（由 M3 计算得出）
- 要求 LLM 引用具体证据内容

**3. 低温度生成**
- 设置 `temperature=0.3`（默认是 1.0）
- 低温度使输出更确定性、更贴近事实
- 减少创造性发挥，增强事实准确性

**4. 格式化输出**
- 要求 LLM 按照固定格式输出（EXPLANATION / GAP_ANALYSIS / IMPROVEMENT_SUGGESTIONS）
- 自动解析和验证输出格式
- 失败时回退到基于规则的简单解释

**5. 检索质量保证**
- 使用与 M2 相同的 sentence-transformers 模型进行检索
- 基于余弦相似度选择最相关的证据
- 确保传递给 LLM 的证据与职位-简历匹配度高

**Prompt 示例片段：**
```
CRITICAL RULES:
- Base your analysis ONLY on the evidence provided above
- Reference specific details from the job and resume evidence
- Do not make assumptions or add information not present in the evidence
- Keep each section concise and focused
```

**后备机制：**
如果 LLM API 调用失败（网络问题、API key 未设置等），系统会回退到基于规则的简单解释：
```python
{
    "explanation": "This position matches 4 of your skills: Python, NLP, LLM, Transformers. The overall compatibility score is 0.68.",
    "gap_analysis": "You may need to develop these skills: Dialogue Systems, Intent Recognition.",
    "improvement_suggestions": "- Review the job requirements carefully\n- Consider online courses for missing skills"
}
```

## M3 配置说明

### 排序权重配置

编辑 `backend/config/ranking_config.yaml` 调整排序策略：

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

`backend/data/skills_vocabulary.txt` 包含 180+ 标准化技能，涵盖：
- 编程语言（Python, Java, JavaScript, ...）
- Web 框架（FastAPI, Django, React, ...）
- ML/AI（Machine Learning, Deep Learning, TensorFlow, ...）
- NLP/LLM（Transformers, BERT, GPT, RAG, ...）
- 推荐/搜索（Recommendation Systems, Elasticsearch, ...）
- 数据工程（Spark, Airflow, ETL, ...）
- 云/基础设施（AWS, Docker, Kubernetes, ...）

可根据需要添加新技能到词表。

## 下一步计划

后续 Milestone 将实现：
- ✅ ~~基于向量嵌入的语义匹配~~（M2 已完成）
- ✅ ~~批量匹配和排序功能~~（M2 已完成）
- ✅ ~~可解释的轻量排序层~~（M3 已完成）
- ✅ ~~集成 LLM 进行更智能的匹配分析和个性化建议~~（M4 已完成）
- 数据库集成存储职位和简历数据
- 用户认证和授权系统
- 缓存优化（Redis）
- 日志和监控
- 更多推荐算法（混合推荐、协同过滤等）

## 许可证

TBD
