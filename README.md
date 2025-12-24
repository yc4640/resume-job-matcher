# LM Match Service

## 项目简介

LM Match Service 是一个基于 FastAPI 的求职简历匹配服务。本项目目前处于 M3 阶段，实现了可解释的轻量排序层，在语义嵌入的基础上提供透明的打分机制。

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
│   ├── services/            # 业务逻辑服务
│   │   ├── __init__.py         # 服务包初始化
│   │   ├── embedding.py        # 文本嵌入服务 (M2)
│   │   ├── retrieval.py        # 检索和排序服务 (M2)
│   │   └── ranking.py          # 可解释排序服务 (M3)
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

### 4. 启动服务

```bash
# 方式一：使用 uvicorn 命令
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 方式二：直接运行 main.py
python main.py
```

服务启动后，访问 http://localhost:8000

### 5. 查看 API 文档

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

**说明（M3 更新）：**
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
- `total_jobs_searched`：从 jobs.jsonl 加载的总职位数量

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
- 集成 LLM 进行更智能的匹配分析和个性化建议
- 数据库集成存储职位和简历数据
- 用户认证和授权系统
- 缓存优化（Redis）
- 日志和监控
- 更多推荐算法（混合推荐、协同过滤等）

## 许可证

TBD
