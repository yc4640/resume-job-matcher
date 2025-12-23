# LM Match Service

## 项目简介

LM Match Service 是一个基于 FastAPI 的求职简历匹配服务。本项目目前处于 M1 阶段，实现了基于技能匹配的结构化匹配算法。

### 当前功能 (M1)

- ✅ 健康检查接口 (`/health`)
- ✅ 职位-简历匹配接口 (`/match`) - 返回结构化匹配结果
- ✅ 使用 Pydantic 定义数据模型（JobPosting、Resume、MatchResponse）
- ✅ 基于技能集合的匹配算法（不使用 LLM）
- ✅ 提供匹配分数、匹配技能、技能差距和学习建议
- ✅ RESTful API 设计

## 项目结构

```
lm/
├── backend/
│   ├── main.py              # FastAPI 主应用文件
│   ├── schemas.py           # Pydantic 数据模型定义
│   ├── test_match.py        # 匹配接口测试文件
│   ├── requirements.txt     # Python 依赖
│   └── data/
│       ├── sample_job.json     # 示例职位数据
│       └── sample_resume.json  # 示例简历数据
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
  "match_score": 57,
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

## 技术栈

- **FastAPI**: 现代、高性能的 Python Web 框架
- **Pydantic**: 数据验证和设置管理
- **Uvicorn**: ASGI 服务器

## 匹配算法说明

当前版本（M1）使用基于技能集合的简单匹配算法：

1. **匹配技能** (matched_skills)：求职者技能与职位要求技能的交集
2. **技能差距** (gaps)：职位要求技能中求职者不具备的技能
3. **匹配分数** (match_score)：匹配技能数量占职位要求技能总数的百分比
   - 公式：`match_score = (len(matched_skills) / len(job.skills)) * 100`
   - 如果职位没有技能要求，则返回 0
4. **学习建议** (suggestions)：针对每个技能差距提供学习建议

## 下一步计划

后续 Milestone 将实现：
- 集成 LLM 进行更智能的匹配分析
- 基于向量嵌入的语义匹配
- 数据库集成存储职位和简历数据
- 批量匹配和排序功能
- 更多的 API 端点
- 认证和授权
- 日志和监控

## 许可证

TBD
