"""
Test file for the /match endpoint.
Demonstrates the job-resume matching functionality.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_match_full_match():
    """Test case where resume has all required skills"""
    request_data = {
        "job": {
            "title": "Senior Python Developer",
            "responsibilities": "Develop backend services",
            "requirements_text": "5+ years Python experience",
            "skills": ["Python", "FastAPI", "PostgreSQL"],
            "company": "Tech Corp",
            "location": "Remote",
            "level": "Senior"
        },
        "resume": {
            "education": "BS Computer Science",
            "projects": "Built several web applications",
            "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
            "experience": "6 years of backend development"
        }
    }

    response = client.post("/match", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["match_score"] == 100
    assert set(data["matched_skills"]) == {"Python", "FastAPI", "PostgreSQL"}
    assert data["gaps"] == []
    assert data["suggestions"] == []

    print("Test 1 passed: Full match - 100% score")


def test_match_partial_match():
    """Test case where resume has some required skills"""
    request_data = {
        "job": {
            "title": "Full Stack Developer",
            "responsibilities": "Build web applications",
            "requirements_text": "Experience with modern web stack",
            "skills": ["Python", "React", "PostgreSQL", "Docker"],
        },
        "resume": {
            "education": "BS Software Engineering",
            "projects": "E-commerce platform",
            "skills": ["Python", "React"],
            "experience": "3 years of web development"
        }
    }

    response = client.post("/match", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["match_score"] == 50  # 2 out of 4 skills
    assert set(data["matched_skills"]) == {"Python", "React"}
    assert set(data["gaps"]) == {"PostgreSQL", "Docker"}
    assert len(data["suggestions"]) == 2

    print("Test 2 passed: Partial match - 50% score")


def test_match_no_match():
    """Test case where resume has no required skills"""
    request_data = {
        "job": {
            "title": "Frontend Developer",
            "responsibilities": "Build UI components",
            "requirements_text": "Frontend experience required",
            "skills": ["React", "TypeScript", "CSS"],
        },
        "resume": {
            "education": "BS Computer Science",
            "projects": "Backend APIs",
            "skills": ["Python", "Django", "PostgreSQL"],
            "experience": "2 years backend development"
        }
    }

    response = client.post("/match", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["match_score"] == 0
    assert data["matched_skills"] == []
    assert set(data["gaps"]) == {"React", "TypeScript", "CSS"}
    assert len(data["suggestions"]) == 3

    print("Test 3 passed: No match - 0% score")


def test_match_empty_job_skills():
    """Test case where job has no required skills"""
    request_data = {
        "job": {
            "title": "General Position",
            "responsibilities": "Various tasks",
            "requirements_text": "Open to all",
            "skills": [],
        },
        "resume": {
            "education": "BS Computer Science",
            "projects": "Various projects",
            "skills": ["Python", "Java"],
            "experience": "1 year"
        }
    }

    response = client.post("/match", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["match_score"] == 0
    assert data["matched_skills"] == []
    assert data["gaps"] == []
    assert data["suggestions"] == []

    print("Test 4 passed: Empty job skills - 0% score")


if __name__ == "__main__":
    test_match_full_match()
    test_match_partial_match()
    test_match_no_match()
    test_match_empty_job_skills()
    print("\nAll tests passed!")
