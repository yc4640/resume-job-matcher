"""
RAG (Retrieval-Augmented Generation) service for M4.
Generates evidence-based explanations for job recommendations using LLM.
"""

import os
import re
from typing import Dict, List, Any, Tuple
from openai import OpenAI

from schemas import JobPosting, Resume
from services.embedding import get_embedding_model
from services.retrieval import cosine_similarity

# Global LLM client
_llm_client = None


def get_llm_client() -> OpenAI:
    """
    Get OpenAI client instance (lazy initialization).

    Returns:
        OpenAI: Initialized OpenAI client
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Did you load .env?")
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _llm_client = OpenAI(api_key=api_key)
    return _llm_client


def construct_job_evidence(job: JobPosting) -> Dict[str, str]:
    """
    Construct evidence dictionary from job posting.

    Args:
        job: JobPosting object

    Returns:
        Dict containing job evidence fields
    """
    return {
        "title": job.title,
        "responsibilities": job.responsibilities,
        "requirements_text": job.requirements_text,
        "skills": ", ".join(job.skills)
    }


def construct_resume_evidence(resume: Resume) -> Dict[str, str]:
    """
    Construct evidence dictionary from resume.

    Args:
        resume: Resume object

    Returns:
        Dict containing resume evidence fields
    """
    return {
        "education": resume.education,
        "projects": resume.projects,
        "experience": resume.experience,
        "skills": ", ".join(resume.skills)
    }


def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    Split text into chunks by sentences with approximate size limit.

    Args:
        text: Input text to chunk
        chunk_size: Approximate maximum chunk size in characters

    Returns:
        List of text chunks
    """
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def retrieve_relevant_context(
    job_evidence: Dict[str, str],
    resume_evidence: Dict[str, str],
    top_k_chunks: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Retrieve the most relevant chunks from job and resume using embedding similarity.

    Args:
        job_evidence: Job evidence dictionary
        resume_evidence: Resume evidence dictionary
        top_k_chunks: Number of top chunks to retrieve from each side

    Returns:
        Tuple of (job_chunks, resume_chunks) - most relevant chunks
    """
    model = get_embedding_model()

    # Chunk job evidence
    job_chunks = []
    for field, content in job_evidence.items():
        if content and field != "skills":  # Skip skills as they're already concise
            chunks = chunk_text(content)
            job_chunks.extend([(field, chunk) for chunk in chunks])

    # Chunk resume evidence
    resume_chunks = []
    for field, content in resume_evidence.items():
        if content and field != "skills":  # Skip skills as they're already concise
            chunks = chunk_text(content)
            resume_chunks.extend([(field, chunk) for chunk in chunks])

    # If we have few chunks, return all
    if len(job_chunks) <= top_k_chunks and len(resume_chunks) <= top_k_chunks:
        return (
            [f"[{field}] {chunk}" for field, chunk in job_chunks],
            [f"[{field}] {chunk}" for field, chunk in resume_chunks]
        )

    # Compute embeddings for all chunks
    job_chunk_texts = [chunk for _, chunk in job_chunks]
    resume_chunk_texts = [chunk for _, chunk in resume_chunks]

    job_embeddings = model.encode(job_chunk_texts, convert_to_numpy=True)
    resume_embeddings = model.encode(resume_chunk_texts, convert_to_numpy=True)

    # Calculate cross-similarity: how relevant each job chunk is to resume chunks
    job_scores = []
    for i, job_emb in enumerate(job_embeddings):
        max_similarity = max([
            cosine_similarity(job_emb, resume_emb)
            for resume_emb in resume_embeddings
        ])
        job_scores.append((max_similarity, i))

    # Calculate cross-similarity: how relevant each resume chunk is to job chunks
    resume_scores = []
    for i, resume_emb in enumerate(resume_embeddings):
        max_similarity = max([
            cosine_similarity(resume_emb, job_emb)
            for job_emb in job_embeddings
        ])
        resume_scores.append((max_similarity, i))

    # Sort and select top-k
    job_scores.sort(reverse=True)
    resume_scores.sort(reverse=True)

    top_job_indices = [idx for _, idx in job_scores[:top_k_chunks]]
    top_resume_indices = [idx for _, idx in resume_scores[:top_k_chunks]]

    # Format selected chunks with field labels
    selected_job_chunks = [
        f"[{job_chunks[i][0]}] {job_chunks[i][1]}"
        for i in top_job_indices
    ]
    selected_resume_chunks = [
        f"[{resume_chunks[i][0]}] {resume_chunks[i][1]}"
        for i in top_resume_indices
    ]

    return selected_job_chunks, selected_resume_chunks


def generate_explanation_with_llm(
    job: JobPosting,
    resume: Resume,
    job_context: List[str],
    resume_context: List[str],
    matched_skills: List[str],
    gap_skills: List[str],
    final_score: float
) -> Dict[str, str]:
    """
    Generate explanation, gap analysis, and improvement suggestions using LLM.

    Args:
        job: JobPosting object
        resume: Resume object
        job_context: Selected job evidence chunks
        resume_context: Selected resume evidence chunks
        matched_skills: List of matched skills
        gap_skills: List of missing skills
        final_score: Final ranking score

    Returns:
        Dict with 'explanation', 'gap_analysis', 'improvement_suggestions'
    """
    # Construct evidence-based prompt
    prompt = f"""You are an expert career advisor analyzing job-resume matches. Based ONLY on the evidence provided below, generate a brief analysis.

JOB POSITION: {job.title}

JOB EVIDENCE (most relevant excerpts):
{chr(10).join(f"- {chunk}" for chunk in job_context)}

Job Required Skills: {", ".join(job.skills)}

RESUME EVIDENCE (most relevant excerpts):
{chr(10).join(f"- {chunk}" for chunk in resume_context)}

Candidate Skills: {", ".join(resume.skills)}

MATCHING ANALYSIS:
- Matched Skills: {", ".join(matched_skills) if matched_skills else "None"}
- Missing Skills: {", ".join(gap_skills) if gap_skills else "None"}
- Overall Match Score: {final_score:.2f}
SOURCE OF TRUTH:
- "Missing Skills" are the ONLY allowed SKILL gaps.
- Non-skill requirements may be flagged ONLY as "not evidenced" if they appear explicitly in requirements_text
  and are not mentioned in the resume evidence.



Based ONLY on the evidence above, provide:

1. EXPLANATION (2-3 sentences): Why this job is a good fit for the candidate. Reference specific evidence from the job requirements and resume.

2. GAP_ANALYSIS (2–4 sentences):
Provide TWO sub-sections:

A) SKILL_GAPS:
- You MUST base this ONLY on "Missing Skills" above.
- Do NOT introduce new skills.
- If "Missing Skills" is "None", state: "No skill gaps identified."

B) REQUIREMENT_GAPS (Qualifications / constraints):
- Consider ONLY explicit constraints in "requirements_text" that are NOT skills
  (e.g., degree level like PhD/MS/BS, years of experience, publication record, location, clearance).
- For each such constraint, check whether the RESUME EVIDENCE mentions it.
- If the resume does NOT mention it, you should explicitly state what constraints that is Not evidenced / Not mentioned in the resume.
- List at most 1–2 items. If all are evidenced or not applicable, state: "No requirement gaps evidenced."



3. IMPROVEMENT_SUGGESTIONS (2-3 bullet points): Concrete, actionable steps the candidate can take to improve their fit for this role.

CRITICAL RULES:
- Base your analysis ONLY on the evidence provided above
- Reference specific details from the job and resume evidence
- Do not make assumptions or add information not present in the evidence
- Keep each section concise and focused


Format your response as:
EXPLANATION: [your explanation]
GAP_ANALYSIS: [your gap analysis]
IMPROVEMENT_SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
- [suggestion 3]
"""

    # Call LLM
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model for explanations
            messages=[
                {"role": "system", "content": "You are an expert career advisor who provides evidence-based job matching analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for factual, consistent outputs
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()

        # Parse the response
        explanation = ""
        gap_analysis = ""
        improvement_suggestions = ""

        # Extract sections using regex
        exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=GAP_ANALYSIS:|$)', content, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip()

        gap_match = re.search(r'GAP_ANALYSIS:\s*(.+?)(?=IMPROVEMENT_SUGGESTIONS:|$)', content, re.DOTALL)
        if gap_match:
            gap_analysis = gap_match.group(1).strip()

        imp_match = re.search(r'IMPROVEMENT_SUGGESTIONS:\s*(.+)', content, re.DOTALL)
        if imp_match:
            improvement_suggestions = imp_match.group(1).strip()

        return {
            "explanation": explanation,
            "gap_analysis": gap_analysis,
            "improvement_suggestions": improvement_suggestions
        }


    except Exception as e:

        print("LLM CALL FAILED:", repr(e))

        raise


def generate_rag_explanation(
    job: JobPosting,
    resume: Resume,
    matched_skills: List[str],
    gap_skills: List[str],
    final_score: float
) -> Dict[str, str]:
    """
    Main RAG pipeline: construct evidence, retrieve context, generate explanation.

    Args:
        job: JobPosting object
        resume: Resume object
        matched_skills: List of matched skills
        gap_skills: List of missing skills
        final_score: Final ranking score

    Returns:
        Dict with 'explanation', 'gap_analysis', 'improvement_suggestions'
    """
    # Step 1: Construct evidence
    job_evidence = construct_job_evidence(job)
    resume_evidence = construct_resume_evidence(resume)

    # Step 2: Retrieve relevant context chunks
    job_context, resume_context = retrieve_relevant_context(
        job_evidence,
        resume_evidence,
        top_k_chunks=3
    )

    # Step 3: Generate explanation with LLM
    result = generate_explanation_with_llm(
        job,
        resume,
        job_context,
        resume_context,
        matched_skills,
        gap_skills,
        final_score
    )

    return result
