"""
Generate LLM-assisted weak labels for job-resume pairs.

This script:
1. Loads all resumes from resumes.jsonl
2. For each resume, gets Top-15 job recommendations using the current ranking system
3. Uses LLM to label each (resume, job) pair with a score 0-3
4. Saves labels to labels_suggested.jsonl
5. Generates labels_final.csv template for human correction

Label system (0-3):
- 0 = No match (clearly irrelevant/misaligned direction)
- 1 = Weak match (some relevant points, but lacks key skills/direction mismatch)
- 2 = Medium match (direction aligned, some skills met, some gaps exist)
- 3 = Strong match (highly aligned direction, high skill coverage, few gaps)
"""

import json
import csv
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Add parent directory to path to import schemas and services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import JobPosting, Resume
from services.retrieval import rank_jobs
from services.ranking import rank_jobs_with_features


def get_base_dir():
    """Get the backend directory path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    return backend_dir


def load_resumes() -> List[Resume]:
    """Load all resumes from JSONL file."""
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, "data", "resumes.jsonl")
    resumes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                resume_data = json.loads(line.strip())
                resumes.append(Resume(**resume_data))
    return resumes


def load_jobs() -> List[JobPosting]:
    """Load all jobs from JSONL file."""
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, "data", "jobs.jsonl")
    jobs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                job_data = json.loads(line.strip())
                jobs.append(JobPosting(**job_data))
    return jobs


def get_top_k_jobs(resume: Resume, jobs: List[JobPosting], k: int = 15) -> List[Dict[str, Any]]:
    """
    Get top-k job recommendations for a resume using the current ranking system.

    Returns list of dicts with keys: job, rank, embedding_score, final_score, matched_skills, gap_skills
    """
    # Step 1: Get embedding-based similarity scores
    embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))

    # Step 2: Re-rank using explainable features
    ranked_results = rank_jobs_with_features(resume, embedding_results)

    # Step 3: Take top-k results
    return ranked_results[:k]


def generate_llm_label(resume: Resume, job: JobPosting) -> Dict[str, Any]:
    """
    Use LLM to generate a label (0-3) for a resume-job pair with evidence and notes.

    IMPORTANT: To avoid evaluation bias (label leakage), this function does NOT receive
    any system-computed results (matched_skills, gap_skills, scores, rankings).
    LLM judges relevance independently based solely on raw resume and job description.

    Returns dict with keys: label, confidence, evidence, notes
    """
    # Prepare the prompt - ONLY raw resume and job data, NO system outputs
    prompt = f"""You are an independent human evaluator assessing job-resume relevance.

CRITICAL CONTEXT:
- You do NOT know how any system ranked this job
- You do NOT have access to any automated matching scores
- Judge relevance solely based on the resume and job description below

RESUME:
Education: {resume.education}
Experience: {resume.experience}
Projects: {resume.projects}
Skills: {', '.join(resume.skills)}

JOB POSTING:
Title: {job.title}
Company: {job.company or 'N/A'}
Responsibilities: {job.responsibilities}
Requirements: {job.requirements_text}
Required Skills: {', '.join(job.skills)}

TASK: Rate the match quality on a 0-3 scale based ONLY on the information above:

Label Definitions:
- 0 = No match (clearly irrelevant, completely different direction)
- 1 = Weak match (some overlap, but lacks key skills or misaligned direction)
- 2 = Medium match (direction aligned, partial skill coverage, noticeable gaps)
- 3 = Strong match (highly aligned direction, strong skill coverage, minimal gaps)

CRITICAL RULES:
- Base your rating ONLY on the resume and job description provided above
- Do NOT fabricate experience or skills not mentioned in the resume
- Evidence must be direct quotes or paraphrases from the resume/job description
- Keep evidence concise (max 2 items, each <50 words)
- Keep notes brief (1 sentence explanation)

Respond in this EXACT format:
LABEL: <0-3>
CONFIDENCE: <0.0-1.0>
EVIDENCE_1: <quote from resume or JD>
EVIDENCE_2: <quote from resume or JD>
NOTES: <one sentence explanation>
"""

    try:
        # Call OpenAI API
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an independent human evaluator judging job-resume relevance. You do NOT know how any system ranked these jobs. Be objective and evidence-based."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for consistent, factual output
            max_tokens=300
        )

        # Parse the response
        content = response.choices[0].message.content.strip()

        # Extract fields using simple parsing
        label = None
        confidence = None
        evidence = []
        notes = None

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('LABEL:'):
                try:
                    label = int(line.split('LABEL:')[1].strip())
                except:
                    label = 1  # Default to weak match if parsing fails
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split('CONFIDENCE:')[1].strip())
                except:
                    confidence = 0.5  # Default confidence
            elif line.startswith('EVIDENCE_'):
                evidence_text = ':'.join(line.split(':')[1:]).strip()
                if evidence_text:
                    evidence.append(evidence_text)
            elif line.startswith('NOTES:'):
                notes = ':'.join(line.split(':')[1:]).strip()

        # Validate and set defaults
        if label is None or label < 0 or label > 3:
            label = 1  # Default to weak match
        if confidence is None or confidence < 0 or confidence > 1:
            confidence = 0.5
        if not evidence:
            evidence = ["No specific evidence provided"]
        if not notes:
            notes = "General assessment based on skills and experience"

        # Limit evidence to 2 items
        evidence = evidence[:2]

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "notes": notes
        }

    except Exception as e:
        print(f"Warning: LLM labeling failed: {e}")
        print("Falling back to neutral default label...")

        # Fallback: Use neutral default when LLM is unavailable
        # Do NOT use system scores to avoid label leakage
        return {
            "label": 1,  # Default to weak match (neutral choice)
            "confidence": 0.3,  # Low confidence indicates fallback
            "evidence": [
                "LLM labeling unavailable",
                "Default label assigned"
            ],
            "notes": "Fallback label (LLM unavailable) - requires manual review"
        }


def main():
    """Main function to generate labels."""
    # Change to backend directory so relative paths work correctly
    backend_dir = get_base_dir()
    original_dir = os.getcwd()
    os.chdir(backend_dir)

    try:
        print("=" * 80)
        print("LLM-Assisted Label Generation for Job-Resume Matching")
        print("=" * 80)

        # Load data
        print("\n[1/5] Loading data...")
        resumes = load_resumes()
        jobs = load_jobs()
        print(f"  Loaded {len(resumes)} resumes and {len(jobs)} jobs")

        # Generate labels
        print("\n[2/5] Generating labels for Top-15 recommendations per resume...")
        labels_data = []
        total_pairs = 0

        for i, resume in enumerate(resumes, 1):
            print(f"\n  Resume {i}/{len(resumes)} (ID: {resume.resume_id})")

            # Get Top-15 jobs for this resume
            top_jobs = get_top_k_jobs(resume, jobs, k=15)

            for j, result in enumerate(top_jobs, 1):
                job = result["job"]
                print(f"    Labeling pair {j}/15: {job.title[:50]}...", end=" ")

                # Generate LLM label - IMPORTANT: Only pass raw resume and job data
                # Do NOT pass system outputs (matched_skills, gap_skills, scores) to avoid label leakage
                llm_result = generate_llm_label(
                    resume=resume,
                    job=job
                )

                # Create label record
                label_record = {
                    "resume_id": resume.resume_id,
                    "job_id": job.job_id,
                    "label": llm_result["label"],
                    "confidence": llm_result["confidence"],
                    "evidence": llm_result["evidence"],
                    "notes": llm_result["notes"]
                }
                labels_data.append(label_record)
                total_pairs += 1

                print(f"[Label={llm_result['label']}, Conf={llm_result['confidence']:.2f}]")

        print(f"\n  Total labeled pairs: {total_pairs}")

        # Save labels_suggested.jsonl (save to eval directory)
        print("\n[3/5] Saving labels to labels_suggested.jsonl...")
        output_path = os.path.join("eval", "labels_suggested.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in labels_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  Saved {len(labels_data)} labels to {output_path}")

        # Generate labels_final.csv template for human correction (save to eval directory)
        print("\n[4/5] Generating labels_final.csv template...")
        csv_path = os.path.join("eval", "labels_final.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['resume_id', 'job_id', 'suggested_label', 'final_label',
                         'confidence', 'evidence_1', 'evidence_2', 'notes']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in labels_data:
                writer.writerow({
                    'resume_id': record['resume_id'],
                    'job_id': record['job_id'],
                    'suggested_label': record['label'],
                    'final_label': '',  # Empty for human to fill
                    'confidence': record['confidence'],
                    'evidence_1': record['evidence'][0] if len(record['evidence']) > 0 else '',
                    'evidence_2': record['evidence'][1] if len(record['evidence']) > 1 else '',
                    'notes': record['notes']
                })

        print(f"  Saved CSV template to {csv_path}")

        # Print summary statistics
        print("\n[5/5] Summary Statistics:")
        print(f"  Total resumes: {len(resumes)}")
        print(f"  Total jobs: {len(jobs)}")
        print(f"  Total labeled pairs: {len(labels_data)}")
        print(f"  Labels per resume: {len(labels_data) // len(resumes)}")

        # Label distribution
        label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for record in labels_data:
            label_counts[record['label']] += 1

        print(f"\n  Label Distribution:")
        print(f"    Label 0 (No match):     {label_counts[0]:3d} ({label_counts[0]/len(labels_data)*100:.1f}%)")
        print(f"    Label 1 (Weak match):   {label_counts[1]:3d} ({label_counts[1]/len(labels_data)*100:.1f}%)")
        print(f"    Label 2 (Medium match): {label_counts[2]:3d} ({label_counts[2]/len(labels_data)*100:.1f}%)")
        print(f"    Label 3 (Strong match): {label_counts[3]:3d} ({label_counts[3]/len(labels_data)*100:.1f}%)")

        avg_confidence = sum(r['confidence'] for r in labels_data) / len(labels_data)
        print(f"\n  Average Confidence: {avg_confidence:.2f}")

        print("\n" + "=" * 80)
        print("Label generation complete!")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  1. {output_path} - LLM-generated labels (JSONL format)")
        print(f"  2. {csv_path} - Template for human correction (CSV format)")
        print(f"\nNext steps:")
        print(f"  1. Review {csv_path} and fill in 'final_label' column where needed")
        print(f"  2. Run evaluation script to compute metrics")

    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
