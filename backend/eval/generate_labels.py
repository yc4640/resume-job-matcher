"""
Generate LLM-assisted weak labels for ALL job-resume pairs (full coverage).

This script:
1. Loads all resumes and jobs from JSONL files
2. Iterates through ALL combinations of (resume, job) pairs
3. Uses LLM to label each pair with a score 1-5
4. Saves labels to labels_suggested.jsonl (OVERWRITES old version directly)
5. Validates full coverage and reports any missing pairs

Label system (1-5):
- 1 = Not a match (Direction clearly irrelevant; little to no overlap)
- 2 = Weak match (Some overlap, but major gaps in core skills or experience)
- 3 = Partial match (Relevant direction and some key skills, but multiple noticeable gaps)
- 4 = Good match (Direction aligned and most important core skills present; gaps exist but are reasonable and commonly acceptable in real hiring)
- 5 = Strong match (Highly aligned direction with strong coverage of core skills; only minor or optional requirements missing)

IMPORTANT: Labels are generated ONLY from resume and job content.
NO ranking, topK, or heuristic information is provided to the LLM (avoid bias/cheating).
"""

import json
import csv
import os
import sys
import random
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Add parent directory to path to import schemas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import JobPosting, Resume


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


def load_existing_labels() -> Dict[tuple, Dict[str, Any]]:
    """
    Load existing labels from labels_suggested.jsonl for resume support.
    Returns: Dict mapping (resume_id, job_id) -> label_record
    """
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, "eval", "labels_suggested.jsonl")

    if not os.path.exists(filepath):
        return {}

    existing = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                key = (record['resume_id'], record['job_id'])
                existing[key] = record

    return existing


def generate_llm_label(resume: Resume, job: JobPosting) -> Dict[str, Any]:
    """
    Use LLM to generate a label (1-5) for a resume-job pair with evidence and notes.

    IMPORTANT: To avoid evaluation bias (label leakage), this function does NOT receive
    any system-computed results (matched_skills, gap_skills, scores, rankings, topK, heuristics).
    LLM judges relevance independently based solely on raw resume and job description.

    Returns dict with keys: label, confidence, evidence, notes
    """
    # Prepare the prompt - ONLY raw resume and job data, NO system outputs
    prompt = f"""You are an independent expert evaluator assessing job-resume match quality.

CRITICAL CONTEXT:
- You do NOT know how any system ranked this job
- You do NOT have access to any automated matching scores or rankings
- Judge match quality solely based on the resume and job description below

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

TASK: Rate the match quality on a 1-5 scale based ONLY on the information above:

Label Definitions:
- 1 = Not a match (Direction clearly irrelevant; little to no overlap)
- 2 = Weak match (Some overlap, but major gaps in core skills or experience)
- 3 = Partial match (Relevant direction and some key skills, but multiple noticeable gaps)
- 4 = Good match (Direction aligned and most important core skills present; gaps exist but are reasonable and commonly acceptable in real hiring)
- 5 = Strong match (Highly aligned direction with strong coverage of core skills; only minor or optional requirements missing)

IMPORTANT CALIBRATION RULE:
- A rating of 4 does NOT require the resume to meet every listed job requirement.
- Missing optional or commonly flexible requirements (e.g., degree level, years of experience, specific tools)
  should NOT automatically prevent a rating of 4 if core skills and direction align.

CRITICAL RULES:
- Base your rating ONLY on the resume and job description provided above
- Do NOT fabricate experience or skills not mentioned in the resume
- Evidence must be direct quotes or paraphrases from the resume/job description (max 200 chars each)
- Provide 2-4 evidence items to support your rating
- Keep notes concise (1-2 sentences)
- If something is not mentioned, you can note "Not evidenced" or "Not mentioned"
- Do NOT make claims without evidence from the text

Respond in this EXACT JSON format:
{{
  "label": <1-5>,
  "confidence": <0.0-1.0>,
  "evidence": [
    "<quote or paraphrase from resume/JD, max 200 chars>",
    "<quote or paraphrase from resume/JD, max 200 chars>",
    ...
  ],
  "notes": "<brief 1-2 sentence explanation>"
}}
"""

    try:
        # Call OpenAI API
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an independent expert evaluator judging job-resume match quality. You do NOT know how any system ranked these jobs. Be objective and evidence-based. Always provide evidence from the actual text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for consistent, factual output
            max_tokens=500,
            response_format={"type": "json_object"}  # Force JSON output
        )

        # Parse the JSON response
        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        # Validate and extract fields
        label = result.get("label", 2)
        confidence = result.get("confidence", 0.5)
        evidence = result.get("evidence", [])
        notes = result.get("notes", "")

        # Validate ranges
        if not isinstance(label, int) or label < 1 or label > 5:
            label = 2  # Default to weak match
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            confidence = 0.5
        if not isinstance(evidence, list):
            evidence = ["No specific evidence provided"]
        if not isinstance(notes, str):
            notes = "General assessment based on skills and experience"

        # Ensure evidence is between 2-4 items and each item is <= 200 chars
        evidence = [str(e)[:200] for e in evidence[:4]]
        if len(evidence) < 2:
            evidence.append("Additional context needed")

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "notes": notes[:500]  # Limit notes length
        }

    except Exception as e:
        print(f"  Warning: LLM labeling failed: {e}")
        print("  Falling back to neutral default label...")

        # Fallback: Use neutral default when LLM is unavailable
        return {
            "label": 2,  # Default to weak match (neutral choice)
            "confidence": 0.3,  # Low confidence indicates fallback
            "evidence": [
                "LLM labeling unavailable",
                "Default label assigned"
            ],
            "notes": "Fallback label (LLM unavailable) - requires manual review"
        }


def validate_coverage(labels_data: List[Dict], resumes: List[Resume], jobs: List[JobPosting]) -> bool:
    """
    Validate that all resume×job combinations are covered.
    Returns True if valid, False otherwise (and prints missing pairs).
    """
    labeled_pairs = set((r['resume_id'], r['job_id']) for r in labels_data)

    all_pairs = set()
    for resume in resumes:
        for job in jobs:
            all_pairs.add((resume.resume_id, job.job_id))

    missing_pairs = all_pairs - labeled_pairs

    if missing_pairs:
        print(f"\n❌ ERROR: Missing {len(missing_pairs)} pairs!")
        print("Missing pairs:")
        for resume_id, job_id in sorted(missing_pairs)[:10]:  # Show first 10
            print(f"  - ({resume_id}, {job_id})")
        if len(missing_pairs) > 10:
            print(f"  ... and {len(missing_pairs) - 10} more")
        return False

    print(f"\n✅ Coverage validation PASSED: All {len(all_pairs)} pairs are labeled!")
    return True


def main():
    """Main function to generate full 1-5 weak labels for all resume-job pairs."""
    # Change to backend directory so relative paths work correctly
    backend_dir = get_base_dir()
    original_dir = os.getcwd()
    os.chdir(backend_dir)

    try:
        print("=" * 80)
        print("LLM-Assisted Full Weak Label Generation (1-5 scale)")
        print("=" * 80)

        # Step 1: Load data
        print("\n[1/5] Loading data...")
        resumes = load_resumes()
        jobs = load_jobs()
        print(f"  Loaded {len(resumes)} resumes and {len(jobs)} jobs")
        print(f"  Total pairs to label: {len(resumes) * len(jobs)}")

        # Step 2: Load existing labels for resume support
        print("\n[2/5] Checking for existing labels (resume support)...")
        existing_labels = load_existing_labels()
        print(f"  Found {len(existing_labels)} existing labels")

        # Step 3: Generate all pairs with shuffling (reduce order bias)
        print("\n[3/5] Generating labels for ALL resume×job pairs...")
        print("  Using shuffled order to reduce positional bias...")

        all_pairs = []
        for resume in resumes:
            for job in jobs:
                all_pairs.append((resume, job))

        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_pairs)

        labels_data = []
        skipped = 0
        generated = 0

        for i, (resume, job) in enumerate(all_pairs, 1):
            key = (resume.resume_id, job.job_id)

            # Check if already labeled (resume support)
            if key in existing_labels:
                labels_data.append(existing_labels[key])
                skipped += 1
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(all_pairs)} ({skipped} skipped, {generated} generated)")
                continue

            # Generate new label
            if i % 5 == 0 or i == 1:
                print(f"\n  Pair {i}/{len(all_pairs)}: {resume.resume_id} × {job.job_id}")
                print(f"    Resume: {resume.education[:50]}...")
                print(f"    Job: {job.title[:50]}...", end=" ")

            # Generate LLM label - IMPORTANT: Only pass raw resume and job data
            llm_result = generate_llm_label(resume=resume, job=job)

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
            generated += 1

            if i % 5 == 0 or i == 1:
                print(f"[Label={llm_result['label']}, Conf={llm_result['confidence']:.2f}]")

        print(f"\n  Total labeled pairs: {len(labels_data)} ({skipped} skipped, {generated} newly generated)")

        # Step 4: Validate coverage
        print("\n[4/5] Validating coverage...")
        if not validate_coverage(labels_data, resumes, jobs):
            print("\n❌ Coverage validation FAILED. Aborting.")
            return

        # Step 5: Save labels_suggested.jsonl
        print("\n[5/5] Saving labels to labels_suggested.jsonl...")
        output_path = os.path.join("eval", "labels_suggested.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in labels_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  ✅ Saved {len(labels_data)} labels to {output_path}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("Summary Statistics:")
        print("=" * 80)
        print(f"Total resumes: {len(resumes)}")
        print(f"Total jobs: {len(jobs)}")
        print(f"Total labeled pairs: {len(labels_data)}")
        print(f"Coverage: {len(labels_data)}/{len(resumes)*len(jobs)} ({len(labels_data)/(len(resumes)*len(jobs))*100:.1f}%)")

        # Label distribution
        label_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for record in labels_data:
            label = record['label']
            if label in label_counts:
                label_counts[label] += 1

        print(f"\nLabel Distribution (1-5 scale):")
        print(f"  Label 1 (Not a match):   {label_counts[1]:3d} ({label_counts[1]/len(labels_data)*100:.1f}%)")
        print(f"  Label 2 (Weak match):    {label_counts[2]:3d} ({label_counts[2]/len(labels_data)*100:.1f}%)")
        print(f"  Label 3 (Partial match): {label_counts[3]:3d} ({label_counts[3]/len(labels_data)*100:.1f}%)")
        print(f"  Label 4 (Good match):    {label_counts[4]:3d} ({label_counts[4]/len(labels_data)*100:.1f}%)")
        print(f"  Label 5 (Strong match):  {label_counts[5]:3d} ({label_counts[5]/len(labels_data)*100:.1f}%)")

        avg_confidence = sum(r['confidence'] for r in labels_data) / len(labels_data)
        print(f"\nAverage Confidence: {avg_confidence:.2f}")

        print("\n" + "=" * 80)
        print("✅ Label generation complete!")
        print("=" * 80)
        print(f"\nOutput file:")
        print(f"  {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Run evaluation script: python scripts/eval_ablation.py")
        print(f"  2. Train LTR model (will be done during evaluation)")

    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
