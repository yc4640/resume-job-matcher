"""
Utility functions for skill extraction and merging.
Helps avoid overly strict skill matching by extracting skills from resume text.
"""

import re
from typing import List, Set
from schemas import Resume


def extract_skills_from_text(text: str, vocab: List[str]) -> List[str]:
    """
    Extract skills from text based on vocabulary matching.

    Uses word-boundary matching to avoid false positives (e.g., "C" shouldn't match "Cloud").
    Handles special characters in skills like "C++", "C#" using re.escape.

    Args:
        text: Input text to search for skills (may be empty)
        vocab: List of skill terms to search for

    Returns:
        List of matched skills (preserving original case from vocab)
    """
    if not text or not vocab:
        return []

    matched_skills = []
    text_lower = text.lower()

    for skill in vocab:
        skill_lower = skill.lower()

        # Escape special regex characters (e.g., C++, C#, .NET)
        skill_pattern = re.escape(skill_lower)

        # Check if skill contains non-word characters (special symbols)
        # For skills like "C++", "C#", ".NET", word boundaries don't work well
        has_special_chars = bool(re.search(r'\W', skill))

        if has_special_chars:
            # For special character skills, use a more flexible pattern
            # Match if preceded/followed by whitespace or start/end of string
            pattern = r'(?:^|\s)' + skill_pattern + r'(?:\s|$|[,;.])'
        else:
            # Use word boundary for normal alphanumeric skills
            # \b ensures we don't match "C" inside "Cloud" or "React" inside "Reactivity"
            pattern = r'\b' + skill_pattern + r'\b'

        # Case-insensitive search
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched_skills.append(skill)  # Return original case from vocab

    return matched_skills


def merge_resume_skills(resume: Resume, vocab: List[str]) -> List[str]:
    """
    Merge user-provided skills with auto-extracted skills from resume text.

    This helps avoid overly strict matching when skills are mentioned in
    experience/projects/education but not explicitly listed in resume.skills.

    Args:
        resume: Resume object
        vocab: Skills vocabulary list

    Returns:
        List of merged skills (user skills + extracted skills, deduplicated)
        Order: user-provided skills first, then extracted skills
    """
    # Assemble text from resume fields
    text_parts = []
    if resume.education:
        text_parts.append(resume.education)
    if resume.projects:
        text_parts.append(resume.projects)
    if resume.experience:
        text_parts.append(resume.experience)

    combined_text = " ".join(text_parts)

    # Extract skills from text
    extracted_skills = extract_skills_from_text(combined_text, vocab)

    # Merge: user skills + extracted skills (deduplicated)
    # Use set for deduplication, but preserve order
    user_skills_set = set(skill.lower() for skill in resume.skills)

    # Create merged list: start with user skills
    merged = list(resume.skills)

    # Add extracted skills that weren't already in user skills
    for skill in extracted_skills:
        if skill.lower() not in user_skills_set:
            merged.append(skill)
            user_skills_set.add(skill.lower())  # Track to avoid duplicates

    return merged


# Soft skills that should not contribute to gap_penalty
# These skills are important but their absence shouldn't penalize candidates heavily
SOFT_SKILLS = {
    "communication",
    "leadership",
    "collaboration",
    "teamwork",
    "problem solving",
    "critical thinking",
    "time management",
    "adaptability",
    "creativity",
    "work ethic",
    "interpersonal skills",
    "presentation skills",
    "negotiation",
    "conflict resolution",
    "decision making",
    "emotional intelligence",
    "mentoring",
    "coaching",
    "stakeholder management",
    "project management",  # Debatable, but often considered soft
    "agile methodologies",  # Debatable
}


def filter_soft_skills(skills: List[str]) -> List[str]:
    """
    Filter out soft skills from a skill list.

    Useful for gap_penalty calculation where we don't want to penalize
    missing soft skills as heavily as technical skills.

    Args:
        skills: List of skills to filter

    Returns:
        List of skills with soft skills removed
    """
    soft_skills_lower = {s.lower() for s in SOFT_SKILLS}
    return [skill for skill in skills if skill.lower() not in soft_skills_lower]
