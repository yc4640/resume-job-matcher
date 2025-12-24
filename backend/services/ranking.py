"""
Ranking service for explainable job-resume matching (M3).
Implements lightweight ranking layer on top of embedding-based retrieval.
"""

import yaml
from typing import List, Dict, Any, Set
from pathlib import Path

from schemas import JobPosting, Resume


# Global variables for caching
_config = None
_skills_vocab = None


def load_config(config_path: str = "config/ranking_config.yaml") -> Dict:
    """
    Load ranking configuration from YAML file.
    Config is cached after first load.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dict: Configuration dictionary
    """
    global _config
    if _config is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
    return _config


def load_skills_vocabulary(vocab_path: str = "data/skills_vocabulary.txt") -> Set[str]:
    """
    Load skills vocabulary from text file.
    Vocabulary is cached after first load.

    Args:
        vocab_path: Path to skills vocabulary file

    Returns:
        Set[str]: Set of normalized skill names
    """
    global _skills_vocab
    if _skills_vocab is None:
        skills = set()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    skills.add(line)
        _skills_vocab = skills
    return _skills_vocab


def normalize_skills(skills: List[str], vocab: Set[str]) -> Set[str]:
    """
    Normalize skills against vocabulary.
    Performs case-insensitive matching.

    Args:
        skills: List of skill strings
        vocab: Set of vocabulary skills

    Returns:
        Set[str]: Normalized skills that match vocabulary
    """
    normalized = set()

    # Create lowercase mapping of vocab
    vocab_lower = {s.lower(): s for s in vocab}

    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in vocab_lower:
            normalized.add(vocab_lower[skill_lower])

    return normalized


def calculate_skill_overlap(resume_skills: Set[str], job_skills: Set[str]) -> float:
    """
    Calculate skill overlap ratio.

    Args:
        resume_skills: Set of normalized resume skills
        job_skills: Set of normalized job skills

    Returns:
        float: Overlap ratio (0-1)
    """
    if not job_skills:
        return 0.0

    matched = resume_skills & job_skills
    return len(matched) / len(job_skills)


def calculate_keyword_bonus(
    resume_skills: Set[str],
    job_skills: Set[str],
    high_priority_keywords: List[str],
    multiplier: float,
    max_keywords: int
) -> float:
    """
    Calculate keyword bonus score.

    Args:
        resume_skills: Set of normalized resume skills
        job_skills: Set of normalized job skills
        high_priority_keywords: List of high-priority keywords
        multiplier: Bonus multiplier for high-priority matches
        max_keywords: Maximum keywords to consider for normalization

    Returns:
        float: Normalized keyword bonus score (0-1)
    """
    # Find matched high-priority keywords
    high_priority_set = {k.lower() for k in high_priority_keywords}
    matched_keywords = resume_skills & job_skills

    bonus_count = 0
    for skill in matched_keywords:
        if skill.lower() in high_priority_set:
            bonus_count += multiplier
        else:
            bonus_count += 1

    # Normalize by max possible keywords
    return min(bonus_count / max_keywords, 1.0) if max_keywords > 0 else 0.0


def calculate_gap_penalty(
    resume_skills: Set[str],
    job_skills: Set[str],
    critical_skills: List[str],
    critical_multiplier: float,
    max_gaps: int
) -> float:
    """
    Calculate gap penalty score.

    Args:
        resume_skills: Set of normalized resume skills
        job_skills: Set of normalized job skills
        critical_skills: List of critical skills
        critical_multiplier: Penalty multiplier for missing critical skills
        max_gaps: Maximum gaps to consider for normalization

    Returns:
        float: Normalized gap penalty score (0-1)
    """
    # Find missing skills
    gaps = job_skills - resume_skills

    if not gaps:
        return 0.0

    critical_set = {s.lower() for s in critical_skills}
    penalty_count = 0

    for skill in gaps:
        if skill.lower() in critical_set:
            penalty_count += critical_multiplier
        else:
            penalty_count += 1

    # Normalize by max possible gaps
    return min(penalty_count / max_gaps, 1.0) if max_gaps > 0 else 0.0


def calculate_final_score(
    embedding_score: float,
    skill_overlap: float,
    keyword_bonus: float,
    gap_penalty: float,
    config: Dict
) -> float:
    """
    Calculate final ranking score using weighted combination.

    Formula: final_score = w1*embedding + w2*skill_overlap + w3*keyword_bonus - w4*gap_penalty

    Args:
        embedding_score: Semantic similarity score (0-1)
        skill_overlap: Skill overlap ratio (0-1)
        keyword_bonus: Keyword bonus score (0-1)
        gap_penalty: Gap penalty score (0-1)
        config: Configuration dictionary

    Returns:
        float: Final ranking score
    """
    weights = config['weights']

    final_score = (
        weights['embedding'] * embedding_score +
        weights['skill_overlap'] * skill_overlap +
        weights['keyword_bonus'] * keyword_bonus -
        weights['gap_penalty'] * gap_penalty
    )

    return final_score


def rank_jobs_with_features(
    resume: Resume,
    jobs_with_embedding_scores: List[Dict[str, Any]],
    config_path: str = "config/ranking_config.yaml",
    vocab_path: str = "data/skills_vocabulary.txt"
) -> List[Dict[str, Any]]:
    """
    Re-rank jobs using explainable features on top of embedding scores.

    Args:
        resume: Resume object
        jobs_with_embedding_scores: List of dicts with 'job' and 'score' (embedding)
        config_path: Path to ranking config file
        vocab_path: Path to skills vocabulary file

    Returns:
        List[Dict]: Ranked jobs with features and final scores
    """
    # Load config and vocabulary
    config = load_config(config_path)
    vocab = load_skills_vocabulary(vocab_path)

    # Normalize resume skills
    resume_skills_normalized = normalize_skills(resume.skills, vocab)

    results = []

    for item in jobs_with_embedding_scores:
        job = item['job']
        embedding_score = item['score']

        # Normalize job skills
        job_skills_normalized = normalize_skills(job.skills, vocab)

        # Calculate features
        skill_overlap = calculate_skill_overlap(resume_skills_normalized, job_skills_normalized)

        keyword_bonus = calculate_keyword_bonus(
            resume_skills_normalized,
            job_skills_normalized,
            config['keywords']['high_priority'],
            config['keywords']['high_priority_multiplier'],
            config['normalization']['max_keywords']
        )

        gap_penalty = calculate_gap_penalty(
            resume_skills_normalized,
            job_skills_normalized,
            config['gap_penalty']['critical_skills'],
            config['gap_penalty']['critical_penalty_multiplier'],
            config['normalization']['max_gaps']
        )

        # Calculate final score
        final_score = calculate_final_score(
            embedding_score,
            skill_overlap,
            keyword_bonus,
            gap_penalty,
            config
        )

        # Store result with features
        results.append({
            'job': job,
            'embedding_score': embedding_score,
            'skill_overlap': skill_overlap,
            'keyword_bonus': keyword_bonus,
            'gap_penalty': gap_penalty,
            'final_score': final_score,
            'matched_skills': list(resume_skills_normalized & job_skills_normalized),
            'gap_skills': list(job_skills_normalized - resume_skills_normalized)
        })

    # Sort by final score (descending)
    results.sort(key=lambda x: x['final_score'], reverse=True)

    # Add rank
    for rank, result in enumerate(results, start=1):
        result['rank'] = rank

    return results


def explain_ranking(top_result: Dict[str, Any], config: Dict) -> str:
    """
    Generate human-readable explanation for why a job ranks first.

    Args:
        top_result: Top-ranked job result with features
        config: Configuration dictionary

    Returns:
        str: Explanation text
    """
    job = top_result['job']
    weights = config['weights']

    explanation_parts = [
        f"【{job.title}】Ranked #1 for the following reasons:",
        f"",
        f"1. Semantic Similarity: {top_result['embedding_score']:.3f} (Weight: {weights['embedding']})",
        f"   - The job description is highly semantically aligned with the resume content",
        f"",
        f"2. Skill Coverage: {top_result['skill_overlap']:.3f} (Weight: {weights['skill_overlap']})",
        f"   - Matched skills ({len(top_result['matched_skills'])}): {', '.join(top_result['matched_skills'][:5])}",
        f"   - Missing skills ({len(top_result['gap_skills'])}): {', '.join(top_result['gap_skills'][:5]) if top_result['gap_skills'] else 'None'}",
        f"",
        f"3. Keyword Bonus: {top_result['keyword_bonus']:.3f} (Weight: {weights['keyword_bonus']})",
        f"   - Matches high-priority skills",
        f"",
        f"4. Gap Penalty: {top_result['gap_penalty']:.3f} (Weight: {weights['gap_penalty']})",
        f"   - Penalty applied for missing critical skills",
        f"",
        f"Overall Score: {top_result['final_score']:.3f}"
    ]

    return "\n".join(explanation_parts)
