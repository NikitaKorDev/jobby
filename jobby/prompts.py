"""Prompt construction for the determinator.

Both prompts keep everything that never changes (rules, resume, criteria) in a
fixed prefix, so Ollama can reuse its cached prefix between listings
(jobby-plan.md item 8).
"""

from __future__ import annotations

from .source import Listing

SYSTEM = (
    "You are a careful job-match evaluator. You compare one job listing with one "
    "candidate's resume and answer strictly in JSON. You never invent facts."
)

EVALUATION_RULES = """\
RULES
- The resume is the only evidence about the candidate. A technology that is not
  in the resume is unknown to the candidate; never assume related skills.
- missing_required lists required technologies from the listing that the resume
  does not show, using the listing's own names. It is [] when none are missing.
- worth_it is "no" when any required technology is missing, when a MUST-HAVE
  criterion is violated, or when the role clearly does not fit the candidate.
- chance is an integer 0-100 used only to rank listings for the candidate. It is
  not a real probability.
- reasoning is at most two short sentences, in English.
- Reply with the JSON object only: no markdown, no extra keys, no commentary."""

TAG_RULES = """\
RULES
- Extract only technologies the resume shows the candidate actually uses.
- Use short canonical names as they appear on job boards (Python, SQL, Docker).
- Do not infer technologies from the candidate's field of study or from titles.
- Reply with the JSON object only: no markdown, no extra keys, no commentary."""


def _listing_block(listing: Listing, description: str) -> str:
    lines = [
        f"Board: {listing.job_board or 'unknown'}",
        f"Title: {listing.title or 'unknown'}",
        f"Company: {listing.company or 'unknown'}",
        f"City: {listing.city or 'not stated'}",
        f"Work mode: {listing.work_mode or 'not stated'}",
        f"Seniority: {listing.seniority or 'not stated'}",
        f"Employment types: {', '.join(listing.employment_types) or 'not stated'}",
    ]
    if listing.salary_month or listing.salary_hour:
        salary = listing.salary_month or listing.salary_hour
        lines.append(f"Salary: {salary} {listing.currency or ''} ({listing.salary_type or 'n/a'})")
    lines.append(f"Tags: {', '.join(listing.technologies) or 'none'}")
    lines.append("")
    lines.append("DESCRIPTION")
    lines.append(description)
    return "\n".join(lines)


def evaluation_prompt(
    *,
    resume: str,
    technologies: list[str],
    criteria: str,
    listing: Listing,
    description: str,
) -> str:
    """Build the evaluation prompt: fixed prefix first, listing last."""
    return f"""\
CANDIDATE RESUME
{resume}

TECHNOLOGIES THE CANDIDATE KNOWS (normalized)
{', '.join(technologies) or 'none extracted'}

EVALUATION CRITERIA
{criteria}

OUTPUT FORMAT
{{"worth_it": "yes" | "no",
  "chance": <integer 0-100>,
  "reasoning": "<at most two short sentences>",
  "missing_required": ["<required technology>", ...]}}

{EVALUATION_RULES}

JOB LISTING
{_listing_block(listing, description)}
"""


def tag_prompt(*, resume: str) -> str:
    """Build the tag-extraction prompt (jobby-plan.md step 2)."""
    return f"""\
CANDIDATE RESUME
{resume}

OUTPUT FORMAT
{{"technologies": ["<technology>", ...]}}

{TAG_RULES}
"""
