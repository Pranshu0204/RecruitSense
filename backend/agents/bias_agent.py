"""Bias signal detector — pure regex/keyword, no LLM call, never affects scoring.

Returns advisory flags about resume content that could (a) bias a human
reviewer or (b) be unnecessary personal information the candidate shouldn't
include. Flags are surfaced in the UI so recruiters can request anonymization,
but are explicitly NOT fed into the scorer prompt.

Word lists for gender-coded language are adapted from the Gaucher, Friesen &
Kay (2011) "agentic vs. communal" lexicons commonly used in resume-bias tools.
"""

from __future__ import annotations

import re

from backend.utils.logger import get_logger

logger = get_logger(__name__)


# --- Gender-coded vocabulary (Gaucher & Friesen-style) -----------------------

MASCULINE_CODED: frozenset[str] = frozenset({
    "active", "adventurous", "aggressive", "ambitious", "analytical", "assertive",
    "athletic", "autonomous", "boast", "challenging", "competitive", "confident",
    "courageous", "decide", "decisive", "determined", "dominant", "driven",
    "fearless", "force", "headstrong", "hostile", "independent", "individual",
    "intellectual", "lead", "logical", "objective", "outspoken", "principled",
    "rugged", "self-confident", "self-reliant", "self-sufficient", "stubborn",
    "superior", "unreasonable",
})

FEMININE_CODED: frozenset[str] = frozenset({
    "agree", "affectionate", "cheer", "collaborative", "committed", "communal",
    "compassionate", "connect", "considerate", "cooperative", "depend",
    "emotional", "empathetic", "feel", "flatter", "gentle", "honest",
    "interpersonal", "interdependent", "kind", "kinship", "loyal", "modesty",
    "nag", "nurturing", "pleasant", "polite", "quiet", "responsible",
    "sensitive", "submissive", "support", "sympathetic", "tender", "together",
    "trust", "understanding", "warm", "whine", "yield",
})


# --- Personal-info regexes / keyword lists -----------------------------------

_LOCATION_REGEX = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b"
)
_GRAD_YEAR_REGEX = re.compile(
    r"\b(?:graduated|graduation|class of|b\.?s\.?|m\.?s\.?|ph\.?d\.?)\s*[:\-]?\s*"
    r"(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_DOB_REGEX = re.compile(r"\bdate of birth\b|\bd\.?o\.?b\.?\b|\bborn:?\s*\d", re.IGNORECASE)
_GENDER_REGEX = re.compile(
    r"\bgender\s*[:\-]\s*(?:male|female|m|f|man|woman)\b", re.IGNORECASE
)

_PHOTO_KEYWORDS: tuple[str, ...] = ("headshot", "photograph attached", "photo attached", "picture attached")
_MARITAL_KEYWORDS: tuple[str, ...] = ("married", "single", "divorced", "marital status")
_RELIGION_KEYWORDS: tuple[str, ...] = (
    "christian", "muslim", "hindu", "jewish", "buddhist", "atheist", "religion:",
)
_NATIONALITY_KEYWORDS: tuple[str, ...] = (
    "nationality:", "citizenship:", "country of origin:",
)


def detect_bias_signals(text: str) -> list[str]:
    """Scan resume text for bias signals and PII over-disclosure.

    Args:
        text: raw resume text.

    Returns:
        Human-readable advisory flags. Empty list if nothing flagged.

    Note:
        These flags MUST NOT be passed to the scorer prompt — the scorer's
        contract is to ignore them entirely. They are attached to ``ScoreOutput``
        for the recruiter UI only.
    """
    if not text:
        return []

    flags: list[str] = []
    lower = text.lower()
    words = set(re.findall(r"\b[a-z]+(?:-[a-z]+)?\b", lower))

    # Gendered language imbalance (only flag when one side dominates strongly)
    masc_hits = words & MASCULINE_CODED
    fem_hits = words & FEMININE_CODED
    if len(masc_hits) >= 3 and len(masc_hits) > len(fem_hits) * 2:
        flags.append(
            f"Heavy masculine-coded language ({len(masc_hits)} terms) — "
            "may bias readers."
        )
    if len(fem_hits) >= 3 and len(fem_hits) > len(masc_hits) * 2:
        flags.append(
            f"Heavy feminine-coded language ({len(fem_hits)} terms) — "
            "may bias readers."
        )

    if _DOB_REGEX.search(text):
        flags.append("Date of birth disclosed — omit to avoid age bias.")

    if _GRAD_YEAR_REGEX.search(text):
        flags.append("Graduation year disclosed — consider omitting (age proxy).")

    if _GENDER_REGEX.search(text):
        flags.append("Explicit gender disclosed — omit unless legally required.")

    if _LOCATION_REGEX.search(text):
        flags.append("Full home address disclosed — city/country alone is sufficient.")

    if any(k in lower for k in _MARITAL_KEYWORDS):
        flags.append("Marital status disclosed — not relevant to hiring.")

    if any(k in lower for k in _RELIGION_KEYWORDS):
        flags.append("Religion disclosed — not relevant to hiring.")

    if any(k in lower for k in _NATIONALITY_KEYWORDS):
        flags.append("Nationality/citizenship disclosed beyond legal need.")

    if any(k in lower for k in _PHOTO_KEYWORDS):
        flags.append("Photo/headshot referenced — omit to reduce appearance bias.")

    return flags


if __name__ == "__main__":  # pragma: no cover — smoke test
    sample = (
        "Jane Doe\nDOB: 1990-05-12\nGender: female\nMarital status: married\n"
        "123 Main Street\nB.S. 2012 Stanford\nAggressive self-confident driven leader."
    )
    for f in detect_bias_signals(sample):
        print("⚑", f)
