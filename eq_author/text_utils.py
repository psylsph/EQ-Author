"""Text processing utilities for chapter validation and manipulation."""

import re
import collections
from typing import Tuple


def count_words(text: str) -> int:
    """Return approximate word count suitable for prose validation."""
    if not text:
        return 0
    from .constants import WORD_RE_PATTERN
    return len(re.findall(WORD_RE_PATTERN, text))


def has_prologue(plan_text: str) -> bool:
    """Check if the plan contains a prologue section."""
    if not plan_text:
        return False
    lower = plan_text.lower()
    return "# prologue" in lower or "## prologue" in lower or "prologue:" in lower


def extract_chapter_ending(chapter_text: str, max_sentences: int = 2) -> str:
    """Extract last 1-2 sentences from a chapter to track ending."""
    sentences = re.split(r'(?<=[.!?])\s+', chapter_text.strip())
    if len(sentences) <= max_sentences:
        return chapter_text.strip()
    return ' '.join(sentences[-max_sentences:]).strip()


def validate_chapter_output(text: str) -> Tuple[bool, str]:
    """Validate chapter output for common issues.

    Returns (is_valid, issue_description).
    If is_valid is False, the chapter should be regenerated.
    """
    if not text or len(text) < 100:
        return False, "Chapter too short"

    stripped = text.strip()
    issues = []

    last_char = stripped[-1]
    # Allow standard and smart quotes
    allowed_punct = '.!?)"\'”’'
    if last_char not in allowed_punct:
        last_words = stripped.split()[-5:] if stripped.split() else []
        if last_words and not any(w[-1] in allowed_punct for w in last_words if len(w) > 1):
            issues.append("Chapter ends mid-sentence (no terminal punctuation)")

    words = stripped.split()
    if len(words) > 50:
        sentences = re.split(r'(?<=[.!?])\s+', stripped)
        sentence_stripped = [s.strip().lower() for s in sentences if s.strip()]
        if len(sentence_stripped) > 3:
            sentence_counts = collections.Counter(sentence_stripped)
            most_common = sentence_counts.most_common(1)
            if most_common and most_common[0][1] >= 3:
                issues.append(f"Repetitive sentence detected: '{most_common[0][0][:50]}...'")

    lines = [l.strip() for l in stripped.split('\n') if l.strip()]
    if len(lines) >= 5:
        line_counts = collections.Counter(lines)
        most_common = line_counts.most_common(1)
        if most_common and most_common[0][1] > 2:
            issues.append(f"Repetitive line detected: '{most_common[0][0][:50]}...'")

    if re.search(r'(.)\1{5,}', stripped):
        issues.append("Excessive character repetition detected")

    if issues:
        return False, "; ".join(issues)
    return True, ""
