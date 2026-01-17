"""Chapter generation and review functions."""

import re
from typing import Tuple
from .constants import DEFAULT_SUMMARY_LENGTH, CHAPTER_MIN_WORDS
from .api import chat_once
from .text_utils import count_words


def generate_chapter_summary(
    client,
    model: str,
    chapter_text: str,
    chapter_num: int,
    target_length: int = DEFAULT_SUMMARY_LENGTH,
    stream: bool = False,
    temperature = None,
    cache = None,
) -> str:
    """Generate a concise summary of a chapter for context management."""

    summary_prompt = (
        f"Create a concise summary of Chapter {chapter_num} (approximately {target_length} words) that captures:\n"
        f"- Key plot developments and events\n"
        f"- Character decisions and changes\n"
        f"- Important revelations or discoveries\n"
        f"- Emotional arcs and relationship dynamics\n"
        f"- Any cliffhangers or setup for future events\n\n"
        f"Chapter text:\n{chapter_text}\n\n"
        "Provide only the summary without commentary or word count mentions."
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that creates concise chapter summaries."},
        {"role": "user", "content": summary_prompt},
    ]

    summary = chat_once(client, model, messages, stream=stream, temperature=temperature, cache=cache)
    return summary.strip()


def auto_review_chapter(
    client,
    model: str,
    chapter_text: str,
    chapter_num: int,
    previous_chapter_ending: str = "",
    stream: bool = False,
    temperature = None,
    cache = None,
) -> Tuple[str, str, bool]:
    """Have the LLM review and optionally rewrite a chapter.

    Returns a tuple of (review_notes, rewritten_text, llm_approved).
    - llm_approved: True if LLM says the chapter is acceptable (final authority)

    The LLM's judgment overrides the word count check. If the LLM says
    the chapter is good quality (even if under 2500 words), it will be approved.
    """
    word_count = count_words(chapter_text)

    review_prompt = f"""You are an expert book editor reviewing Chapter {chapter_num} of a novel.

The chapter has {word_count} words (target: {CHAPTER_MIN_WORDS}+, but quality matters more than length).

Please review thoroughly and determine if the chapter is acceptable as-is or needs improvement.

{('Previous chapter ended: ' + previous_chapter_ending) if previous_chapter_ending else 'No previous chapter context (this is chapter 1).'}

Evaluate:
1. **Narrative Flow**: Is the story engaging and well-paced?
2. **Character Consistency**: Are characters consistent and believable?
3. **Dialogue Quality**: Is dialogue natural and purposeful?
4. **Scene Description**: Are scenes vivid and immersive?
5. **Pacing**: Does the chapter maintain good momentum?
6. **Continuity**: Does it flow naturally from the previous chapter?
7. **Completeness**: Does it accomplish its narrative purpose?

IMPORTANT: A chapter under {CHAPTER_MIN_WORDS} words can still be APPROVED if:
- It tells a complete, satisfying story segment
- Quality of prose is excellent
- It serves its narrative purpose well
- It's not unnecessarily short

OUTPUT YOUR REVIEW IN THIS FORMAT:
```
## Quality Assessment
- Word Count: X (target: {CHAPTER_MIN_WORDS}+)
- Narrative Flow: X/5
- Character Consistency: X/5
- Dialogue Quality: X/5
- Scene Description: X/5
- Pacing: X/5
- Overall: X/5

## Verdict
ACCEPTABLE (chapter meets quality standards) - No rewrite needed
OR
NEEDS_IMPROVEMENT (specific issues listed below)

## Review Notes
[Your detailed review of what's working, what could be better, and why you made your verdict]

## Rewrite Recommendation
ACCEPTABLE or NEEDS_IMPROVEMENT

[If NEEDS_IMPROVEMENT: Provide the rewritten chapter below, keeping continuity and improving the identified issues]
[If ACCEPTABLE: Just output "ACCEPTABLE - Chapter approved as-is" for the chapter text]
```

Remember: Your judgment is final. Approve good chapters even if they're shorter than the target. Rewrite only if there are significant quality issues.

Chapter to review:
{chapter_text}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a professional book editor. Your judgment is final - approve good chapters even if they're shorter than the target. Only recommend rewrite for significant quality issues.",
        },
        {"role": "user", "content": review_prompt},
    ]

    review = chat_once(client, model, messages, stream=stream, temperature=temperature, cache=cache)

    upper_review = review.upper()
    is_acceptable = (
        "VERDICT" in upper_review
        and "ACCEPTABLE" in upper_review
        and "NEEDS_IMPROVEMENT" not in upper_review[: upper_review.find("VERDICT") + 50]
    ) or "ACCEPTABLE - CHAPTER APPROVED" in upper_review
    is_needs_improvement = (
        "NEEDS_IMPROVEMENT" in upper_review
        and "ACCEPTABLE" not in upper_review[: upper_review.find("NEEDS_IMPROVEMENT") + 50]
    )

    if "RECOMMENDATION" in upper_review:
        recap_start = upper_review.find("RECOMMENDATION")
        recap_section = upper_review[recap_start:recap_start + 200]
        is_acceptable = is_acceptable or "ACCEPTABLE" in recap_section
        is_needs_improvement = is_needs_improvement or "NEEDS_IMPROVEMENT" in recap_section

    if is_acceptable and not is_needs_improvement:
        return review, chapter_text, True

    if is_needs_improvement:
        lines = review.split("\n")
        in_rewrite_section = False
        rewritten_parts = []
        for line in lines:
            line_upper = line.upper()
            if "ACCEPTABLE" in line_upper and "CHAPTER APPROVED" in line_upper:
                break
            if "RECOMMENDATION" in line_upper:
                if "ACCEPTABLE" in line_upper:
                    break
                else:
                    in_rewrite_section = True
                continue
            if in_rewrite_section and line.strip() and not line.strip().startswith("##"):
                rewritten_parts.append(line)
        rewritten_text = "\n".join(rewritten_parts).strip()
        if not rewritten_text or len(rewritten_text) < 100:
            rewritten_text = chapter_text
        return review, rewritten_text, False

    return review, chapter_text, True
