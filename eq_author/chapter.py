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
    on_token = None,
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

    summary = chat_once(client, model, messages, stream=stream, temperature=temperature, cache=cache, on_token=on_token)
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
    on_token = None,
) -> Tuple[str, str, bool]:
    """Have the LLM review and optionally rewrite a chapter.

    Returns a tuple of (review_notes, rewritten_text, llm_approved).
    - llm_approved: True if LLM says the chapter is acceptable (final authority)

    The LLM's judgment overrides the word count check. If the LLM says
    the chapter is good quality (even if under 2000 words), it will be approved.
    """
    word_count = count_words(chapter_text)

    review_prompt = f"""You are a ruthlessly critical book editor reviewing Chapter {chapter_num} of a novel.
You have high standards for literary fiction and thriller pacing. You despise "AI-slop" (repetitive phrasing, bland descriptions, lack of subtext).

The chapter has {word_count} words (target: {CHAPTER_MIN_WORDS}+).

Please review thoroughly. Be harsh. Look for:
1. **Show, Don't Tell**: Does the author explain emotions instead of showing them?
2. **Dialogue**: Is it on-the-nose? Do characters say exactly what they mean (bad) or is there subtext (good)?
3. **Pacing**: Does the scene drag? Is there unnecessary exposition?
4. **Repetition**: Are words or sentence structures repeated? (e.g., starting every sentence with "He" or "She")
5. **Logic**: Do character actions make sense?

{('Previous chapter ended: ' + previous_chapter_ending) if previous_chapter_ending else 'No previous chapter context (this is chapter 1).'}

OUTPUT YOUR REVIEW IN THIS FORMAT:
```
## Critical Assessment
- Word Count: X (target: {CHAPTER_MIN_WORDS}+)
- Narrative Flow: X/5
- Dialogue Authenticity: X/5
- "Show, Don't Tell": X/5
- Pacing & Tension: X/5
- Prose Quality (avoiding repetition): X/5
- Overall: X/5

## Verdict
ACCEPTABLE (Only if it's high quality and needs no major changes)
OR
NEEDS_IMPROVEMENT (If any of the above scores are below 4/5, or if significant issues exist)

## Critical Notes
[Bulleted list of SPECIFIC flaws. Quote specific sentences that are bad. Be direct.]

## Rewrite Recommendation
[If NEEDS_IMPROVEMENT: rewrite the ENTIRE chapter below.
CRITICAL INSTRUCTIONS FOR REWRITE:
1. **RETAIN LENGTH**: You must output a full chapter of similar length ({word_count} words or more). DO NOT SUMMARIZE.
2. **RETAIN PLOT**: Keep all plot points and events. Do not change the story, only the execution.
3. **FIX STYLE**: Apply your critical notes (fix dialogue, show don't tell, vary sentence structure).
4. **OUTPUT FULL TEXT**: Provide the complete, polished text ready for publication.]

[If ACCEPTABLE: Just output "ACCEPTABLE - Chapter approved as-is"]
```

Chapter Text to Review:
{chapter_text}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a critical, high-standards book editor. You rarely approve a first draft. You hate cliche and repetitive AI writing styles.",
        },
        {"role": "user", "content": review_prompt},
    ]

    review = chat_once(client, model, messages, stream=stream, temperature=temperature, cache=cache, on_token=on_token)

    upper_review = review.upper()
    is_acceptable = (
        "VERDICT" in upper_review
        and "ACCEPTABLE" in upper_review
        and "NEEDS_IMPROVEMENT" not in upper_review[upper_review.find("VERDICT"):upper_review.find("VERDICT") + 100]
    ) or "ACCEPTABLE - CHAPTER APPROVED" in upper_review

    # Check specifically for NEEDS_IMPROVEMENT in the Verdict section
    verdict_pos = upper_review.find("VERDICT")
    is_needs_improvement = (
        "NEEDS_IMPROVEMENT" in upper_review[verdict_pos:]
    )

    if "REWRITE RECOMMENDATION" in upper_review:
        recap_start = upper_review.find("REWRITE RECOMMENDATION")
        recap_section = upper_review[recap_start:recap_start + 200]
        # If recommendation section mentions acceptable, it overrides?
        # But if verdict said needs improvement, we should respect that.
        pass

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
            if "REWRITE RECOMMENDATION" in line_upper:
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
        
        # Safety check: if rewrite is significantly shorter (< 70% of original), discard it
        new_wc = count_words(rewritten_text)
        original_wc = count_words(chapter_text)
        if original_wc > 500 and new_wc < (original_wc * 0.7):
            print(f"Warning: Auto-rewrite was too short ({new_wc} words vs {original_wc}). Discarding rewrite.", flush=True)
            return review + "\n\n[NOTE: Rewrite discarded because it was too short]", chapter_text, False

        return review, rewritten_text, False

    return review, chapter_text, True
