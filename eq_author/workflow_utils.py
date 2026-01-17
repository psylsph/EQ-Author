"""Shared workflow utilities for chapter generation."""

import sys
from pathlib import Path
from typing import List, Optional, Callable

from .api import chat_once, PromptCache
from .context import ContextManager
from .prompts import chapter_prompt
from .text_utils import extract_chapter_ending, count_words, validate_chapter_output
from .chapter import generate_chapter_summary, auto_review_chapter
from .model_helpers import calculate_safe_max_tokens
from .constants import CHAPTER_MAX_ATTEMPTS, CHAPTER_MIN_WORDS, DEFAULT_SUMMARY_LENGTH


def _setup_callbacks(stream: bool, stream_callback: Optional[Callable[[str], None]] = None, progress_callback: Optional[Callable[[str], None]] = None):
    """Setup callback handlers for streaming and progress."""
    token_handler: Optional[Callable[[str], None]] = None
    if stream:
        if stream_callback is not None:
            token_handler = stream_callback
        else:
            token_handler = lambda t: print(t, end="", flush=True)

    def emit_progress(label: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(label)
        except Exception:
            pass

    def maybe_collect_feedback(label: str) -> Optional[str]:
        from .file_io import collect_feedback
        fb = collect_feedback(label)
        return fb if fb else None

    return token_handler, emit_progress, maybe_collect_feedback


def _generate_chapter_auto(
    client,
    model: str,
    chapter_num: int,
    context_manager: ContextManager,
    context_messages: List,
    stream: bool,
    temperature: Optional[float],
    cache: Optional[PromptCache],
    token_handler: Optional[Callable[[str], None]],
    emit_progress: Callable[[str], None],
) -> tuple[str, int]:
    """Generate a chapter in auto-gen mode (no retry loop)."""
    max_output_tokens = calculate_safe_max_tokens(
        context_manager.check_context_size(context_messages)["estimated_tokens"],
        context_manager.max_context_tokens,
        model,
    )
    final_text = chat_once(
        client, model, context_messages, stream=stream, temperature=temperature, on_token=token_handler if stream else None, cache=cache, max_tokens=max_output_tokens
    )
    if stream:
        print("\n", flush=True)
    last_word_count = count_words(final_text)
    print(f"Chapter {chapter_num} word count: {last_word_count}", flush=True)
    return final_text, last_word_count


def _generate_chapter_interactive(
    client,
    model: str,
    chapter_num: int,
    context_manager: ContextManager,
    context_messages: List,
    stream: bool,
    temperature: Optional[float],
    cache: Optional[PromptCache],
    token_handler: Optional[Callable[[str], None]],
    emit_progress: Callable[[str], None],
    maybe_collect_feedback: Optional[Callable[[str], str]],
) -> tuple[str, int]:
    """Generate a chapter in interactive mode (with retry loop)."""
    attempts = 0
    final_text = ""
    last_response = ""
    last_word_count = 0

    while attempts < CHAPTER_MAX_ATTEMPTS:
        max_output_tokens = calculate_safe_max_tokens(
            context_manager.check_context_size(context_messages)["estimated_tokens"],
            context_manager.max_context_tokens,
            model,
        )
        ch = chat_once(
            client,
            model,
            context_messages,
            stream=stream,
            temperature=temperature,
            on_token=token_handler if stream else None,
            cache=cache,
            max_tokens=max_output_tokens,
        )
        if stream:
            print("\n", flush=True)

        last_response = ch
        last_word_count = count_words(ch)

        if last_word_count >= CHAPTER_MIN_WORDS:
            is_valid, issue = validate_chapter_output(ch)
            if not is_valid:
                if stream:
                    print(f"Chapter {chapter_num} invalid output: {issue}. Retrying...", flush=True)
                attempts += 1
                if attempts >= CHAPTER_MAX_ATTEMPTS:
                    final_text = ch
                    print(f"Warning: Chapter {chapter_num} has {issue} after {CHAPTER_MAX_ATTEMPTS} attempts, using anyway.", file=sys.stderr, flush=True)
                    break
                retry_prompt = (
                    f"The chapter you just provided has issues: {issue}. "
                    "Please rewrite entire chapter, ensuring:\n"
                    "- It ends with proper punctuation (complete sentences)\n"
                    "- No repetitive phrases or sentences\n"
                    "- Natural, flowing narrative\n"
                    "Output only the revised chapter text with no commentary."
                )
                context_messages.append({"role": "user", "content": retry_prompt})
                continue
            final_text = ch
            break

        attempts += 1
        if stream:
            print(f"Chapter {chapter_num} returned {last_word_count} words (needs >= {CHAPTER_MIN_WORDS}).", flush=True)

        if attempts >= CHAPTER_MAX_ATTEMPTS:
            break

        retry_prompt = (
            f"The chapter you just provided for Chapter {chapter_num} contains {last_word_count} words, "
            f"but it must be at least {CHAPTER_MIN_WORDS} words of narrative prose. "
            "Please rewrite the entire chapter, keeping continuity with earlier chapters, and expand the storytelling so that the final output meets or exceeds the requirement. "
            "Output only the revised chapter text with no commentary."
        )
        context_messages.append({"role": "user", "content": retry_prompt})

    if not final_text:
        final_text = last_response

    if last_word_count < CHAPTER_MIN_WORDS:
        print(f"Warning: Chapter {chapter_num} final word count {last_word_count} < {CHAPTER_MIN_WORDS} after {CHAPTER_MAX_ATTEMPTS} attempt(s).", file=sys.stderr, flush=True)
    else:
        print(f"Chapter {chapter_num} word count: {last_word_count}", flush=True)

    return final_text, last_word_count


def _auto_review_chapter(
    client,
    model: str,
    chapter_num: int,
    final_text: str,
    context_manager: ContextManager,
    out_dir: Path,
    stream: bool,
    temperature: Optional[float],
    cache: Optional[PromptCache],
) -> tuple[str, int]:
    """Auto-review a chapter and potentially rewrite."""
    print(f"\n=== Auto-Reviewing Chapter {chapter_num} ===", flush=True)
    review_count = 0
    max_auto_reviews = 2
    llm_approved = False
    last_word_count = count_words(final_text)

    while review_count < max_auto_reviews and not llm_approved:
        previous_ending = context_manager.get_previous_chapter_ending(chapter_num) if chapter_num > 1 else ""
        review_notes, reviewed_text, llm_approved = auto_review_chapter(
            client, model, final_text, chapter_num, previous_ending, stream, temperature, cache
        )

        from .file_io import write_chapter_output
        review_file = out_dir / "chapters" / f"chapter_{chapter_num:02d}_review.md"
        (out_dir / "chapters").mkdir(parents=True, exist_ok=True)
        review_file.write_text(review_notes, encoding="utf-8")

        if reviewed_text != final_text:
            print(f"Chapter {chapter_num} auto-rewritten based on review.", flush=True)
            final_text = reviewed_text
            last_word_count = count_words(final_text)
            print(f"New word count: {last_word_count}", flush=True)
            review_count += 1
        else:
            print(f"Chapter {chapter_num} review: No rewrite needed.", flush=True)
            break

    return final_text, last_word_count


def _apply_feedback_rewrite(
    client,
    model: str,
    chapter_num: int,
    context_manager: ContextManager,
    final_text: str,
    feedback: str,
    stream: bool,
    temperature: Optional[float],
    cache: Optional[PromptCache],
    token_handler: Optional[Callable[[str], None]],
    emit_progress: Callable[[str], None]],
) -> tuple[str, int]:
    """Apply user feedback and rewrite chapter."""
    feedback_prompt = (
        f"Please rewrite Chapter {chapter_num} incorporating this feedback:\n\n"
        f"FEEDBACK: {feedback}\n\n"
        f"Previous chapter text:\n{final_text[:2000]}...\n\n"
        "Rewrite the entire chapter with the feedback applied. "
        "Maintain continuity with previous chapters. "
        "Output only the revised chapter text with no commentary."
    )
    rewrite_messages = context_manager.build_context(chapter_num).copy()
    rewrite_messages.append({"role": "user", "content": chapter_prompt(chapter_num, context_manager.get_previous_chapter_ending(chapter_num))})
    rewrite_messages.append({"role": "user", "content": feedback_prompt})

    max_output_tokens = calculate_safe_max_tokens(
        context_manager.check_context_size(rewrite_messages)["estimated_tokens"],
        context_manager.max_context_tokens,
        model,
    )
    ch = chat_once(
        client,
        model,
        rewrite_messages,
        stream=stream,
        temperature=temperature,
        on_token=token_handler if stream else None,
        cache=cache,
        max_tokens=max_output_tokens,
    )
    if stream:
        print("\n", flush=True)

    new_word_count = count_words(ch)
    if new_word_count >= CHAPTER_MIN_WORDS:
        final_text = ch
        last_word_count = new_word_count
        print(f"Chapter {chapter_num} rewritten. Word count: {last_word_count}", flush=True)
    else:
        print(f"Warning: Rewritten chapter has {new_word_count} words (needs >= {CHAPTER_MIN_WORDS})", flush=True)

    return final_text, last_word_count
