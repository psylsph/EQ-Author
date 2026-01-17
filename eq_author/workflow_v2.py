"""Main full workflow function (steps 1-5 + chapter generation)."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from .api import make_client, chat_once, PromptCache
from .context import ContextManager
from .prompts import build_step1_prompt, build_followup_prompts, chapter_prompt
from .file_io import write_step_output, write_chapter_output, load_prompt_override_file, is_adult_target, parse_proposed_chapters
from .chapter import generate_chapter_summary, auto_review_chapter
from .text_utils import extract_chapter_ending, count_words, validate_chapter_output
from .model_helpers import calculate_safe_max_tokens, get_default_max_context_tokens, get_default_temperature
from .constants import DEFAULT_SUMMARY_LENGTH, CHAPTER_MAX_ATTEMPTS, CHAPTER_MIN_WORDS, DEFAULT_CONTEXT_STRATEGY, DEFAULT_RECENT_CHAPTERS
from .workflow_utils import (
    _setup_callbacks,
    _generate_chapter_auto,
    _generate_chapter_interactive,
    _auto_review_chapter,
    _apply_feedback_rewrite,
)


def run_workflow_v2(
    api_key: str,
    base_url: str,
    model: str,
    story_prompt: str,
    n_chapters: Optional[int],
    interactive: bool,
    out_dir: Path,
    stream: bool = False,
    temperature: Optional[float] = None,
    cache: Optional[PromptCache] = None,
    feedback_callback: Optional[Callable[[str], str]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    context_strategy: str = DEFAULT_CONTEXT_STRATEGY,
    summary_length: int = DEFAULT_SUMMARY_LENGTH,
    recent_chapters: int = DEFAULT_RECENT_CHAPTERS,
    max_context_tokens: Optional[int] = None,
    always_autogen_chapters: bool = False,
) -> None:
    """Run full workflow: planning steps 1-5 + chapter generation."""
    client = make_client(api_key, base_url)

    token_handler, emit_progress, maybe_collect_feedback = _setup_callbacks(stream, stream_callback, progress_callback)

    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens,
    )

    messages: List[Dict[str, str]] = []
    try:
        override = ""
        if is_adult_target(story_prompt):
            override = load_prompt_override_file() or ""

        if override:
            messages.append({"role": "system", "content": override})

            try:
                print("[PROMPT OVERRIDE] Applied prompt_override.txt (adult target detected)", flush=True)
                try:
                    (out_dir / "prompt_override_used.txt").write_text(override, encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                pass
        else:
            try:
                print("[PROMPT OVERRIDE] Detected adult target but `prompt_override.txt` is missing or empty", flush=True)
                try:
                    (out_dir / "prompt_override_missing.txt").write_text("(no override file present)", encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    messages.append({"role": "system", "content": "You are a helpful assistant"})

    step1 = build_step1_prompt().replace("{STORY_PROMPT}", story_prompt)
    messages.append({"role": "user", "content": step1})
    if stream:
        print("\n=== Step 1: Brainstorm & Reflection (streaming) ===", flush=True)
    emit_progress("Step 1: Brainstorm & Reflection")
    s1 = chat_once(client, model, messages, stream=stream, temperature=temperature, on_token=token_handler if stream else None, cache=cache)
    if stream:
        print("\n", flush=True)
    write_step_output(out_dir, 1, s1)
    messages.append({"role": "assistant", "content": s1})

    proposed = parse_proposed_chapters(s1)
    final_count: Optional[int] = n_chapters if n_chapters is not None else proposed

    if interactive:
        print(f"Model proposed chapter count: {proposed if proposed is not None else 'N/A'}")
        default_display = final_count if final_count is not None else "<required>"
        user_in = input(f"Enter chapter count to use (Enter = {default_display}): ").strip()
        if user_in:
            try:
                final_count = max(1, int(user_in))
            except ValueError:
                print("Invalid number; keeping previous value.")

    if final_count is None:
        raise RuntimeError("No chapter count available. Rerun interactively or pass --n-chapters.")

    fb = maybe_collect_feedback(1)
    if fb:
        messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 1:\n{fb}\nPlease apply this feedback in the next steps."})

    while True:
        fb = maybe_collect_feedback(1)
        if fb:
            messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 1:\n{fb}\nPlease apply this feedback and rewrite your output for Brainstorming & Reflection."})
            if stream:
                print("\n=== Re-applying feedback for Step 1: Brainstorming & Reflection (streaming) ===", flush=True)
            s1 = chat_once(client, model, messages, stream=stream, temperature=temperature, on_token=token_handler if stream else None, cache=cache)
            if stream:
                print("\n", flush=True)
            write_step_output(out_dir, 1, s1)
            messages[-1] = {"role": "assistant", "content": s1}
        else:
            break

    step_texts = build_followup_prompts(final_count)

    for step_num in range(2, 6):
        step_label = {2: "Intention & Chapter Planning", 3: "Human vs LLM Critique", 4: "Final Plan", 5: "Characters"}[step_num]
        messages.append({"role": "user", "content": step_texts[step_num - 2]})
        if stream:
            print(f"\n=== Step {step_num}: {step_label} (streaming) ===", flush=True)
        emit_progress(f"Step {step_num}: {step_label}")
        s_out = chat_once(client, model, messages, stream=stream, temperature=temperature, on_token=token_handler if stream else None, cache=cache)
        if stream:
            print("\n", flush=True)
        write_step_output(out_dir, step_num, s_out)
        messages.append({"role": "assistant", "content": s_out})

        while True:
            fb = maybe_collect_feedback(step_num)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP {step_num}:\n{fb}\nPlease apply this feedback and rewrite your output for {step_label}."})
                if stream:
                    print(f"\n=== Re-applying feedback for Step {step_num}: {step_label} (streaming) ===", flush=True)
                s_out = chat_once(client, model, messages, stream=stream, temperature=temperature, on_token=token_handler if stream else None, cache=cache)
                if stream:
                    print("\n", flush=True)
                write_step_output(out_dir, step_num, s_out)
                messages[-1] = {"role": "assistant", "content": s_out}
            else:
                break

    context_manager.add_core_context(messages)

    for i in range(1, final_count + 1):
        previous_chapter_ending = context_manager.get_previous_chapter_ending(i) if i > 1 else ""
        context_messages = context_manager.build_context(i)
        context_messages.append({"role": "user", "content": chapter_prompt(i, previous_chapter_ending)})

        context_status = context_manager.check_context_size(context_messages)
        if context_status["is_warning"]:
            print(f"Warning: Context usage at {context_status['usage_ratio']:.1%} ({context_status['estimated_tokens']}/{context_status['max_tokens']} tokens)", flush=True)
        if context_status["is_critical"]:
            print(f"CRITICAL: Context usage at {context_status['usage_ratio']:.1%} - consider reducing context size", flush=True)

        if stream:
            print(f"\n=== Chapter {i} (streaming) ===", flush=True)
        emit_progress(f"Chapter {i}")

        if always_autogen_chapters:
            final_text, last_word_count = _generate_chapter_auto(
                client, model, i, context_manager, context_messages, stream, temperature, cache, token_handler, emit_progress
            )
        else:
            final_text, last_word_count = _generate_chapter_interactive(
                client, model, i, context_manager, context_messages, stream, temperature, cache, token_handler, emit_progress, maybe_collect_feedback
            )

        if not always_autogen_chapters:
            feedback_loop_count = 0
            max_feedback_loops = 3
            while feedback_loop_count < max_feedback_loops:
                fb = maybe_collect_feedback(f"chapter {i}")
                if not fb:
                    break

                final_text, last_word_count = _apply_feedback_rewrite(
                    client, model, i, context_manager, final_text, fb, stream, temperature, cache, token_handler, emit_progress
                )
                feedback_loop_count += 1

        if always_autogen_chapters:
            final_text, last_word_count = _auto_review_chapter(
                client, model, i, final_text, context_manager, out_dir, stream, temperature, cache
            )

        write_chapter_output(out_dir, i, final_text)

        chapter_ending = extract_chapter_ending(final_text)

        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(client, model, final_text, i, summary_length, stream, temperature, cache)

        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")

        context_manager.add_chapter(i, final_text, chapter_summary, chapter_ending)

        if not always_autogen_chapters:
            fb = maybe_collect_feedback(f"chapter {i}")
            if fb:
                context_messages.append({"role": "user", "content": f"FEEDBACK AFTER CHAPTER {i}:\n{fb}\nPlease apply this feedback in the next chapter(s)."})
