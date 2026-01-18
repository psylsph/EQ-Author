"""Workflow for resuming from a previous run."""

from pathlib import Path
from typing import Optional, Callable, List, Dict

from .api import make_client, PromptCache, chat_once
from .context import ContextManager
from .progress import detect_progress, load_existing_context
from .file_io import write_chapter_output
from .prompts import chapter_prompt
from .constants import DEFAULT_SUMMARY_LENGTH
from .chapter import generate_chapter_summary
from .text_utils import extract_chapter_ending
from .model_helpers import get_default_max_context_tokens
from .workflow_utils import (
    _setup_callbacks,
    _generate_chapter_auto,
    _generate_chapter_interactive,
    _auto_review_chapter,
    _apply_feedback_rewrite,
)


def load_chapter_context(out_dir: Path) -> List[Dict[str, str]]:
    """Load only steps 2 (characters) and 5 (final plan) for chapter context."""
    from .constants import STEP_FILENAMES
    
    context = []
    context.append({"role": "system", "content": "You are a helpful assistant"})
    
    # Load step 2: Characters
    char_file = out_dir / STEP_FILENAMES[2]
    if char_file.exists():
        char_content = char_file.read_text(encoding="utf-8")
        context.append({"role": "assistant", "content": char_content})
    
    # Load step 5: Final Plan  
    plan_file = out_dir / STEP_FILENAMES[5]
    if plan_file.exists():
        plan_content = plan_file.read_text(encoding="utf-8")
        context.append({"role": "assistant", "content": plan_content})
    
    return context


def run_workflow_v2_resume(
    api_key: str,
    base_url: str,
    model: str,
    story_prompt: str,
    out_dir: Path,
    stream: bool = False,
    temperature: Optional[float] = None,
    cache: Optional[PromptCache] = None,
    feedback_callback: Optional[Callable[[str], str]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    context_strategy: str = "aggressive",
    summary_length: int = DEFAULT_SUMMARY_LENGTH,
    recent_chapters: int = 2,
    max_context_tokens: Optional[int] = None,
    always_autogen_chapters: bool = False,
    n_chapters: Optional[int] = None,
) -> None:
    """Resume a previous workflow run from where it left off."""
    client = make_client(api_key, base_url)
    token_handler, emit_progress, maybe_collect_feedback = _setup_callbacks(stream, stream_callback, progress_callback)

    progress = detect_progress(out_dir)
    last_step = progress["last_step"]
    last_chapter = progress["last_chapter"]

    print(f"Detected progress: Steps 1-{last_step} complete, Chapters 1-{last_chapter} complete")

    if n_chapters is not None:
        print(f"Using provided chapter count: {n_chapters}")
        detected_n_chapters = n_chapters
    elif progress.get("n_chapters") is not None:
        detected_n_chapters = progress["n_chapters"]
        print(f"Detected chapter count from previous run: {detected_n_chapters}")
    else:
        print("Warning: Could not determine total chapter count from previous run")
        print("Using default chapter count: 12 (use --n-chapters to override)")
        detected_n_chapters = 12

    n_chapters = int(detected_n_chapters)

    messages = load_existing_context(out_dir, progress)
    print(f"Loaded {len(messages)} messages from previous run")

    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens,
    )

    context_manager.add_core_context(messages)

    # Check for prologue in Step 5 (messages is loaded from steps 1-5 + chapters)
    # The messages list contains system, steps 1-5, then chapters.
    # Step 5 is at index 5 (system is 0, steps 1-5 are 1-5).
    # But load_existing_context might vary if steps are missing (unlikely in resume).
    # Safer to check file directly.
    start_chapter = 1
    from .constants import STEP_FILENAMES
    step5_file = out_dir / STEP_FILENAMES.get(5, "05_final_plan.md")
    
    if step5_file.exists():
        try:
            step5_content = step5_file.read_text(encoding="utf-8")
            from .text_utils import has_prologue
            if has_prologue(step5_content):
                start_chapter = 0
                print("Prologue detected in plan. Will generate Prologue as Chapter 0.", flush=True)
        except Exception:
            pass
    elif len(messages) >= 6:
        # Fallback to messages
        from .text_utils import has_prologue
        if has_prologue(messages[5].get("content", "")):
             start_chapter = 0
             print("Prologue detected in plan (context). Will generate Prologue as Chapter 0.", flush=True)

    if last_chapter > 0:
        for i in range(start_chapter, last_chapter + 1):
            if i < 1 and start_chapter > i: continue # Just safety
            chapter_file = out_dir / "chapters" / f"chapter_{i:02d}.md"
            summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"

            if chapter_file.exists():
                try:
                    chapter_text = chapter_file.read_text(encoding="utf-8")
                    chapter_ending = extract_chapter_ending(chapter_text)

                    if summary_file.exists():
                        chapter_summary = summary_file.read_text(encoding="utf-8")
                    else:
                        chapter_summary = chapter_text

                    context_manager.add_chapter(i, chapter_text, chapter_summary, chapter_ending)
                except Exception as e:
                    print(f"Warning: Failed to load chapter {i}: {e}")

    for step_num in range(1, 6):
        if last_step < step_num:
            from .file_io import collect_feedback
            fb = maybe_collect_feedback(step_num)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP {step_num}:\n{fb}\nPlease apply this feedback and rewrite your output for Brainstorming & Reflection."})
                if stream:
                    print(f"\n=== Re-applying feedback for Step {step_num}: Brainstorming & Reflection (streaming) ===", flush=True)
                s_out = chat_once(client, model, messages, stream=stream, temperature=temperature, on_token=token_handler if stream else None, cache=cache)
                if stream:
                    print("\n", flush=True)
                from .file_io import write_step_output
                write_step_output(out_dir, step_num, s_out)
                messages[-1] = {"role": "assistant", "content": s_out}
        else:
            break

    context_manager.add_core_context(messages)

    # Determine next chapter index
    # Resume logic:
    # If last_chapter is 0 (found nothing or found chapter_00), next is last_chapter + 1?
    # Wait, if detect_progress found chapter_00.md, last_chapter would be 0?
    # detect_progress logic: int(f.name[8:10]). So chapter_00 -> 0.
    # If chapters_dir has only chapter_00.md, max_chapter = 0.
    # If chapters_dir is empty, max_chapter = 0.
    
    # If max_chapter is 0, we need to check if chapter_00.md exists to distinguish "done prologue" from "done nothing".
    next_chapter_index = last_chapter + 1
    if start_chapter == 0 and last_chapter == 0:
        if not (out_dir / "chapters" / "chapter_00.md").exists():
            next_chapter_index = 0
            
    print(f"Resuming generation from Chapter {next_chapter_index}")

    for i in range(next_chapter_index, n_chapters + 1):
        previous_chapter_ending = context_manager.get_previous_chapter_ending(i) if i > start_chapter else ""
        # Load only characters (step 2) and final plan (step 5) for chapter context
        chapter_base_context = load_chapter_context(out_dir)
        # Add chapter summaries from previous chapters
        chapter_context = chapter_base_context.copy()
        if i > start_chapter:
            # Add summaries from all previous chapters
            for summary_info in context_manager.chapter_summaries:
                chapter_context.append({
                    "role": "assistant",
                    "content": f"Chapter {summary_info['chapter']} Summary:\n{summary_info['summary']}",
                })
        
        context_messages = chapter_context
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

        if always_autogen_chapters:
            final_text, last_word_count = _auto_review_chapter(
                client, model, i, final_text, context_manager, out_dir, stream, temperature, cache, token_handler
            )
        else:
            feedback_loop_count = 0
            max_feedback_loops = 3
            while feedback_loop_count < max_feedback_loops:
                fb = maybe_collect_feedback(f"chapter {i}")
                if not fb:
                    # User skipped manual feedback, apply auto-review
                    print(f"Manual feedback skipped. Running auto-review for Chapter {i}...", flush=True)
                    final_text, last_word_count = _auto_review_chapter(
                        client, model, i, final_text, context_manager, out_dir, stream, temperature, cache, token_handler
                    )
                    break

                # Manual feedback provided
                final_text, last_word_count = _apply_feedback_rewrite(
                    client, model, i, context_manager, final_text, fb, stream, temperature, cache, token_handler, emit_progress
                )
                feedback_loop_count += 1

        write_chapter_output(out_dir, i, final_text)

        chapter_ending = extract_chapter_ending(final_text)

        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(client, model, final_text, i, summary_length, stream, temperature, cache, token_handler)

        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")

        context_manager.add_chapter(i, final_text, chapter_summary, chapter_ending)


