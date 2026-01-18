"""Workflow for generating chapters using existing planning files."""

from pathlib import Path
from typing import Optional, Callable, List, Dict

from .api import make_client, PromptCache
from .context import ContextManager
from .progress import detect_progress, load_existing_context
from .file_io import write_chapter_output, collect_feedback
from .prompts import chapter_prompt
from .constants import DEFAULT_SUMMARY_LENGTH
from .text_utils import extract_chapter_ending
from .model_helpers import get_default_max_context_tokens
from .chapter import generate_chapter_summary
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


def run_workflow_v2_skip_planning(
    api_key: str,
    base_url: str,
    model: str,
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
) -> None:
    """Generate chapters using existing planning files (steps 1-5 must exist in out_dir).

    This function skips the planning workflow and uses existing planning files
    to generate chapters directly. Useful for when you've prepared planning
    files manually or want to regenerate chapters with different settings.
    """
    token_handler, emit_progress, maybe_collect_feedback = _setup_callbacks(stream, stream_callback, progress_callback)

    progress = detect_progress(out_dir)
    last_step = progress["last_step"]
    last_chapter = progress["last_chapter"]

    if last_step < 5:
        raise RuntimeError(
            f"Planning files incomplete. Found steps 1-{last_step}, but need all steps 1-5. "
            f"Use --resume-from with a directory containing complete planning files, "
            f"or run without --skip-planning to generate planning files first."
        )

    client = make_client(api_key, base_url)

    print(f"Using existing planning files from {out_dir}")
    print(f"Detected {last_chapter} existing chapters")

    messages = load_existing_context(out_dir, progress)
    print(f"Loaded {len(messages)} messages from planning files")

    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens,
    )

    context_manager.add_core_context(messages)

    # Check for prologue in Step 5 explicitly
    start_chapter = 1
    from .constants import STEP_FILENAMES
    step5_file = out_dir / STEP_FILENAMES.get(5, "05_final_plan.md")
    
    if step5_file.exists():
        try:
            step5_content = step5_file.read_text(encoding="utf-8")
            from .text_utils import has_prologue
            if has_prologue(step5_content):
                start_chapter = 0
                print("Prologue detected in plan (file check). Will generate Prologue as Chapter 0.", flush=True)
        except Exception as e:
            print(f"Warning: Could not read step 5 file for prologue check: {e}")
    else:
        # Fallback to checking loaded messages if file not found (unlikely if we passed checks above)
        from .text_utils import has_prologue
        # Try to find the step 5 message in loaded messages
        # messages contains steps 1-5 then chapters.
        # Step 5 should be at index 5 (index 0 is system prompt)
        if len(messages) >= 6:
             if has_prologue(messages[5].get("content", "")):
                 start_chapter = 0
                 print("Prologue detected in plan (message check). Will generate Prologue as Chapter 0.", flush=True)

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

    n_chapters = progress.get("n_chapters")
    if n_chapters is None:
        from .file_io import parse_proposed_chapters
        step4_file = out_dir / "04_final_plan.md"
        if step4_file.exists():
            step4_content = step4_file.read_text(encoding="utf-8")
            for line in step4_content.splitlines():
                if "chapter" in line.lower() and ":" in line:
                    parts = line.split(":")
                    try:
                        n_chapters = int(parts[-1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
        if n_chapters is None:
            raise RuntimeError("Could not determine chapter count from planning files. Ensure step 4 contains a chapter count (e.g., 'CHAPTER_COUNT: 10').")

    # If we are resuming from 0, we need to cover 0. 
    # Logic adjustment: last_chapter is what we found on disk.
    # If last_chapter is 0 (found chapter_00.md), we start at 1.
    # If last_chapter is 5 (found up to chapter_05.md), we start at 6.
    # If nothing found, last_chapter is usually 0. But if prologue needed, we must start at 0.
    
    # If we detected prologue and last_chapter is 0 (default "nothing found"), we start at 0.
    # If last_chapter > 0, we start at last_chapter + 1.
    
    next_chapter_index = last_chapter + 1
    if start_chapter == 0 and last_chapter == 0:
        # Check if chapter_00.md actually exists. detect_progress might not count 0 as a 'last_chapter' if logic is purely 1-based?
        # detect_progress implementation check needed. Assuming it finds max index.
        # If max index is 0, does it mean "found 0" or "found nothing"?
        # usually detect_progress returns 0 if no chapters found.
        # So we check if chapter_00 exists.
        if not (out_dir / "chapters" / "chapter_00.md").exists():
            next_chapter_index = 0

    print(f"Generating chapters {next_chapter_index} through {n_chapters}")

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

