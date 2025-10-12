"""Streamlit UI for EQ Author workflow."""
from __future__ import annotations

import contextlib
import html
import json
import hashlib
import io
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import streamlit as st
from streamlit.components.v1 import html as components_html
from eq_author import (
    PromptCache, ensure_output_dir, make_client, chat_once,
    build_step1_prompt, build_followup_prompts, chapter_prompt,
    write_step_output, write_chapter_output, count_words,
    CHAPTER_MIN_WORDS, CHAPTER_MAX_ATTEMPTS
)
from publish_to_pdf import process_story_to_pdf


load_dotenv()

def parse_proposed_chapters(text: str) -> Optional[int]:
    """Extract the suggested chapter count from the LLM's planning response."""
    if not text:
        return None
    # Look for direct chapter count statements
    patterns = [
        r"(?i)plan for (\d+) chapters",
        r"(?i)(\d+) chapters? would (?:be|work)",
        r"(?i)(\d+) chapters? to tell",
        r"(?i)story into (\d+) chapters",
        r"(?i)recommend (\d+) chapters",
        r"(?i)suggest (\d+) chapters",
        r"(?i)divid\w+ into (\d+) chapters",
        r"(?i)(\d+) chapters? total",
    ]
    for pattern in patterns:
        if match := re.search(pattern, text):
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None

def main():
    global should_run_stage
    should_run_stage = False
    
    st.set_page_config(
        page_title="EQ Author",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

CHAPTERS_SUBDIR = "chapters"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CACHE_DIR = ".prompt_cache"
STEP_LABELS = {
    0: "Step 1: Brainstorm & Reflection",
    1: "Step 2: Intention & Chapter Planning",
    2: "Step 3: Human vs LLM Critique",
    3: "Step 4: Final Plan",
    4: "Step 5: Characters",
}

FEEDBACK_TEXTAREA_STYLE = """
<style>
textarea[aria-label="Feedback to apply before the next step"]:not(:placeholder-shown),
textarea[aria-label="Feedback to apply before the next chapter"]:not(:placeholder-shown),
textarea[aria-label="General feedback for upcoming chapters"]:not(:placeholder-shown) {
    background-color: #ffe6e6 !important;
    border-color: #ff4d6a !important;
}
</style>
"""

STREAM_BOX_STYLE = """
<style>
.stream-box {
    min-height: 50px;
    overflow-y: auto;
    padding: 0.75rem;
    border: 1px solid var(--divider-color, rgba(120, 120, 120, 0.3));
    border-radius: 0.75rem;
    background-color: var(--secondary-background-color, rgba(245, 245, 245, 0.9));
    color: var(--text-color, #111);
    font-family: var(--font-mono, "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace);
    font-size: 0.9rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.stream-box :is(p, pre, code, ul, ol) {
    margin-top: 0;
    color: inherit;
}
.stream-box pre {
    background: transparent;
}
.stream-box code {
    background-color: rgba(120, 120, 120, 0.12);
    padding: 0.1rem 0.3rem;
    border-radius: 0.35rem;
}
.dark .stream-box {
    background-color: rgba(15, 17, 23, 0.85);
    border-color: rgba(250, 250, 250, 0.2);
    color: rgba(255, 255, 255, 0.92);
}
</style>
"""



def fetch_available_models(api_key: str, base_url: str) -> List[str]:
    """Fetch model IDs from the DeepSeek API via the OpenAI-compatible client."""
    client = make_client(api_key, base_url)
    try:
        response = client.models.list()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(str(exc)) from exc

    candidates: List[str] = []
    data = getattr(response, "data", None)
    if data is None and hasattr(response, "__iter__"):
        data = list(response)
    if not data:
        return []

    for entry in data:
        model_id = getattr(entry, "id", None)
        if not model_id and hasattr(entry, "get"):
            model_id = entry.get("id")
        if model_id:
            candidates.append(str(model_id))

    seen: set[str] = set()
    unique: List[str] = []
    for model_id in candidates:
        if model_id in seen:
            continue
        seen.add(model_id)
        unique.append(model_id)
    return unique


def update_model_options(api_key: str, base_url: str, *, force: bool = False) -> None:
    """Refresh the cached list of models when credentials change or a manual refresh is requested."""
    key = api_key.strip()
    url = base_url.strip() or DEFAULT_BASE_URL
    params = (key, url)
    should_refresh = force or params != st.session_state.model_params
    st.session_state.model_params = params

    if not key:
        st.session_state.model_options = [DEFAULT_MODEL]
        st.session_state.model_fetch_error = "Provide an API key to list available models."
        if st.session_state.selected_model not in st.session_state.model_options:
            st.session_state.selected_model = st.session_state.model_options[0]
        return

    if not should_refresh:
        return

    try:
        models = fetch_available_models(key, url)
    except Exception as exc:  # noqa: BLE001
        st.session_state.model_options = [DEFAULT_MODEL]
        st.session_state.model_fetch_error = f"Could not load models: {exc}"
    else:
        st.session_state.model_options = models or [DEFAULT_MODEL]
        st.session_state.model_fetch_error = ""

    if st.session_state.selected_model not in st.session_state.model_options:
        st.session_state.selected_model = st.session_state.model_options[0]



def slugify_filename(value: str, fallback: str = "story") -> str:
    """Return a filesystem-friendly slug for the provided value."""
    value = (value or "").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value)
    slug = slug.strip("_")
    return slug or fallback



def assemble_story_text(chapter_outputs: List[tuple[str, str]]) -> str:
    """Combine chapter markdown into a single string suitable for PDF rendering."""
    parts: List[str] = []
    for index, (_, content) in enumerate(chapter_outputs, start=1):
        body = (content or "").strip()
        if not body:
            continue
        parts.append(f"Chapter {index}\n\n{body}\n")
    return "\n\n".join(parts).strip()



def initialize_workflow_state(
    *,
    config: Dict[str, Any],
    story_prompt: str,
) -> Dict[str, Any]:
    out_dir = ensure_output_dir(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    state: Dict[str, Any] = {
        "config": config,
        "story_prompt": story_prompt,
        "stage_index": 0,
        "messages": [{"role": "system", "content": "You are a helpful assistant"}],
        "final_count": None,
        "proposed_chapters": None,
        "followup_prompts": None,
        "followup_for": None,
        "chapters_written": 0,
        "out_dir": str(out_dir),
        "applied_feedback": {},
        "planning_outputs": [],
        "chapter_outputs": [],
        "stage_history": [],
    }
    return state


def append_log(key: str, new_text: str) -> None:
    text = (new_text or "").strip()
    if not text:
        return
    existing = st.session_state.get(key, "")
    if existing:
        st.session_state[key] = f"{existing}\n{text}"
    else:
        st.session_state[key] = text


def apply_feedback_message(
    state: Dict[str, Any],
    *,
    key: str,
    text: str,
    header: str,
    footer: str,
) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    applied: Dict[str, str] = state.setdefault("applied_feedback", {})
    if applied.get(key) == cleaned:
        return False
    message = f"{header}\n{cleaned}\n{footer}"
    state["messages"].append({"role": "user", "content": message})
    applied[key] = cleaned
    return True


def apply_pending_feedback(state: Dict[str, Any]) -> None:
    stage_index = state.get("stage_index", 0)
    if stage_index == 0:
        return

    applied = state.setdefault("applied_feedback", {})

    if stage_index <= 5:
        key = f"feedback_step_{stage_index}"
        text = st.session_state.get(key, "")
        apply_feedback_message(
            state,
            key=key,
            text=text,
            header=f"FEEDBACK FOR STEP {stage_index}:",
            footer="Please apply this feedback in the next steps.",
        )

    if stage_index >= 5:
        chapter_num = stage_index - 4
        specific_key = f"feedback_chapter_{chapter_num:02d}"
        specific_text = st.session_state.get(specific_key, "")
        apply_feedback_message(
            state,
            key=specific_key,
            text=specific_text,
            header=f"FEEDBACK AFTER CHAPTER {chapter_num}:",
            footer="Please apply this feedback in the next chapter(s).",
        )

        general_text = st.session_state.get("feedback_chapter_all", "")
        apply_feedback_message(
            state,
            key="feedback_chapter_all",
            text=general_text,
            header="GENERAL CHAPTER FEEDBACK:",
            footer="Please keep this guidance in upcoming chapters.",
        )


def rewind_state_to_stage(state: Dict[str, Any], target_stage_index: int) -> None:
    history: List[Dict[str, Any]] = state.get("stage_history", [])
    if not history:
        state["stage_index"] = target_stage_index
        return

    removed: List[Dict[str, Any]] = []
    while history and history[-1]["stage_index"] >= target_stage_index:
        removed.append(history.pop())

    if not removed:
        state["stage_index"] = target_stage_index
        return

    remove_from = min(record["messages_start"] for record in removed)
    state["messages"] = state["messages"][:remove_from]

    applied = state.setdefault("applied_feedback", {})
    planning_outputs = state.get("planning_outputs", [])
    chapter_outputs = state.get("chapter_outputs", [])

    for record in removed:
        if record["type"] == "step":
            if planning_outputs:
                planning_outputs.pop()
            applied.pop(f"feedback_step_{record['step_number']}", None)
        elif record["type"] == "chapter":
            if chapter_outputs:
                chapter_outputs.pop()
            applied.pop(f"feedback_chapter_{record['chapter_number']:02d}", None)

    state["chapters_written"] = len(chapter_outputs)
    state["stage_index"] = target_stage_index


def update_active_tab(tab_label: Optional[str]) -> None:
    """Update the active tab in session state."""
    if not tab_label:
        return
    st.session_state["active_tab"] = tab_label


AUTO_SCROLL_ENABLE_SCRIPT = """
<script>
(function() {
    const win = (window.parent && window.parent !== window) ? window.parent : window;
    if (!win) {
        return;
    }
    if (!win.__eqAutoScrollTimer) {
        const tick = () => {
            if (!win.__eqAutoScrollActive) {
                return;
            }
            const doc = win.document;
            if (!doc) {
                return;
            }
            const height = (doc.documentElement && doc.documentElement.scrollHeight) || (doc.body && doc.body.scrollHeight) || 0;
            if (height) {
                win.scrollTo({ top: height, behavior: 'smooth' });
            }
        };
        win.__eqAutoScrollTimer = win.setInterval(tick, 200);
    }
    win.__eqAutoScrollActive = true;
})();
</script>
"""


AUTO_SCROLL_DISABLE_SCRIPT = """
<script>
(function() {
    const win = (window.parent && window.parent !== window) ? window.parent : window;
    if (!win) {
        return;
    }
    win.__eqAutoScrollActive = false;
})();
</script>
"""


def enable_auto_scroll() -> None:
    components_html(AUTO_SCROLL_ENABLE_SCRIPT, height=0)


def disable_auto_scroll() -> None:
    components_html(AUTO_SCROLL_DISABLE_SCRIPT, height=0)


def sync_final_count_from_ui(state: Dict[str, Any]) -> Optional[int]:
    if "confirmed_chapter_count" in st.session_state:
        value = st.session_state.confirmed_chapter_count
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = None
        if candidate and candidate > 0:
            state["final_count"] = candidate
            return candidate

    final_count = state.get("final_count")
    if final_count:
        try:
            return int(final_count)
        except (TypeError, ValueError):
            return None
    return None


def advance_stage(
    state: Dict[str, Any],
    *,
    cache_obj: Optional[PromptCache],
    stream_enabled: bool,
    progress_callback,
    stream_callback,
) -> Dict[str, Any]:
    config = state["config"]
    api_key = config["api_key"]
    base_url = config.get("base_url", DEFAULT_BASE_URL)
    model = config.get("model", DEFAULT_MODEL)
    temperature = config.get("temperature", 1.0)

    client = make_client(api_key, base_url)
    stage_index = state.get("stage_index", 0)
    progress_label = STEP_LABELS.get(stage_index)
    progress_callback(progress_label or f"Chapter {stage_index - 4}")

    apply_pending_feedback(state)
    if stage_index >= 1:
        sync_final_count_from_ui(state)

    messages: List[Dict[str, str]] = state["messages"]
    history: List[Dict[str, Any]] = state.setdefault("stage_history", [])
    out_dir = Path(state["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_prompt(prompt: str) -> str:
        messages.append({"role": "user", "content": prompt})
        result = chat_once(
            client,
            model,
            messages,
            stream=stream_enabled,
            temperature=temperature,
            cache=cache_obj,
            on_token=stream_callback if stream_enabled else None,
        )
        messages.append({"role": "assistant", "content": result})
        return result

    # Step 1
    if stage_index == 0:
        messages_before = len(messages)
        prompt = build_step1_prompt().replace("{STORY_PROMPT}", state["story_prompt"])
        content = run_prompt(prompt)
        write_step_output(out_dir, 1, content)
        state["planning_outputs"].append(("Step 1", content))
        state["stage_index"] = 1

        proposed = parse_proposed_chapters(content)
        state["proposed_chapters"] = proposed
        if state.get("final_count") is None:
            state["final_count"] = proposed

        if "confirmed_chapter_count" not in st.session_state:
            seed = state.get("final_count") or proposed or 1
            st.session_state.confirmed_chapter_count = max(int(seed), 1)

        history.append(
            {
                "stage_index": 0,
                "type": "step",
                "label": "Step 1",
                "step_number": 1,
                "messages_start": messages_before,
                "messages_end": len(messages),
            }
        )

        return {
            "label": STEP_LABELS[0],
            "type": "step",
            "index": 1,
            "content": content,
        }

    # Planning steps 2-5
    if stage_index < 5:
        final_count = sync_final_count_from_ui(state)
        if not final_count:
            raise RuntimeError("Set a chapter count before continuing.")

        followup_for = state.get("followup_for")
        if state.get("followup_prompts") is None or followup_for != final_count:
            state["followup_prompts"] = build_followup_prompts(final_count)
            state["followup_for"] = final_count

        prompt_index = stage_index - 1
        prompt = state["followup_prompts"][prompt_index]
        messages_before = len(messages)
        content = run_prompt(prompt)
        write_step_output(out_dir, stage_index + 1, content)
        label = STEP_LABELS[stage_index]
        step_label = f"Step {stage_index + 1}"
        state["planning_outputs"].append((step_label, content))
        state["stage_index"] += 1

        history.append(
            {
                "stage_index": stage_index,
                "type": "step",
                "label": step_label,
                "step_number": stage_index + 1,
                "messages_start": messages_before,
                "messages_end": len(messages),
            }
        )

        return {
            "label": label,
            "type": "step",
            "index": stage_index + 1,
            "content": content,
        }

    # Chapter writing
    final_count = sync_final_count_from_ui(state)
    if not final_count:
        raise RuntimeError("Set a chapter count before writing chapters.")

    chapter_num = stage_index - 4
    if chapter_num > final_count:
        raise RuntimeError("All chapters have already been generated for this run.")

    prompt = chapter_prompt(chapter_num)
    messages_before = len(messages)
    messages.append({"role": "user", "content": prompt})

    attempts = 0
    final_text = ""
    last_response = ""
    last_word_count = 0

    while attempts < CHAPTER_MAX_ATTEMPTS:
        result = chat_once(
            client,
            model,
            messages,
            stream=stream_enabled,
            temperature=temperature,
            cache=cache_obj,
            on_token=stream_callback if stream_enabled else None,
        )
        messages.append({"role": "assistant", "content": result})
        last_response = result
        last_word_count = count_words(result)
        if last_word_count >= CHAPTER_MIN_WORDS:
            final_text = result
            break

        attempts += 1
        if attempts >= CHAPTER_MAX_ATTEMPTS:
            break

        progress_callback(f"Chapter {chapter_num} retry {attempts + 1}")
        retry_prompt = (
            f"The chapter you just provided for Chapter {chapter_num} contains {last_word_count} words, "
            f"but it must be at least {CHAPTER_MIN_WORDS} words of narrative prose. "
            "Please rewrite the entire chapter, keeping continuity with earlier chapters, and expand the storytelling so the final output meets or exceeds the requirement. "
            "Output only the revised chapter text with no commentary."
        )
        messages.append({"role": "user", "content": retry_prompt})

    if not final_text:
        final_text = last_response

    if last_word_count < CHAPTER_MIN_WORDS:
        print(
            f"Warning: Chapter {chapter_num} final word count {last_word_count} < {CHAPTER_MIN_WORDS} after {CHAPTER_MAX_ATTEMPTS} attempt(s).",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"Chapter {chapter_num} word count: {last_word_count}", flush=True)

    write_chapter_output(out_dir, chapter_num, final_text)
    state["chapter_outputs"].append((f"Chapter {chapter_num:02d}", final_text))
    state["chapters_written"] = chapter_num
    state["stage_index"] += 1

    history.append(
        {
            "stage_index": stage_index,
            "type": "chapter",
            "label": f"Chapter {chapter_num:02d}",
            "chapter_number": chapter_num,
            "messages_start": messages_before,
            "messages_end": len(messages),
        }
    )

    return {
        "label": f"Chapter {chapter_num}",
        "type": "chapter",
        "index": chapter_num,
        "content": final_text,
        "word_count": last_word_count,
    }


def workflow_complete(state: Dict[str, Any]) -> bool:
    final_count = state.get("final_count")
    if not final_count:
        return False
    try:
        total = int(final_count)
    except (TypeError, ValueError):
        return False
    return state.get("stage_index", 0) >= 5 + total


def reset_workflow_state() -> None:
    st.session_state.pop("workflow_state", None)
    st.session_state.pop("running_story_prompt", None)
    st.session_state.pop("confirmed_chapter_count", None)
    st.session_state["planning_outputs"] = []
    st.session_state["chapter_outputs"] = []
    st.session_state["last_stdout"] = ""
    st.session_state["last_stderr"] = ""
    st.session_state.pop("last_run_dir", None)
    st.session_state["generated_pdf"] = None
    st.session_state["generated_pdf_name"] = None
    st.session_state["pdf_status"] = ""
    st.session_state["pdf_title_input"] = "AI Author Story"
    st.session_state["uploaded_file_meta"] = None
    st.session_state.pop("active_tab", None)
    st.session_state.pop("planning_tab_index", None)
    st.session_state.pop("chapter_tab_index", None)


def execute_stage(state: Dict[str, Any], *, stream_enabled: bool) -> Optional[Dict[str, Any]]:
    cache_obj: Optional[PromptCache] = None
    cache_dir = state["config"].get("cache_dir")
    if cache_dir:
        try:
            cache_obj = PromptCache(Path(cache_dir))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not initialise cache ({exc}); continuing without caching.")
            cache_obj = None

    progress_placeholder = st.empty()
    stream_placeholder = None
    if stream_enabled:
        st.markdown(STREAM_BOX_STYLE, unsafe_allow_html=True)
        stream_placeholder = st.empty()
    stream_buffer = io.StringIO()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    def handle_progress(label: str) -> None:
        progress_placeholder.markdown(f"**{label}**")
        if stream_placeholder is not None:
            stream_buffer.seek(0)
            stream_buffer.truncate(0)
            stream_placeholder.empty()

    def handle_stream(token: str) -> None:
        stream_buffer.write(token)
        if stream_placeholder is not None:
            content = stream_buffer.getvalue()
            if not content:
                stream_placeholder.empty()
            else:
                    # Show streaming content as markdown with auto-scroll and accessibility label
                    stream_placeholder.markdown(
                        f'<div class="stream-box" aria-label="Streaming Response">{content}</div>'
                        '<script>\n'
                        'const box = window.document.querySelector(".stream-box");\n'
                        'if (box) { box.scrollTop = box.scrollHeight; }\n'
                        '</script>',
                        unsafe_allow_html=True
                    )

    runner_context = contextlib.nullcontext() if stream_enabled else st.spinner("Running stage...")

    auto_scroll_active = stream_enabled
    if auto_scroll_active:
        enable_auto_scroll()

    try:
        with runner_context:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                info = advance_stage(
                    state,
                    cache_obj=cache_obj,
                    stream_enabled=stream_enabled,
                    progress_callback=handle_progress,
                    stream_callback=handle_stream,
                )
    except Exception as exc:  # noqa: BLE001
        progress_placeholder.markdown("**Stage failed**")
        append_log("last_stdout", stdout_capture.getvalue())
        append_log("last_stderr", stderr_capture.getvalue())
        st.session_state.workflow_state = state
        st.session_state.last_run_dir = state.get("out_dir")
        st.error(f"Stage failed: {exc}")
        if auto_scroll_active:
            disable_auto_scroll()
        if stream_placeholder is not None:
            stream_placeholder.empty()
        return None

    append_log("last_stdout", stdout_capture.getvalue())
    append_log("last_stderr", stderr_capture.getvalue())
    progress_placeholder.markdown(f"**{info['label']} complete**")

    if stream_placeholder is not None:
        stream_placeholder.empty()

    st.session_state.workflow_state = state
    st.session_state.last_run_dir = state.get("out_dir")
    st.session_state.planning_outputs = state.get("planning_outputs", [])
    st.session_state.chapter_outputs = state.get("chapter_outputs", [])
    st.session_state.generated_pdf = None
    st.session_state.generated_pdf_name = None
    st.session_state.pdf_status = ""
    if info:
        # Update tab indices for new content
        if info:
            if info.get("type") == "step":
                step_idx = info.get("index", 0)
                if step_idx:
                    st.session_state["active_tab"] = f"Step {step_idx}"
                    # Force the planning tab index to update
                    st.session_state["planning_tab_index"] = len(state.get("planning_outputs", [])) - 1
            elif info.get("type") == "chapter":
                try:
                    chapter_idx = int(info.get("index", 0))
                    if chapter_idx > 0:
                        st.session_state["active_tab"] = f"Chapter {chapter_idx:02d}"
                        # Force the chapter tab index to update
                        st.session_state["chapter_tab_index"] = len(state.get("chapter_outputs", [])) - 1
                except (TypeError, ValueError):
                    pass
    if auto_scroll_active:
        disable_auto_scroll()
    return info


# Global variables
manual_next_clicked = False
should_run_stage = False

if "story_prompt" not in st.session_state:
    st.session_state.story_prompt = "Write me a children's story about a naughty ginger cat."
if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_stdout" not in st.session_state:
    st.session_state.last_stdout = ""
if "last_stderr" not in st.session_state:
    st.session_state.last_stderr = ""
if "planning_outputs" not in st.session_state:
    st.session_state.planning_outputs = []
if "chapter_outputs" not in st.session_state:
    st.session_state.chapter_outputs = []
if "uploaded_file_meta" not in st.session_state:
    st.session_state.uploaded_file_meta = None
if "pdf_title_input" not in st.session_state:
    st.session_state.pdf_title_input = "AI Author Story"
if "generated_pdf" not in st.session_state:
    st.session_state.generated_pdf = None
if "generated_pdf_name" not in st.session_state:
    st.session_state.generated_pdf_name = None
if "pdf_status" not in st.session_state:
    st.session_state.pdf_status = ""
if "model_options" not in st.session_state:
    st.session_state.model_options = [DEFAULT_MODEL]
if "model_fetch_error" not in st.session_state:
    st.session_state.model_fetch_error = ""
if "model_params" not in st.session_state:
    st.session_state.model_params = ("", "")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "sidebar_api_key" not in st.session_state:
    st.session_state.sidebar_api_key = os.getenv("API_KEY", "")
if "config_visible" not in st.session_state:
    st.session_state.config_visible = False
if "sidebar_base_url" not in st.session_state:
    st.session_state.sidebar_base_url = DEFAULT_BASE_URL
if "sidebar_temperature" not in st.session_state:
    st.session_state.sidebar_temperature = 1.0
if "sidebar_stream" not in st.session_state:
    st.session_state.sidebar_stream = True


if "workflow_state" in st.session_state and "running_story_prompt" in st.session_state:
    if st.session_state.story_prompt != st.session_state.running_story_prompt:
        reset_workflow_state()
        st.toast("Story idea changed, workflow reset.")


st.markdown(
    """
<style>
    .main .block-container > div[data-testid="stVerticalBlock"]:nth-child(3) {
        position: sticky;
        top: 2.875rem;
        background-color: var(--background-color, white);
        z-index: 999;
        border-bottom: 1px solid var(--divider-color, rgba(120, 120, 120, 0.3));
        /* padding-bottom: 1rem; */
    }
    .dark .main .block-container > div[data-testid="stVerticalBlock"]:nth-child(3) {
        background-color: var(--background-color, #0e1117);
    }
    section[data-testid="stTabs"] {
        margin-bottom: 0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(FEEDBACK_TEXTAREA_STYLE, unsafe_allow_html=True)
st.markdown(STREAM_BOX_STYLE, unsafe_allow_html=True)

workflow_state: Optional[Dict[str, Any]] = st.session_state.get("workflow_state")
workflow_active = workflow_state is not None
workflow_finished = workflow_active and workflow_complete(workflow_state)  # type: ignore[arg-type]


with st.container():
    st.title("AI Author")
    st.caption("Plan and draft stories using the DeepSeek workflow from a friendly UI")
    top_cols = st.columns([1,1])
    with top_cols[0]:
        show_config = st.checkbox("Show configuration sidebar", key="config_visible")
    with top_cols[1]:
        reset_clicked = st.button(
            "Reset",
            key="sidebar_restart_button",
            disabled=not st.session_state.get("workflow_state"),
        )
    setup_expander = st.expander("Story Setup", expanded=not workflow_active)

if show_config:
    with st.sidebar:
        st.header("Configuration")
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.sidebar_api_key,
            type="password",
            help="Stored locally in session state; never written to disk.",
            key="sidebar_api_key",
        )
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.sidebar_base_url,
            key="sidebar_base_url",
        )
        refresh_models_clicked = st.button(
            "Refresh models", help="Call /models using the current credentials"
        )
        update_model_options(api_key_input, base_url, force=refresh_models_clicked)
        selected_model = st.selectbox(
            "Model",
            options=st.session_state.model_options,
            key="selected_model",
        )
        if st.session_state.model_fetch_error:
            st.caption(st.session_state.model_fetch_error)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.sidebar_temperature,
            step=0.1,
            key="sidebar_temperature",
        )
        stream_toggle = st.checkbox(
            "Stream responses",
            value=st.session_state.sidebar_stream,
            help="Show tokens live instead of waiting for full responses",
            key="sidebar_stream",
        )
else:
    api_key_input = st.session_state.sidebar_api_key
    base_url = st.session_state.sidebar_base_url
    update_model_options(api_key_input, base_url, force=False)
    selected_model = st.session_state.selected_model
    if st.session_state.model_fetch_error:
        st.caption(st.session_state.model_fetch_error)
    temperature = st.session_state.sidebar_temperature
    stream_toggle = st.session_state.sidebar_stream

if workflow_state:
    workflow_state["config"]["temperature"] = temperature
start_clicked = False
with setup_expander:
    uploaded_file = st.file_uploader(
        "Upload idea file",
        type=["md", "txt", "markdown"],
        help="Optional: load the story idea from a local file.",
    )
    if uploaded_file is not None:
        content_bytes = uploaded_file.getvalue()
        checksum = hashlib.sha256(content_bytes).hexdigest()
        meta = (uploaded_file.name, checksum)
        if st.session_state.uploaded_file_meta != meta:
            try:
                loaded_text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                loaded_text = content_bytes.decode("utf-8", errors="replace")
            st.session_state.story_prompt = loaded_text.strip()
            st.session_state.uploaded_file_meta = meta
            st.toast(f"Loaded idea from {uploaded_file.name}")

    story_prompt_text = st.text_area(
        "Story idea",
        height=300,
        key="story_prompt",
        help="Provide the seed prompt the workflow should elaborate into a story.",
    )

    start_cols = st.columns([1, 1, 1])
    with start_cols[0]:
        start_clicked = st.button("Let's Begin", disabled=workflow_active)


should_run_stage = False

if reset_clicked:
    reset_workflow_state()
    st.session_state.workflow_state = None
    st.rerun()

if start_clicked:
    if not api_key_input.strip():
        st.error("Provide an API key to call the DeepSeek API.")
    elif not story_prompt_text.strip():
        st.error("Enter a story idea or load one of the samples.")
    else:
        reset_workflow_state()
        st.session_state.running_story_prompt = story_prompt_text.strip()
        config = {
            "api_key": api_key_input.strip(),
            "base_url": base_url.strip() or DEFAULT_BASE_URL,
            "model": selected_model or DEFAULT_MODEL,
            "temperature": temperature,
            "cache_dir": DEFAULT_CACHE_DIR,
            "output_dir": DEFAULT_OUTPUT_DIR,
        }
        workflow_state = initialize_workflow_state(
            config=config,
            story_prompt=story_prompt_text.strip(),
        )
        st.session_state.workflow_state = workflow_state
        st.session_state.last_stdout = ""
        st.session_state.last_stderr = ""
        st.session_state.planning_outputs = []
        st.session_state.chapter_outputs = []
        st.session_state.last_run_dir = workflow_state["out_dir"]
        execute_stage(workflow_state, stream_enabled=stream_toggle)
        st.rerun()

if workflow_state:
    st.session_state.planning_outputs = workflow_state.get("planning_outputs", [])
    st.session_state.chapter_outputs = workflow_state.get("chapter_outputs", [])

if st.session_state.get("last_run_dir"):

    if st.session_state.get("last_stdout"):
        with st.expander("Workflow log"):
            st.code(st.session_state.get("last_stdout"))

    if st.session_state.get("last_stderr"):
        with st.expander("Warnings / errors"):
            st.code(st.session_state.get("last_stderr"))

    if st.session_state.planning_outputs:
        # Set tab index to the newest tab when content is added
        st.markdown("### Planning Steps")
        for label, content in st.session_state.planning_outputs:
            with st.expander(label, expanded=True):
                st.markdown(content)
                step_parts = label.strip().split()
                step_num = step_parts[-1] if step_parts and step_parts[-1].isdigit() else None
                if label.startswith("Step 1") and "confirmed_chapter_count" in st.session_state:
                    st.number_input(
                        "Confirmed chapter count",
                        min_value=1,
                        max_value=50,
                        key="confirmed_chapter_count",
                    )
                if step_num:
                    step_index = int(step_num)
                    feedback_key = f"feedback_step_{step_index}"
                    st.text_area(
                        "Feedback to apply before the next step",
                        key=feedback_key,
                        height=150,
                        placeholder="Enter feedback for this step",
                    )
                    can_apply = (
                        workflow_state is not None
                        and not workflow_finished
                        and workflow_state.get("stage_index", 0) == step_index
                    )
                    feedback_raw = st.session_state.get(feedback_key, "")
                    feedback_value = feedback_raw.strip()
                    if can_apply:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            apply_clicked = st.button(
                                "Apply Feedback",
                                key=f"apply_feedback_step_{step_index}",
                                disabled=not feedback_raw,
                            )
                        with col2:
                            # Only disable if truly blocked (stage 1 and no final_count)
                            # Only disable if in Step 1 and final_count is missing
                            next_disabled = (
                                workflow_state is not None and
                                workflow_state.get("stage_index", 0) == 1 and
                                not workflow_state.get("final_count")
                            )
                            if st.button("Run Next Stage", disabled=next_disabled, key=f"next_stage_{step_index}"):
                                should_run_stage = True
                        if apply_clicked and workflow_state is not None and feedback_value:
                            target_stage_index = step_index - 1
                            rewind_state_to_stage(workflow_state, target_stage_index)
                            apply_feedback_message(
                                workflow_state,
                                key=feedback_key,
                                text=feedback_value,
                                header=f"FEEDBACK FOR STEP {step_index}:",
                                footer="Please apply this feedback in the next steps.",
                            )
                            st.session_state.workflow_state = workflow_state
                            st.session_state.planning_outputs = workflow_state.get("planning_outputs", [])
                            st.session_state.chapter_outputs = workflow_state.get("chapter_outputs", [])
                            st.session_state[feedback_key] = ""  # Clear feedback text area
                            should_run_stage = True
if st.session_state.chapter_outputs:
    st.markdown("### Chapters")
    for idx, (label, content) in enumerate(st.session_state.chapter_outputs, start=1):
        with st.expander(label, expanded=False):
            st.markdown(content)
            chapter_feedback_key = f"feedback_chapter_{idx:02d}"
            st.text_area(
                "Feedback to apply before the next chapter",
                key=chapter_feedback_key,
                height=80,
                placeholder="Enter feedback for this chapter",
            )
            can_apply_chapter = (
                workflow_state is not None

            and not workflow_finished
            and workflow_state.get("stage_index", 0) == idx + 4
        )
        chapter_feedback_raw = st.session_state.get(chapter_feedback_key, "")
        chapter_feedback_val = chapter_feedback_raw.strip()
        if can_apply_chapter:
            apply_chapter_clicked = st.button(
                "Apply Chapter Feedback",
                key=f"apply_feedback_chapter_{idx:02d}",
                disabled=not chapter_feedback_raw,
            )
            if apply_chapter_clicked and workflow_state is not None and chapter_feedback_val:
                target_stage_index = idx + 4
                rewind_state_to_stage(workflow_state, target_stage_index)
                apply_feedback_message(
                    workflow_state,
                    key=chapter_feedback_key,
                    text=chapter_feedback_val,
                    header=f"FEEDBACK AFTER CHAPTER {idx}:",
                    footer="Please apply this feedback in the next chapter(s).",
                )
                st.session_state.workflow_state = workflow_state
                st.session_state.planning_outputs = workflow_state.get("planning_outputs", [])
                st.session_state.chapter_outputs = workflow_state.get("chapter_outputs", [])
                should_run_stage = True

    # General feedback expander after all chapters
    with st.expander("General feedback for upcoming chapters", expanded=False):
        general_feedback_raw = st.text_area(
            "General feedback for upcoming chapters",
            key="feedback_chapter_all",
            height=80,
            placeholder="Enter overall guidance for upcoming chapters",
        )
        can_apply_chapter_all = (
            workflow_state is not None
            and not workflow_finished
            and workflow_state.get("stage_index", 0) >= 5
        )
        general_feedback_value = general_feedback_raw.strip() if general_feedback_raw else ""
        if can_apply_chapter_all:
            apply_general_clicked = st.button(
                "Apply General Chapter Feedback",
                key="apply_feedback_chapter_all",
                disabled=not general_feedback_raw,
            )
            if apply_general_clicked and workflow_state is not None and general_feedback_value:
                apply_feedback_message(
                    workflow_state,
                    key="feedback_chapter_all",
                    text=general_feedback_value,
                    header="GENERAL CHAPTER FEEDBACK:",
                    footer="Please keep this guidance in upcoming chapters.",
                )
                st.session_state.workflow_state = workflow_state
                st.session_state.planning_outputs = workflow_state.get("planning_outputs", [])
                st.session_state.chapter_outputs = workflow_state.get("chapter_outputs", [])
                should_run_stage = True

    if st.session_state.chapter_outputs:
        st.markdown("### Export to PDF")
        title_col, button_col = st.columns([3, 1])
        with title_col:
            st.text_input("PDF title", key="pdf_title_input")
        with button_col:
            render_pdf_clicked = st.button("Render PDF", key="render_pdf_button")

        if render_pdf_clicked:
            title_value = st.session_state.pdf_title_input.strip() or "EQ Author Story"
            story_text = assemble_story_text(st.session_state.chapter_outputs)
            if not story_text:
                st.error("No chapter content available for PDF export yet.")
            else:
                destination_dir = Path("story_output")
                destination_dir.mkdir(parents=True, exist_ok=True)
                run_segment = Path(st.session_state.get("last_run_dir") or '').name
                suffix = f"_{run_segment}" if run_segment else ""
                filename = f"{slugify_filename(title_value)}{suffix}.pdf"
                pdf_path = destination_dir / filename
                try:
                    process_story_to_pdf(story_text, str(pdf_path), title_value)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.generated_pdf = None
                    st.session_state.generated_pdf_name = None
                    st.session_state.pdf_status = ""
                    st.error(f"Failed to render PDF: {exc}")
                else:
                    try:
                        pdf_bytes = pdf_path.read_bytes()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Rendered PDF but could not read file: {exc}")
                    else:
                        st.session_state.generated_pdf = pdf_bytes
                        st.session_state.generated_pdf_name = filename
                        st.session_state.pdf_status = f"PDF saved to `{pdf_path}`."
                        st.success("PDF ready for download.")

        if st.session_state.get("generated_pdf"):
            # Ensure we have valid PDF data before showing download button
            pdf_data = st.session_state.generated_pdf
            if isinstance(pdf_data, (str, bytes)) and pdf_data:
                st.download_button(
                    "Download PDF",
                    data=pdf_data,
                    file_name=st.session_state.generated_pdf_name or "story.pdf",
                    mime="application/pdf",
                    key="download_pdf_button",
                )
            if st.session_state.pdf_status:
                st.caption(st.session_state.pdf_status)
        elif st.session_state.pdf_status:
            st.caption(st.session_state.pdf_status)

    if workflow_state and not workflow_finished:
        sync_final_count_from_ui(workflow_state)

if should_run_stage and workflow_state and not workflow_finished:
    execute_stage(workflow_state, stream_enabled=stream_toggle)
    st.rerun()

if workflow_active and workflow_finished:
    st.success("Workflow complete. You can review the outputs or reset to start over.")
