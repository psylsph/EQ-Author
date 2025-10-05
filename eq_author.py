import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dotenv import load_dotenv

try:
    # OpenAI SDK v1 style import; works with DeepSeek via base_url
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore


STEP_FILENAMES = {
    1: "01_brainstorm_and_reflection.md",
    2: "02_intention_and_chapter_planning.md",
    3: "03_human_vs_llm_critique.md",
    4: "04_final_plan.md",
    5: "05_characters.md",
}

CHAPTER_WORD_TARGET = 3000
CHAPTER_MIN_WORDS = CHAPTER_WORD_TARGET
CHAPTER_MAX_ATTEMPTS = 3

WORD_RE = re.compile(r"\b[\w'\-]+\b")

# Context management constants
DEFAULT_CONTEXT_STRATEGY = "aggressive"
DEFAULT_SUMMARY_LENGTH = 250
DEFAULT_RECENT_CHAPTERS = 2
CONTEXT_WARNING_THRESHOLD = 0.8
CONTEXT_CRITICAL_THRESHOLD = 0.95


class ContextManager:
    """Manages conversation context to prevent overflow in limited-context models."""
    
    def __init__(self, strategy: str = DEFAULT_CONTEXT_STRATEGY, summary_length: int = DEFAULT_SUMMARY_LENGTH,
                 recent_chapters: int = DEFAULT_RECENT_CHAPTERS, max_context_tokens: int = 8000):
        self.strategy = strategy
        self.summary_length = summary_length
        self.recent_chapters = recent_chapters
        self.max_context_tokens = max_context_tokens
        
        # Context storage
        self.core_context = []  # Planning steps, character info
        self.chapter_summaries = []  # All chapter summaries
        self.recent_full_chapters = []  # Last N full chapters (if strategy allows)
        self.current_messages = []  # Current conversation context
        
    def add_core_context(self, messages: List[Dict[str, str]]) -> None:
        """Add core planning context that should always be preserved."""
        self.core_context = messages.copy()
        
    def add_chapter(self, chapter_num: int, chapter_text: str, summary: str) -> None:
        """Add a new chapter and its summary to the context manager."""
        self.chapter_summaries.append({
            "chapter": chapter_num,
            "summary": summary,
            "full_text": chapter_text if self.strategy == "full" else None
        })
        
        # Keep recent full chapters if strategy allows
        if self.strategy == "balanced" or self.strategy == "full":
            self.recent_full_chapters.append({
                "chapter": chapter_num,
                "text": chapter_text
            })
            # Keep only the most recent chapters
            if len(self.recent_full_chapters) > self.recent_chapters:
                self.recent_full_chapters.pop(0)
    
    def build_context(self, current_chapter: int) -> List[Dict[str, str]]:
        """Build the context for the current chapter generation."""
        context = []
        
        # Always include core context (planning, characters)
        context.extend(self.core_context)
        
        # Add chapter summaries based on strategy
        if self.strategy == "aggressive":
            # Only summaries of recent chapters
            recent_summaries = self.chapter_summaries[-self.recent_chapters:]
            for summary_info in recent_summaries:
                context.append({
                    "role": "assistant",
                    "content": f"Chapter {summary_info['chapter']} Summary:\n{summary_info['summary']}"
                })
                
        elif self.strategy == "balanced":
            # Summaries of all chapters + full text of recent ones
            for summary_info in self.chapter_summaries:
                context.append({
                    "role": "assistant",
                    "content": f"Chapter {summary_info['chapter']} Summary:\n{summary_info['summary']}"
                })
            
            # Add full text of recent chapters
            for chapter_info in self.recent_full_chapters:
                context.append({
                    "role": "assistant",
                    "content": f"Chapter {chapter_info['chapter']} Full Text:\n{chapter_info['text']}"
                })
                
        elif self.strategy == "full":
            # Try to include everything (may cause overflow)
            for summary_info in self.chapter_summaries:
                context.append({
                    "role": "assistant",
                    "content": f"Chapter {summary_info['chapter']} Full Text:\n{summary_info['full_text']}"
                })
        
        return context
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (approximately 4 characters per token)."""
        return len(text) // 4
    
    def check_context_size(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Check if context is approaching limits and return status."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = self.estimate_tokens("".join(msg.get("content", "") for msg in messages))
        
        usage_ratio = estimated_tokens / self.max_context_tokens
        
        return {
            "estimated_tokens": estimated_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_ratio": usage_ratio,
            "is_warning": usage_ratio > CONTEXT_WARNING_THRESHOLD,
            "is_critical": usage_ratio > CONTEXT_CRITICAL_THRESHOLD
        }


def generate_chapter_summary(client: Any, model: str, chapter_text: str, chapter_num: int,
                           target_length: int = DEFAULT_SUMMARY_LENGTH,
                           stream: bool = False, temperature: Optional[float] = None,
                           cache: Optional['PromptCache'] = None) -> str:
    """Generate a concise summary of a chapter for context management."""
    
    summary_prompt = (
        f"Create a concise summary of Chapter {chapter_num} (approximately {target_length} words) that captures:\n"
        f"- Key plot developments and events\n"
        f"- Character decisions and changes\n"
        f"- Important revelations or discoveries\n"
        f"- Emotional arcs and relationship dynamics\n"
        f"- Any cliffhangers or setup for future events\n\n"
        f"Chapter text:\n{chapter_text}\n\n"
        f"Provide only the summary without commentary or word count mentions."
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that creates concise chapter summaries."},
        {"role": "user", "content": summary_prompt}
    ]
    
    summary = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        cache=cache
    )
    
    return summary.strip()


def count_words(text: str) -> int:
    """Return an approximate word count suitable for prose validation."""
    if not text:
        return 0
    return len(WORD_RE.findall(text))


class PromptCache:
    """File-backed cache for chat responses to avoid repeat prompting."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self, *, model: str, messages: List[Dict[str, str]], temperature: Optional[float]
    ) -> Path:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(
        self, *, model: str, messages: List[Dict[str, str]], temperature: Optional[float]
    ) -> Optional[str]:
        cache_path = self._cache_path(model=model, messages=messages, temperature=temperature)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data.get("content")

    def set(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        content: str,
    ) -> None:
        cache_path = self._cache_path(model=model, messages=messages, temperature=temperature)
        cache_path.write_text(
            json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8"
        )


def load_story_prompt(story_file: Optional[str], story_text: Optional[str]) -> str:
    if story_text and story_text.strip():
        return story_text.strip()
    if not story_file:
        raise ValueError("Provide either --story-file or --story-text")
    p = Path(story_file)
    if not p.exists():
        raise FileNotFoundError(f"Story file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def build_step1_prompt() -> str:
    # Step 1 does not assume a chapter count; the model proposes one.
    return (
        "Initial Writing Prompt:\n"
        "{{STORY_PROMPT}}\n--\n"
        "Your task is to create a writing plan for this prompt. The scope will be a long format novel; do not assume a fixed number of chapters yet. "
        "Your plan should be comprehensive and in this format:\n"
        "# Brainstorming\n"
        "<Brainstorm ideas for characters, plot, tone, story beats, and possible pacing. The purpose of brainstorming is to cast a wide net of ideas, not to settle on any specific direction. "
        "Think about various ways you could take the prompt.>\n"
        "# Reflection\n"
        "<Reflect out loud on what works and doesn't work in these ideas. The purpose of this reflection is to narrow in on what you think will work best to make a piece that is a. compelling, and b. fits the prompt requirements. "
        "You are not making any decisions just yet, just reflecting.>\n"
        "Finally, propose the ideal number of chapters for this long format novel based on the prompt and your analysis.\n"
        "Output a single line at the end in this exact format so it can be parsed reliably:\n"
        "CHAPTER_COUNT: <integer>\n"
    )


def build_followup_prompts(n_chapters: int) -> List[str]:
    # Steps 2-5, now that a chapter count is known
    p2 = (
        "Great now let's continue with planning the long format novel. Output in this format:\n"
        "# Intention\n"
        "<State your formulated intentions for the piece, synthesised from the the parts of the brainstorming session that worked, and avoiding the parts that didn't. "
        "Be explicit about the choices you have made about plot, voice, stylistic choices, things you intend to aim for & avoid.>\n"
        "# Chapter Planning\n"
        "<Write a brief chapter plan for all {n_chapters} chapters.>"
    ).format(n_chapters=n_chapters)

    p3 = (
        "With a view to making the writing more human, discuss how a human might approach this particular piece (given the original prompt). "
        "Discuss telltale LLM approaches to writing (generally) and ways they might not serve this particular piece. For example, common LLM failings are to write safely, or to always wrap things up with a bow, or trying to write impressively at the expense of readability. "
        "Then do a deep dive on the intention & plan, critiquing ways it might be falling into typical LLM tropes & pitfalls. Brainstorm ideas to make it more human. Be comprehensive. We aren't doing any rewriting of the plan yet, just critique & brainstorming."
    )

    p4 = (
        "Ok now with these considerations in mind, formulate the final plan for the a human like, compelling short piece in {n_chapters} chapters. Bear in mind the constraints of the piece (each chapter is just 3000 words). "
        "Above all things, the plan must serve the original prompt. We will use the same format as before:\n"
        "# Intention\n"
        "<State your formulated intentions for the piece, synthesised from the the parts of the brainstorming session that worked, and avoiding the parts that didn't. "
        "Be explicit about the choices you have made about plot, voice, stylistic choices, things you intend to aim for & avoid.>\n"
        "# Chapter Planning\n"
        "<Write a brief chapter plan for all {n_chapters} chapters.>"
    ).format(n_chapters=n_chapters)

    p5 = (
        "Perfect. Now with the outline more crystallised, and bearing in mind the discussion on human writing vs LLM pitfalls, we will flesh out our characters. Lets go through each of our main characters:\n"
        "- Write about their background, personality, idiosyncrasies, flaws. Be specific and come up with examples to anchor & ground the character's profile (both core and trivial)\n"
        "- Briefly describe their physicality: appearance, how they carry themselves, express, interact with the world.\n"
        "- Concisely detail their motives, allegiances and existing relationships. Think from the perspective of the character as a real breathing thinking feeling individual in this world.\n"
        "- Write a couple quotes of flavour dialogue / internal monologue from the character to experiment with their voice.\n"
        "Output like this:\n"
        "# Character 1 name\n<character exploration>\n# Character 2 name\n<character exploration>\n etc"
    )

    return [p2, p3, p4, p5]


def ensure_output_dir(base_dir: str) -> Path:
    base = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / f"run-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    # also a subdir for chapters
    (out / "chapters").mkdir(parents=True, exist_ok=True)
    return out


def write_step_output(out_dir: Path, step: int, content: str) -> Path:
    name = STEP_FILENAMES.get(step, f"step_{step:02d}.md")
    p = out_dir / name
    p.write_text(content, encoding="utf-8")
    return p


def write_chapter_output(out_dir: Path, chapter_index: int, content: str) -> Path:
    p = out_dir / "chapters" / f"chapter_{chapter_index:02d}.md"
    p.write_text(content, encoding="utf-8")
    return p


def parse_proposed_chapters(text: str) -> Optional[int]:
    # Look for a line like: CHAPTER_COUNT: 6
    for line in reversed(text.splitlines()):
        line = line.strip().replace("*", "")  # clean markdown bold if any
        if line.upper().find("CHAPTER_COUNT:") >= 0:
            parts = line.split(":", 1)
            if len(parts) == 2:
                num = parts[1].strip()
                try:
                    return int(num)
                except ValueError:
                    return None
    return None


def collect_feedback(step_label: Any) -> str:
    try:
        print(f"\nProvide feedback for {step_label} (press Enter to skip):")
        fb = input().strip()
        return fb
    except KeyboardInterrupt:
        print("\nSkipping feedback.")
        return ""


def chapter_prompt(i: int) -> str:
    base_intro = (
        f"Write Chapter {i} of the story, following the approved plan and prior chapters.\n"
        f"- Produce at least {CHAPTER_MIN_WORDS} words of narrative prose.\n"
        "- Count only the words in your final story text; do not include planning notes or analysis.\n"
        "- Output only the polished chapter text (you may open with a 'Chapter {i}' heading if that matches the style), and do not mention the word count or include any commentary."
    )
    if i == 1:
        return (
            "Great. Let's begin the story.\n" + base_intro
        )
    return (
        f"Continue with the next installment.\n" + base_intro
    )


def make_client(api_key: str, base_url: str) -> Any:
    if OpenAI is None:
        raise RuntimeError(
            "openai package not available. Install with: pip install openai"
        )
    
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # Test connection with a simple models list request
        print(f"Testing connection to: {base_url}")
        models = client.models.list()
        print(f"Connection successful. Available models: {len(models.data)}")
        return client
    except Exception as e:
        error_msg = f"Failed to connect to API at {base_url}: {str(e)}"
        if "Connection" in str(e) or "connect" in str(e).lower():
            error_msg += "\n\nConnection troubleshooting tips:"
            error_msg += "\n- Ensure the server is running and accessible"
            error_msg += "\n- Check if the URL is correct (e.g., http://localhost:1234 for LM Studio)"
            error_msg += "\n- Verify no firewall is blocking the connection"
            error_msg += "\n- For LM Studio, make sure 'Server' mode is enabled"
        elif "401" in str(e) or "unauthorized" in str(e).lower():
            error_msg += "\n\nAuthentication troubleshooting tips:"
            error_msg += "\n- Check if API key is correct"
            error_msg += "\n- For LM Studio, API key may not be required (use any non-empty string)"
        elif "404" in str(e) or "not found" in str(e).lower():
            error_msg += "\n\nEndpoint troubleshooting tips:"
            error_msg += "\n- Verify the base URL is correct"
            error_msg += "\n- Check if the API endpoint path is correct"
        raise RuntimeError(error_msg)


def chat_once(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    temperature: Optional[float] = None,
    on_token: Optional[Any] = None,
    cache: Optional[PromptCache] = None,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    if cache is not None:
        cached = cache.get(model=model, messages=messages, temperature=temperature)
        if cached is not None:
            if stream and on_token is not None:
                try:
                    on_token(cached)
                except Exception:
                    pass
            return cached

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        error_msg = f"API request failed: {str(e)}"
        if "timeout" in str(e).lower():
            error_msg += "\n\nTimeout troubleshooting tips:"
            error_msg += "\n- The request took too long to complete"
            error_msg += "\n- Try again with a shorter prompt or simpler request"
            error_msg += "\n- Check if the server is overloaded"
        elif "rate" in str(e).lower() and "limit" in str(e).lower():
            error_msg += "\n\nRate limit troubleshooting tips:"
            error_msg += "\n- Wait a moment before making another request"
            error_msg += "\n- For LM Studio, this shouldn't occur locally"
        elif "model" in str(e).lower() and "not" in str(e).lower() and "found" in str(e).lower():
            error_msg += "\n\nModel troubleshooting tips:"
            error_msg += "\n- Check if the model name is correct"
            error_msg += "\n- For LM Studio, ensure a model is loaded"
            error_msg += "\n- Use client.models.list() to see available models"
        raise RuntimeError(error_msg)

    if stream:
        # When streaming, aggregate content
        full = []
        try:
            for chunk in resp:  # type: ignore[attr-defined]
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta  # type: ignore[index]
                    if delta and getattr(delta, "content", None):
                        text = delta.content
                        full.append(text)
                        if on_token is not None:
                            try:
                                on_token(text)
                            except Exception:
                                # Don't break streaming if printing fails
                                pass
        except Exception as e:
            error_msg = f"Streaming error occurred: {str(e)}"
            error_msg += "\n\nStreaming troubleshooting tips:"
            error_msg += "\n- Connection may have been interrupted"
            error_msg += "\n- Try again with streaming disabled (--no-stream)"
            raise RuntimeError(error_msg)
        result = "".join(full)
    else:
        try:
            result = resp.choices[0].message.content or ""  # type: ignore[attr-defined]
        except (AttributeError, IndexError) as e:
            error_msg = f"Invalid response format from API: {str(e)}"
            error_msg += "\n\nResponse format troubleshooting tips:"
            error_msg += "\n- The API response structure was unexpected"
            error_msg += "\n- Check if the server is functioning correctly"
            error_msg += "\n- Try with a different model or server"
            raise RuntimeError(error_msg)

    if cache is not None:
        try:
            cache.set(
                model=model, messages=messages, temperature=temperature, content=result
            )
        except Exception:
            pass

    return result


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
    max_context_tokens: int = 8000,
) -> None:
    client = make_client(api_key, base_url)

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

    def maybe_collect_feedback(label: Any) -> str:
        if feedback_callback is not None:
            try:
                return feedback_callback(str(label)) or ""
            except Exception:
                return ""
        if interactive:
            return collect_feedback(label)
        return ""

    # Initialize context manager
    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant"},
    ]

    # Step 1: brainstorm + reflection + propose chapter count
    step1 = build_step1_prompt().replace("{STORY_PROMPT}", story_prompt)
    messages.append({"role": "user", "content": step1})
    if stream:
        print("\n=== Step 1: Brainstorm & Reflection (streaming) ===", flush=True)
    emit_progress("Step 1: Brainstorm & Reflection")
    s1 = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        on_token=token_handler if stream else None,
        cache=cache,
    )
    if stream:
        print("\n", flush=True)
    write_step_output(out_dir, 1, s1)
    messages.append({"role": "assistant", "content": s1})

    proposed = parse_proposed_chapters(s1)
    # Prefer CLI override if provided; otherwise use model proposal
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
    # Non-interactive: if we still don't have a chapter count, bail with a clear error
    if final_count is None:
        raise RuntimeError("No chapter count available. Rerun interactively or pass --n-chapters.")

    fb = maybe_collect_feedback(1)
    if fb:
        messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 1:\n{fb}\nPlease apply this feedback in the next steps."})

    # Build follow-up prompts with confirmed chapter count
    step_texts = build_followup_prompts(final_count)

    # Step 2: intention + chapter planning
    messages.append({"role": "user", "content": step_texts[0]})
    if stream:
        print("\n=== Step 2: Intention & Chapter Planning (streaming) ===", flush=True)
    emit_progress("Step 2: Intention & Chapter Planning")
    s2 = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        on_token=token_handler if stream else None,
        cache=cache,
    )
    if stream:
        print("\n", flush=True)
    write_step_output(out_dir, 2, s2)
    messages.append({"role": "assistant", "content": s2})
    fb = maybe_collect_feedback(2)
    if fb:
        messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 2:\n{fb}\nPlease apply this feedback in the next steps."})

    # Step 3: human vs LLM critique
    messages.append({"role": "user", "content": step_texts[1]})
    if stream:
        print("\n=== Step 3: Human vs LLM Critique (streaming) ===", flush=True)
    emit_progress("Step 3: Human vs LLM Critique")
    s3 = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        on_token=token_handler if stream else None,
        cache=cache,
    )
    if stream:
        print("\n", flush=True)
    write_step_output(out_dir, 3, s3)
    messages.append({"role": "assistant", "content": s3})
    fb = maybe_collect_feedback(3)
    if fb:
        messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 3:\n{fb}\nPlease apply this feedback in the next steps."})

    # Step 4: final plan
    messages.append({"role": "user", "content": step_texts[2]})
    if stream:
        print("\n=== Step 4: Final Plan (streaming) ===", flush=True)
    emit_progress("Step 4: Final Plan")
    s4 = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        on_token=token_handler if stream else None,
        cache=cache,
    )
    if stream:
        print("\n", flush=True)
    write_step_output(out_dir, 4, s4)
    messages.append({"role": "assistant", "content": s4})
    fb = maybe_collect_feedback(4)
    if fb:
        messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 4:\n{fb}\nPlease apply this feedback in the next steps."})

    # Step 5: characters
    messages.append({"role": "user", "content": step_texts[3]})
    if stream:
        print("\n=== Step 5: Characters (streaming) ===", flush=True)
    emit_progress("Step 5: Characters")
    s5 = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        on_token=token_handler if stream else None,
        cache=cache,
    )
    if stream:
        print("\n", flush=True)
    write_step_output(out_dir, 5, s5)
    messages.append({"role": "assistant", "content": s5})
    fb = maybe_collect_feedback(5)
    if fb:
        messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 5:\n{fb}\nPlease apply this feedback in the next steps."})

    # Store core planning context in the context manager
    context_manager.add_core_context(messages)

    # Chapter writing: chapter 1..N
    for i in range(1, final_count + 1):
        # Build context for current chapter
        context_messages = context_manager.build_context(i)
        context_messages.append({"role": "user", "content": chapter_prompt(i)})
        
        # Check context size
        context_status = context_manager.check_context_size(context_messages)
        if context_status["is_warning"]:
            print(f"Warning: Context usage at {context_status['usage_ratio']:.1%} ({context_status['estimated_tokens']}/{context_status['max_tokens']} tokens)", flush=True)
        if context_status["is_critical"]:
            print(f"CRITICAL: Context usage at {context_status['usage_ratio']:.1%} - consider reducing context size", flush=True)

        if stream:
            print(f"\n=== Chapter {i} (streaming) ===", flush=True)
        emit_progress(f"Chapter {i}")

        attempts = 0
        final_text = ""
        last_response = ""
        last_word_count = 0

        while attempts < CHAPTER_MAX_ATTEMPTS:
            ch = chat_once(
                client,
                model,
                context_messages,
                stream=stream,
                temperature=temperature,
                on_token=token_handler if stream else None,
                cache=cache,
            )
            if stream:
                print("\n", flush=True)

            last_response = ch
            # Don't add to context_messages here - let context manager handle it
            last_word_count = count_words(ch)

            if last_word_count >= CHAPTER_MIN_WORDS:
                final_text = ch
                break

            attempts += 1
            if stream:
                print(
                    f"Chapter {i} returned {last_word_count} words (needs >= {CHAPTER_MIN_WORDS}).",
                    flush=True,
                )

            if attempts >= CHAPTER_MAX_ATTEMPTS:
                break

            retry_prompt = (
                f"The chapter you just provided for Chapter {i} contains {last_word_count} words, "
                f"but it must be at least {CHAPTER_MIN_WORDS} words of narrative prose. "
                "Please rewrite the entire chapter, keeping continuity with earlier chapters, and expand the storytelling so the final output meets or exceeds the requirement. "
                "Output only the revised chapter text with no commentary."
            )
            context_messages.append({"role": "user", "content": retry_prompt})

        if not final_text:
            final_text = last_response

        if last_word_count < CHAPTER_MIN_WORDS:
            print(
                f"Warning: Chapter {i} final word count {last_word_count} < {CHAPTER_MIN_WORDS} after {CHAPTER_MAX_ATTEMPTS} attempt(s).",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"Chapter {i} word count: {last_word_count}", flush=True)

        write_chapter_output(out_dir, i, final_text)

        # Generate chapter summary for context management
        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(
            client, model, final_text, i, summary_length, stream, temperature, cache
        )
        
        # Add chapter to context manager
        context_manager.add_chapter(i, final_text, chapter_summary)
        
        # Save summary to file for reference
        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")

        fb = maybe_collect_feedback(f"chapter {i}")
        if fb:
            # Add feedback to context manager for next chapter
            context_messages.append({"role": "user", "content": f"FEEDBACK AFTER CHAPTER {i}:\n{fb}\nPlease apply this feedback in the next chapter(s)."})


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EQ-Author DeepSeek Planner & Writer")
    p.add_argument("--story-file", type=str, help="Path to file containing the story idea", default=None)
    p.add_argument("--story-text", type=str, help="Story idea text (overrides file if provided)", default=None)
    p.add_argument("--n-chapters", type=int, help="Override: number of chapters to write (if omitted, model proposes)", default=None)
    p.add_argument("--output-dir", type=str, help="Base directory for outputs", default="outputs")
    p.add_argument("--api-key", type=str, help="DeepSeek API key (or set API_KEY)", default=None)
    p.add_argument("--base-url", type=str, help="DeepSeek base URL", default="https://api.deepseek.com")
    p.add_argument("--model", type=str, help="Model name", default="deepseek-reasoner")
    p.add_argument("--stream", action="store_true", help="Stream responses (aggregated in output)", default=True)
    p.add_argument("--no-stream", action="store_true", help="Disable response streaming", default=False)
    p.add_argument("--temperature", type=float, help="Sampling temperature", default=1.0)
    p.add_argument("--non-interactive", action="store_true", help="Run without feedback prompts and chapter count confirmation")
    p.add_argument("--no-cache", action="store_true", help="Disable prompt caching", default=False)
    p.add_argument(
        "--cache-dir",
        type=str,
        default=".prompt_cache",
        help="Directory for prompt/response cache (ignored if --no-cache is used)",
    )
    
    # Context management options
    p.add_argument("--context-strategy", type=str, choices=["aggressive", "balanced", "full"],
                   default=DEFAULT_CONTEXT_STRATEGY, help="Strategy for managing context window (default: aggressive)")
    p.add_argument("--summary-length", type=int, default=DEFAULT_SUMMARY_LENGTH,
                   help=f"Target word count for chapter summaries (default: {DEFAULT_SUMMARY_LENGTH})")
    p.add_argument("--recent-chapters", type=int, default=DEFAULT_RECENT_CHAPTERS,
                   help=f"Number of recent full chapters to keep in context (default: {DEFAULT_RECENT_CHAPTERS})")
    p.add_argument("--max-context-tokens", type=int, default=8000,
                   help="Maximum context tokens to maintain (default: 8000)")
    
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Load .env if present (root of project or current working dir)
    load_dotenv()

    api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Provide --api-key or set API_KEY/OPENAI_API_KEY", file=sys.stderr)
        return 2

    try:
        story_prompt = load_story_prompt(args.story_file, args.story_text)
    except Exception as e:
        print(f"Error loading story prompt: {e}", file=sys.stderr)
        return 2

    out_dir = ensure_output_dir(args.output_dir)
    print(f"Writing outputs to: {out_dir}")

    cache: Optional[PromptCache] = None
    if not args.no_cache:
        cache_dir = (args.cache_dir or "").strip()
        if cache_dir:
            try:
                cache = PromptCache(Path(cache_dir))
            except Exception as exc:
                print(f"Warning: prompt cache disabled ({exc})")
                cache = None

    try:
        # Handle streaming options
        stream_enabled = args.stream and not args.no_stream
        
        run_workflow_v2(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            story_prompt=story_prompt,
            n_chapters=args.n_chapters,
            interactive=not args.non_interactive,
            out_dir=out_dir,
            stream=stream_enabled,
            temperature=args.temperature,
            cache=cache,
            context_strategy=args.context_strategy,
            summary_length=args.summary_length,
            recent_chapters=args.recent_chapters,
            max_context_tokens=args.max_context_tokens,
        )
    except Exception as e:
        print(f"Error during workflow: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
