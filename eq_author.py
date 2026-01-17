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

CHAPTER_WORD_TARGET = 2500
CHAPTER_MIN_WORDS = CHAPTER_WORD_TARGET
CHAPTER_MAX_ATTEMPTS = 5

WORD_RE = re.compile(r"\b[\w'\-]+\b")

# Context management constants
DEFAULT_CONTEXT_STRATEGY = "aggressive"
DEFAULT_SUMMARY_LENGTH = 250
DEFAULT_RECENT_CHAPTERS = 2
CONTEXT_WARNING_THRESHOLD = 0.75  # Warn at 75%
CONTEXT_CRITICAL_THRESHOLD = 0.90  # Critical at 90%

# Model-specific context limits
DEFAULT_MAX_CONTEXT_TOKENS = 8000  # Default for most models
DEEPSEEK_REASONER_MAX_TOKENS = 128000  # Max for deepseek-reasoner
DEEPSEEK_REASONER_DEFAULT_TOKENS = 32000  # Default for deepseek-reasoner

# Model-specific temperature defaults
DEFAULT_TEMPERATURE = 1.0  # Default for most models
DEEPSEEK_REASONER_TEMPERATURE = 1.0  # Default for deepseek-reasoner


class ContextManager:
    """Manages conversation context to prevent overflow in limited-context models."""
    
    def __init__(self, strategy: str = DEFAULT_CONTEXT_STRATEGY, summary_length: int = DEFAULT_SUMMARY_LENGTH,
                 recent_chapters: int = DEFAULT_RECENT_CHAPTERS, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS):
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
        
    def add_chapter(self, chapter_num: int, chapter_text: str, summary: str, ending: str = "") -> None:
        """Add a new chapter and its summary to the context manager."""
        self.chapter_summaries.append({
            "chapter": chapter_num,
            "summary": summary,
            "ending": ending,
        })
        
        # Keep recent full chapters if strategy allows
        if self.strategy == "balanced":
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
        
        return context
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (approximately 4 characters per token)."""
        return len(text) // 4
    
    def get_previous_chapter_ending(self, current_chapter: int) -> str:
        """Get the ending text of the previous chapter."""
        for summary_info in reversed(self.chapter_summaries):
            if summary_info["chapter"] == current_chapter - 1:
                return summary_info.get("ending", "")
        return ""
    
    def check_context_size(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Check if context is approaching limits and return status."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = self.estimate_tokens("".join(msg.get("content", "") for msg in messages))
        
        usage_ratio = estimated_tokens / self.max_context_tokens
        
        suggestions = []
        if self.recent_chapters > 1:
            suggestions.append("reduce --recent-chapters to 1")
        if self.summary_length > 200:
            suggestions.append("reduce --summary-length to 200")
        
        return {
            "estimated_tokens": estimated_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_ratio": usage_ratio,
            "is_warning": usage_ratio > CONTEXT_WARNING_THRESHOLD,
            "is_critical": usage_ratio > CONTEXT_CRITICAL_THRESHOLD,
            "suggestions": suggestions
        }
    
    def truncate_context(self, target_ratio: float = 0.85) -> None:
        """Reduce context size to fit within target ratio."""
        current_ratio = self.check_context_size(self.build_context(999))["usage_ratio"]
        
        if current_ratio <= target_ratio:
            return
        
        if len(self.chapter_summaries) > 1:
            self.chapter_summaries.pop(0)
        
        if len(self.recent_full_chapters) > 1:
            self.recent_full_chapters.pop(0)
            self.summary_length = max(100, self.summary_length - 50)


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


def auto_review_chapter(
    client: Any,
    model: str,
    chapter_text: str,
    chapter_num: int,
    previous_chapter_ending: str = "",
    stream: bool = False,
    temperature: Optional[float] = None,
    cache: Optional['PromptCache'] = None,
) -> tuple[str, str, bool]:
    """Have the LLM review and optionally rewrite a chapter.

    Returns a tuple of (review_notes, rewritten_text, llm_approved).
    - llm_approved: True if the LLM says the chapter is acceptable (final authority)

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
6. **Continuity**: Does it flow naturally from previous chapter?
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
        {"role": "system", "content": "You are a professional book editor. Your judgment is final - approve good chapters even if they're shorter than the target. Only recommend rewrite for significant quality issues."},
        {"role": "user", "content": review_prompt}
    ]

    review = chat_once(
        client,
        model,
        messages,
        stream=stream,
        temperature=temperature,
        cache=cache
    )

    # Check for acceptance
    upper_review = review.upper()
    is_acceptable = (
        "VERDICT" in upper_review and "ACCEPTABLE" in upper_review and "NEEDS_IMPROVEMENT" not in upper_review[:upper_review.find("VERDICT")+50]
    ) or "ACCEPTABLE - CHAPTER APPROVED" in upper_review
    is_needs_improvement = "NEEDS_IMPROVEMENT" in upper_review and "ACCEPTABLE" not in upper_review[:upper_review.find("NEEDS_IMPROVEMENT")+50]

    # Also check "Rewrite Recommendation" section
    if "RECOMMENDATION" in upper_review:
        recap_start = upper_review.find("RECOMMENDATION")
        recap_section = upper_review[recap_start:recap_start + 200]
        is_acceptable = is_acceptable or "ACCEPTABLE" in recap_section
        is_needs_improvement = is_needs_improvement or "NEEDS_IMPROVEMENT" in recap_section

    if is_acceptable and not is_needs_improvement:
        return review, chapter_text, True

    if is_needs_improvement:
        # Extract the rewritten chapter
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
            # Fallback: use original if extraction failed
            rewritten_text = chapter_text
        return review, rewritten_text, False

    # Default: if verdict is unclear, accept the chapter (LLM is authoritative)
    return review, chapter_text, True


def count_words(text: str) -> int:
    """Return an approximate word count suitable for prose validation."""
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def extract_chapter_ending(chapter_text: str, max_sentences: int = 2) -> str:
    """Extract the last 1-2 sentences from a chapter to track the ending."""
    sentences = re.split(r'(?<=[.!?])\s+', chapter_text.strip())
    if len(sentences) <= max_sentences:
        return chapter_text.strip()
    return ' '.join(sentences[-max_sentences:]).strip()


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


def is_adult_target(story_text: str) -> bool:
    """Return True if the story text/metadata indicates an adult target audience."""
    if not story_text:
        return False
    lower = story_text.lower()

    # Check common metadata lines like 'Target Audience: Adults' or genre containing 'adult'
    for line in story_text.splitlines():
        l = line.lower().strip()
        if "target audience" in l and ("adult" in l or "adults" in l or "18+" in l):
            return True
        if l.startswith("**target audience:**") and ("adult" in l or "adults" in l or "18+" in l):
            return True

    # Fallback: check for the word 'adult' or 'erotic' in genre or anywhere near the top
    head = "\n".join(story_text.splitlines()[:10]).lower()
    if "adult" in head or "erotic" in head:
        return True

    return False


def load_prompt_override_file() -> str:
    """Load the project-level `prompt_override.txt` if present next to this script.

    Returns empty string if file is missing or unreadable.
    """
    try:
        base = Path(__file__).parent
        p = base / "prompt_override.txt"
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def strip_injection_markers(text: str) -> str:
    """Remove any lines that begin with [^420] or similar injection markers."""
    lines = text.splitlines(keepends=True)
    filtered = [line for line in lines if not line.strip().startswith("[^")]
    result = "".join(filtered)
    # Also remove [^XXX] patterns anywhere in the text
    import re
    result = re.sub(r'\[\^[0-9]+\]', '', result)
    return result.strip()


def strip_thinking_sections(text: str) -> str:
    """Remove thinking/planning sections from LLM output.

    This function strips out the initial planning/thinking content that appears
    before the actual markdown content. The thinking sections typically contain
    meta-commentary, planning notes, and instructions. The actual content usually
    starts with a proper heading like "# Heading" or "Chapter X".

    Returns the content starting from the first real heading or the original text
    if no clear thinking section is detected.
    """
    if not text:
        return text

    lines = text.splitlines()

    # Patterns that indicate the start of actual content (headings)
    heading_patterns = [
        r'^#\s+',           # Markdown headings: # Title
        r'^##\s+',          # ## Title
        r'^Chapter\s+\d+',  # Chapter X
        r'^Part\s+\d+',     # Part X
    ]

    content_start_index = 0
    found_content = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines at the start
        if not stripped:
            continue

        # Check if this line matches a content heading pattern
        for pattern in heading_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                content_start_index = i
                found_content = True
                break

        if found_content:
            break

    # If we found content heading, return from that point
    if found_content:
        return "\n".join(lines[content_start_index:]).strip()

    # If no clear heading found, return original text
    return text


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
        "Ok now with these considerations in mind, formulate the final plan for the a human like, compelling short piece in {n_chapters} chapters. Bear in mind the constraints of the piece (each chapter is just {word_target} words). "
        "Above all things, the plan must serve the original prompt. We will use the same format as before:\n"
        "# Intention\n"
        "<State your formulated intentions for the piece, synthesised from the the parts of the brainstorming session that worked, and avoiding the parts that didn't. "
        "Be explicit about the choices you have made about plot, voice, stylistic choices, things you intend to aim for & avoid.>\n"
        "# Chapter Planning\n"
        "<Write a brief chapter plan for all {n_chapters} chapters.>"
    ).format(n_chapters=n_chapters, word_target=CHAPTER_WORD_TARGET)

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
    content = strip_injection_markers(content)
    content = strip_thinking_sections(content)
    p.write_text(content, encoding="utf-8")
    return p


def write_chapter_output(out_dir: Path, chapter_index: int, content: str) -> Path:
    p = out_dir / "chapters" / f"chapter_{chapter_index:02d}.md"
    content = strip_injection_markers(content)
    content = strip_thinking_sections(content)
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


def chapter_prompt(i: int, previous_chapter_ending: Optional[str] = None) -> str:
    base_intro = (
        f"Write Chapter {i} of the story, following the approved plan and prior chapters.\n"
        f"- Produce at least {CHAPTER_MIN_WORDS} words of narrative prose.\n"
        "- Count only the words in your final story text; do not include planning notes or analysis.\n"
        "- Output only the polished chapter text (you may open with a 'Chapter {i}' heading if that matches the style), and do not mention the word count or include any commentary.\n"
    )
    
    if i == 1:
        return (
            "Great. Let's begin the story.\n" + base_intro
        )
    
    # For subsequent chapters, add explicit instructions to avoid overlap
    overlap_instructions = (
        "\nIMPORTANT: Begin this chapter at a natural point after the previous chapter ended.\n"
        "- DO NOT repeat or restate events that just occurred at the end of the previous chapter.\n"
        "- Assume the reader has just finished the previous chapter and start with new content.\n"
        "- If referencing the previous chapter's ending, do so subtly without retelling the events.\n"
        "- Create a clear boundary between chapters - each chapter should feel distinct.\n"
        "- Think of this as a TV show episode that picks up after the previous one ended, not by replaying the last scene.\n"
    )
    
    if previous_chapter_ending:
        overlap_instructions += (
            f"\nNOTE: The previous chapter ended with: {previous_chapter_ending}\n"
            f"Start this chapter AFTER this moment, not by repeating it."
        )
    
    return (
        f"Continue with the next installment.\n" + base_intro + overlap_instructions
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


def get_default_max_context_tokens(model: str) -> int:
    """Return the appropriate default max_context_tokens based on the model."""
    if model.lower() == "deepseek-reasoner":
        return DEEPSEEK_REASONER_DEFAULT_TOKENS
    return DEFAULT_MAX_CONTEXT_TOKENS


def get_default_temperature(model: str) -> float:
    """Return the appropriate default temperature based on the model."""
    if model.lower() == "deepseek-reasoner":
        return DEEPSEEK_REASONER_TEMPERATURE
    return DEFAULT_TEMPERATURE


def detect_progress(out_dir: Path) -> Dict[str, Any]:
    """Detect the current progress from an output directory.

    Returns a dict with:
    - last_step: highest completed step (1-5), or 0 if only step 1 exists
    - last_chapter: highest completed chapter number, or 0 if no chapters
    - n_chapters: total number of chapters (from step 4 if available)
    - has_all_steps: whether all steps 1-5 are complete
    """
    result = {
        "last_step": 0,
        "last_chapter": 0,
        "n_chapters": None,
        "has_all_steps": False,
    }

    # Check for step files
    max_step = 0
    for step_num in range(1, 6):
        step_file = out_dir / STEP_FILENAMES.get(step_num, f"step_{step_num:02d}.md")
        if step_file.exists():
            max_step = step_num

    result["last_step"] = max_step
    result["has_all_steps"] = max_step >= 5

    # If step 4 exists, try to extract chapter count
    if max_step >= 4:
        step4_file = out_dir / STEP_FILENAMES[4]
        try:
            step4_content = step4_file.read_text(encoding="utf-8")
            # Look for chapter count in the format we expect
            for line in step4_content.splitlines():
                if "chapter" in line.lower() and ":" in line:
                    parts = line.split(":")
                    try:
                        num = int(parts[-1].strip())
                        result["n_chapters"] = num
                        break
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

    # Check for chapters
    chapters_dir = out_dir / "chapters"
    if chapters_dir.exists():
        max_chapter = 0
        for f in chapters_dir.iterdir():
            if f.name.startswith("chapter_") and f.name.endswith(".md"):
                try:
                    num = int(f.name[8:10])
                    max_chapter = max(max_chapter, num)
                except (ValueError, IndexError):
                    continue
        result["last_chapter"] = max_chapter

    return result


def load_existing_context(out_dir: Path, progress: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load the existing conversation context from previous outputs.

    Returns a list of messages ready to continue the conversation.
    """
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": "You are a helpful assistant"})

    # Load all completed steps in order
    for step_num in range(1, 6):
        step_file = out_dir / STEP_FILENAMES.get(step_num, f"step_{step_num:02d}.md")
        if step_file.exists():
            try:
                content = step_file.read_text(encoding="utf-8")
                messages.append({"role": "assistant", "content": content})
            except Exception:
                pass

    # If we have chapters, add them as assistant messages for context
    last_chapter = progress.get("last_chapter", 0)
    for i in range(1, last_chapter + 1):
        chapter_file = out_dir / "chapters" / f"chapter_{i:02d}.md"
        if chapter_file.exists():
            try:
                content = chapter_file.read_text(encoding="utf-8")
                messages.append({"role": "assistant", "content": content})
            except Exception:
                pass

    return messages


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
    context_strategy: str = DEFAULT_CONTEXT_STRATEGY,
    summary_length: int = DEFAULT_SUMMARY_LENGTH,
    recent_chapters: int = DEFAULT_RECENT_CHAPTERS,
    max_context_tokens: Optional[int] = None,
    always_autogen_chapters: bool = False,
) -> None:
    """Generate chapters using existing planning files (steps 1-5 must exist in out_dir).

    This function skips the planning workflow and uses existing planning files
    to generate chapters directly. Useful for when you've prepared planning
    files manually or want to regenerate chapters with different settings.
    """
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
        return ""

    # Load existing planning files
    progress = detect_progress(out_dir)
    last_step = progress["last_step"]
    last_chapter = progress["last_chapter"]

    # Check that planning files exist
    if last_step < 5:
        raise RuntimeError(
            f"Planning files incomplete. Found steps 1-{last_step}, but need all steps 1-5. "
            f"Use --resume-from with a directory containing complete planning files, "
            f"or run without --skip-planning to generate planning files first."
        )

    print(f"Using existing planning files from {out_dir}")
    print(f"Detected {last_chapter} existing chapters")

    # Load existing context from planning files
    messages = load_existing_context(out_dir, progress)
    print(f"Loaded {len(messages)} messages from planning files")

    # Use model-specific default if max_context_tokens is not provided
    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    # Initialize context manager
    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens
    )

    # Add core context from planning files
    context_manager.add_core_context(messages)

    # Rebuild context manager state from existing chapters
    if last_chapter > 0:
        for i in range(1, last_chapter + 1):
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

    # Determine chapter count from step 4
    n_chapters = progress.get("n_chapters")
    if n_chapters is None:
        # Try to extract from step 4 content
        step4_file = out_dir / STEP_FILENAMES[4]
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
            raise RuntimeError(
                "Could not determine chapter count from planning files. "
                "Ensure step 4 contains a chapter count (e.g., 'CHAPTER_COUNT: 10')."
            )

    print(f"Generating chapters {last_chapter + 1} through {n_chapters}")
    
    # Chapter writing: chapter last_chapter+1 .. N
    for i in range(last_chapter + 1, n_chapters + 1):
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

        attempts = 0
        final_text = ""
        last_response = ""
        last_word_count = 0

        while attempts < CHAPTER_MAX_ATTEMPTS:
            ch = chat_once(
                client, model, context_messages,
                stream=stream, temperature=temperature,
                on_token=token_handler if stream else None,
                cache=cache,
            )
            if stream:
                print("\n", flush=True)

            last_response = ch
            last_word_count = count_words(ch)

            if last_word_count >= CHAPTER_MIN_WORDS:
                final_text = ch
                break

            attempts += 1
            if stream:
                print(f"Chapter {i} returned {last_word_count} words (needs >= {CHAPTER_MIN_WORDS}).", flush=True)

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
            print(f"Warning: Chapter {i} final word count {last_word_count} < {CHAPTER_MIN_WORDS} after {CHAPTER_MAX_ATTEMPTS} attempt(s).", file=sys.stderr, flush=True)
        else:
            print(f"Chapter {i} word count: {last_word_count}", flush=True)

        # Auto-review loop: LLM reviews and potentially rewrites chapter
        if always_autogen_chapters:
            print(f"\n=== Auto-Reviewing Chapter {i} ===", flush=True)
            review_count = 0
            max_auto_reviews = 2
            llm_approved = False
            while review_count < max_auto_reviews and not llm_approved:
                previous_ending = context_manager.get_previous_chapter_ending(i) if i > 1 else ""
                review_notes, reviewed_text, llm_approved = auto_review_chapter(
                    client, model, final_text, i, previous_ending, stream, temperature, cache
                )

                review_file = out_dir / "chapters" / f"chapter_{i:02d}_review.md"
                (out_dir / "chapters").mkdir(parents=True, exist_ok=True)
                review_file.write_text(review_notes, encoding="utf-8")

                if reviewed_text != final_text:
                    print(f"Chapter {i} auto-rewritten based on review.", flush=True)
                    final_text = reviewed_text
                    last_word_count = count_words(final_text)
                    print(f"New word count: {last_word_count}", flush=True)
                    review_count += 1
                else:
                    print(f"Chapter {i} review: No rewrite needed.", flush=True)
                    break

        write_chapter_output(out_dir, i, final_text)

        chapter_ending = extract_chapter_ending(final_text)

        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(
            client, model, final_text, i, summary_length, stream, temperature, cache
        )
        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")

        context_manager.add_chapter(i, final_text, chapter_summary, chapter_ending)

        if not always_autogen_chapters:
            fb = maybe_collect_feedback(f"chapter {i}")
            if fb:
                context_messages.append({"role": "user", "content": f"FEEDBACK AFTER CHAPTER {i}:\n{fb}\nPlease apply this feedback in the next chapter(s)."})


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
    context_strategy: str = DEFAULT_CONTEXT_STRATEGY,
    summary_length: int = DEFAULT_SUMMARY_LENGTH,
    recent_chapters: int = DEFAULT_RECENT_CHAPTERS,
    max_context_tokens: Optional[int] = None,
    always_autogen_chapters: bool = False,
    n_chapters: Optional[int] = None,
) -> None:
    """Resume a previous workflow run from where it left off."""
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
        return ""

    # Detect current progress
    progress = detect_progress(out_dir)
    last_step = progress["last_step"]
    last_chapter = progress["last_chapter"]

    print(f"Detected progress: Steps 1-{last_step} complete, Chapters 1-{last_chapter} complete")

    # Use provided n_chapters, or detect from previous run, or use default
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

    # Ensure n_chapters is an int
    n_chapters = int(detected_n_chapters)

    # Load existing context
    messages = load_existing_context(out_dir, progress)
    print(f"Loaded {len(messages)} messages from previous run")

    # Use model-specific default if max_context_tokens is not provided
    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    # Initialize context manager
    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens
    )

    # Add core context from previous messages
    context_manager.add_core_context(messages)

    # Rebuild context manager state from existing chapters
    if last_chapter > 0:
        for i in range(1, last_chapter + 1):
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

    # Resume from where we left off
    if last_step < 1:
        # Re-run step 1
        step1 = build_step1_prompt().replace("{STORY_PROMPT}", story_prompt)
        messages.append({"role": "user", "content": step1})
        if stream:
            print("\n=== Step 1: Brainstorm & Reflection (streaming) ===", flush=True)
        emit_progress("Step 1: Brainstorm & Reflection")
        s1 = chat_once(
            client, model, messages,
            stream=stream, temperature=temperature,
            on_token=token_handler if stream else None,
            cache=cache,
        )
        if stream:
            print("\n", flush=True)
        write_step_output(out_dir, 1, s1)
        messages.append({"role": "assistant", "content": s1})

        proposed = parse_proposed_chapters(s1)
        final_count = n_chapters or proposed

        while True:
            fb = maybe_collect_feedback(1)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 1:\n{fb}\nPlease apply this feedback and rewrite your output for Brainstorming & Reflection."})
                if stream:
                    print("\n=== Re-applying feedback for Step 1: Brainstorming & Reflection (streaming) ===", flush=True)
                s1 = chat_once(
                    client, model, messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)
                write_step_output(out_dir, 1, s1)
                messages[-1] = {"role": "assistant", "content": s1}
            else:
                break

        last_step = 1

    # Build follow-up prompts
    step_texts = build_followup_prompts(n_chapters)

    if last_step < 2:
        messages.append({"role": "user", "content": step_texts[0]})
        if stream:
            print("\n=== Step 2: Intention & Chapter Planning (streaming) ===", flush=True)
        emit_progress("Step 2: Intention & Chapter Planning")
        s2 = chat_once(
            client, model, messages,
            stream=stream, temperature=temperature,
            on_token=token_handler if stream else None,
            cache=cache,
        )
        if stream:
            print("\n", flush=True)
        write_step_output(out_dir, 2, s2)
        messages.append({"role": "assistant", "content": s2})

        while True:
            fb = maybe_collect_feedback(2)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 2:\n{fb}\nPlease apply this feedback and rewrite your output for Intention & Chapter Planning."})
                if stream:
                    print("\n=== Re-applying feedback for Step 2: Intention & Chapter Planning (streaming) ===", flush=True)
                s2 = chat_once(
                    client, model, messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)
                write_step_output(out_dir, 2, s2)
                messages[-1] = {"role": "assistant", "content": s2}
            else:
                break

        last_step = 2

    if last_step < 3:
        messages.append({"role": "user", "content": step_texts[1]})
        if stream:
            print("\n=== Step 3: Human vs LLM Critique (streaming) ===", flush=True)
        emit_progress("Step 3: Human vs LLM Critique")
        s3 = chat_once(
            client, model, messages,
            stream=stream, temperature=temperature,
            on_token=token_handler if stream else None,
            cache=cache,
        )
        if stream:
            print("\n", flush=True)
        write_step_output(out_dir, 3, s3)
        messages.append({"role": "assistant", "content": s3})

        while True:
            fb = maybe_collect_feedback(3)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 3:\n{fb}\nPlease apply this feedback and rewrite your output for Human vs LLM Critique."})
                if stream:
                    print("\n=== Re-applying feedback for Step 3: Human vs LLM Critique (streaming) ===", flush=True)
                s3 = chat_once(
                    client, model, messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)
                write_step_output(out_dir, 3, s3)
                messages[-1] = {"role": "assistant", "content": s3}
            else:
                break

        last_step = 3

    if last_step < 4:
        messages.append({"role": "user", "content": step_texts[2]})
        if stream:
            print("\n=== Step 4: Final Plan (streaming) ===", flush=True)
        emit_progress("Step 4: Final Plan")
        s4 = chat_once(
            client, model, messages,
            stream=stream, temperature=temperature,
            on_token=token_handler if stream else None,
            cache=cache,
        )
        if stream:
            print("\n", flush=True)
        write_step_output(out_dir, 4, s4)
        messages.append({"role": "assistant", "content": s4})

        while True:
            fb = maybe_collect_feedback(4)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 4:\n{fb}\nPlease apply this feedback and rewrite your output for Final Plan."})
                if stream:
                    print("\n=== Re-applying feedback for Step 4: Final Plan (streaming) ===", flush=True)
                s4 = chat_once(
                    client, model, messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)
                write_step_output(out_dir, 4, s4)
                messages[-1] = {"role": "assistant", "content": s4}
            else:
                break

        last_step = 4

    if last_step < 5:
        messages.append({"role": "user", "content": step_texts[3]})
        if stream:
            print("\n=== Step 5: Characters (streaming) ===", flush=True)
        emit_progress("Step 5: Characters")
        s5 = chat_once(
            client, model, messages,
            stream=stream, temperature=temperature,
            on_token=token_handler if stream else None,
            cache=cache,
        )
        if stream:
            print("\n", flush=True)
        write_step_output(out_dir, 5, s5)
        messages.append({"role": "assistant", "content": s5})

        while True:
            fb = maybe_collect_feedback(5)
            if fb:
                messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 5:\n{fb}\nPlease apply this feedback and rewrite your output for Characters."})
                if stream:
                    print("\n=== Re-applying feedback for Step 5: Characters (streaming) ===", flush=True)
                s5 = chat_once(
                    client, model, messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)
                write_step_output(out_dir, 5, s5)
                messages[-1] = {"role": "assistant", "content": s5}
            else:
                break

        last_step = 5

    # Update core context
    context_manager.add_core_context(messages)

    # Continue with chapters
    for i in range(last_chapter + 1, n_chapters + 1):
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

        attempts = 0
        final_text = ""
        last_response = ""
        last_word_count = 0

        while attempts < CHAPTER_MAX_ATTEMPTS:
            ch = chat_once(
                client, model, context_messages,
                stream=stream, temperature=temperature,
                on_token=token_handler if stream else None,
                cache=cache,
            )
            if stream:
                print("\n", flush=True)

            last_response = ch
            last_word_count = count_words(ch)

            if last_word_count >= CHAPTER_MIN_WORDS:
                final_text = ch
                break

            attempts += 1
            if stream:
                print(f"Chapter {i} returned {last_word_count} words (needs >= {CHAPTER_MIN_WORDS}).", flush=True)

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
            print(f"Warning: Chapter {i} final word count {last_word_count} < {CHAPTER_MIN_WORDS} after {CHAPTER_MAX_ATTEMPTS} attempt(s).", file=sys.stderr, flush=True)
        else:
            print(f"Chapter {i} word count: {last_word_count}", flush=True)

        # Feedback loop: allow re-writing chapter based on feedback
        if not always_autogen_chapters:
            feedback_loop_count = 0
            max_feedback_loops = 3
            while feedback_loop_count < max_feedback_loops:
                fb = maybe_collect_feedback(f"chapter {i}")
                if not fb:
                    break

                print(f"\n=== Re-writing Chapter {i} with feedback (attempt {feedback_loop_count + 1}) ===", flush=True)
                feedback_prompt = (
                    f"Please rewrite Chapter {i} incorporating this feedback:\n\n"
                    f"FEEDBACK: {fb}\n\n"
                    f"Previous chapter text:\n{final_text[:2000]}...\n\n"
                    "Rewrite the entire chapter with the feedback applied. "
                    "Maintain continuity with previous chapters. "
                    "Output only the revised chapter text with no commentary."
                )
                rewrite_messages = context_manager.build_context(i).copy()
                rewrite_messages.append({"role": "user", "content": chapter_prompt(i, context_manager.get_previous_chapter_ending(i))})
                rewrite_messages.append({"role": "user", "content": feedback_prompt})

                ch = chat_once(
                    client, model, rewrite_messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)

                new_word_count = count_words(ch)
                if new_word_count >= CHAPTER_MIN_WORDS:
                    final_text = ch
                    last_word_count = new_word_count
                    print(f"Chapter {i} rewritten. Word count: {last_word_count}", flush=True)
                else:
                    print(f"Warning: Rewritten chapter has {new_word_count} words (needs >= {CHAPTER_MIN_WORDS})", flush=True)

                context_messages.append({"role": "user", "content": f"FEEDBACK FOR CHAPTER {i}:\n{fb}\nPlease apply this feedback."})
                context_messages.append({"role": "assistant", "content": final_text})

                feedback_loop_count += 1

        # Auto-review loop: LLM reviews and potentially rewrites chapter
        if always_autogen_chapters:
            print(f"\n=== Auto-Reviewing Chapter {i} ===", flush=True)
            review_count = 0
            max_auto_reviews = 2
            llm_approved = False
            while review_count < max_auto_reviews and not llm_approved:
                previous_ending = context_manager.get_previous_chapter_ending(i) if i > 1 else ""
                review_notes, reviewed_text, llm_approved = auto_review_chapter(
                    client, model, final_text, i, previous_ending, stream, temperature, cache
                )

                review_file = out_dir / "chapters" / f"chapter_{i:02d}_review.md"
                (out_dir / "chapters").mkdir(parents=True, exist_ok=True)
                review_file.write_text(review_notes, encoding="utf-8")

                if reviewed_text != final_text:
                    print(f"Chapter {i} auto-rewritten based on review.", flush=True)
                    final_text = reviewed_text
                    last_word_count = count_words(final_text)
                    print(f"New word count: {last_word_count}", flush=True)
                    review_count += 1
                else:
                    print(f"Chapter {i} review: No rewrite needed.", flush=True)
                    break

        write_chapter_output(out_dir, i, final_text)

        chapter_ending = extract_chapter_ending(final_text)

        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(
            client, model, final_text, i, summary_length, stream, temperature, cache
        )
        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")

        context_manager.add_chapter(i, final_text, chapter_summary, chapter_ending)

        if not always_autogen_chapters:
            fb = maybe_collect_feedback(f"chapter {i}")
            if fb:
                context_messages.append({"role": "user", "content": f"FEEDBACK AFTER CHAPTER {i}:\n{fb}\nPlease apply this feedback in the next chapter(s)."})


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

    # Use model-specific default if max_context_tokens is not provided
    if max_context_tokens is None:
        max_context_tokens = get_default_max_context_tokens(model)

    # Initialize context manager
    context_manager = ContextManager(
        strategy=context_strategy,
        summary_length=summary_length,
        recent_chapters=recent_chapters,
        max_context_tokens=max_context_tokens
    )

    # Prepare initial system message(s). If the story prompt indicates an adult target,
    # prepend the contents of `prompt_override.txt` (if present) to the system prompt.
    messages: List[Dict[str, str]] = []
    try:
        override = ""
        if is_adult_target(story_prompt):
            override = load_prompt_override_file() or ""

        if override:
            # Prepend override content before the default assistant system message
            messages.append({"role": "system", "content": override})

            # CLI indicator for override usage and persist the override to the output directory
            try:
                print("[PROMPT OVERRIDE] Applied prompt_override.txt (adult target detected)", flush=True)
                # write a copy of the override into the run output dir for traceability
                try:
                    (out_dir / "prompt_override_used.txt").write_text(override, encoding="utf-8")
                except Exception:
                    # If out_dir isn't available or write fails, ignore silently
                    pass
            except Exception:
                pass
        else:
            # Adult target detected but no override content available
            try:
                print("[PROMPT OVERRIDE] Detected adult target but `prompt_override.txt` is missing or empty", flush=True)
                try:
                    (out_dir / "prompt_override_missing.txt").write_text("(no override file present)", encoding="utf-8")
                except Exception:
                    pass
            except Exception:
                pass

    except Exception:
        # If anything goes wrong reading override, just fall back to default
        pass

    # Always include the standard system instruction
    messages.append({"role": "system", "content": "You are a helpful assistant"})

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

    # Handle feedback loop for step 1
    while True:
        fb = maybe_collect_feedback(1)
        if fb:
            messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 1:\n{fb}\nPlease apply this feedback and rewrite your output for Brainstorming & Reflection."})
            if stream:
                print("\n=== Re-applying feedback for Step 1: Brainstorming & Reflection (streaming) ===", flush=True)
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
            messages[-1] = {"role": "assistant", "content": s1}
        else:
            break

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

    # Handle feedback loop for step 2
    while True:
        fb = maybe_collect_feedback(2)
        if fb:
            messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 2:\n{fb}\nPlease apply this feedback and rewrite your output for Intention & Chapter Planning."})
            if stream:
                print("\n=== Re-applying feedback for Step 2: Intention & Chapter Planning (streaming) ===", flush=True)
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
            messages[-1] = {"role": "assistant", "content": s2}
        else:
            break

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

    # Handle feedback loop for step 3
    while True:
        fb = maybe_collect_feedback(3)
        if fb:
            messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 3:\n{fb}\nPlease apply this feedback and rewrite your output for Human vs LLM Critique."})
            if stream:
                print("\n=== Re-applying feedback for Step 3: Human vs LLM Critique (streaming) ===", flush=True)
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
            messages[-1] = {"role": "assistant", "content": s3}
        else:
            break

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

    # Handle feedback loop for step 4
    while True:
        fb = maybe_collect_feedback(4)
        if fb:
            messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 4:\n{fb}\nPlease apply this feedback and rewrite your output for Final Plan."})
            if stream:
                print("\n=== Re-applying feedback for Step 4: Final Plan (streaming) ===", flush=True)
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
            messages[-1] = {"role": "assistant", "content": s4}
        else:
            break

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

    # Handle feedback loop for step 5
    while True:
        fb = maybe_collect_feedback(5)
        if fb:
            messages.append({"role": "user", "content": f"FEEDBACK FOR STEP 5:\n{fb}\nPlease apply this feedback and rewrite your output for Characters."})
            if stream:
                print("\n=== Re-applying feedback for Step 5: Characters (streaming) ===", flush=True)
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
            messages[-1] = {"role": "assistant", "content": s5}
        else:
            break

    # Store core planning context in the context manager
    context_manager.add_core_context(messages)

    # Chapter writing: chapter 1..N
    for i in range(1, final_count + 1):
        # Get previous chapter ending for context
        previous_chapter_ending = context_manager.get_previous_chapter_ending(i) if i > 1 else ""
        
        # Build context for current chapter
        context_messages = context_manager.build_context(i)
        context_messages.append({"role": "user", "content": chapter_prompt(i, previous_chapter_ending)})
        
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

        # Feedback loop: allow re-writing chapter based on feedback
        if not always_autogen_chapters:
            feedback_loop_count = 0
            max_feedback_loops = 3
            while feedback_loop_count < max_feedback_loops:
                fb = maybe_collect_feedback(f"chapter {i}")
                if not fb:
                    break

                print(f"\n=== Re-writing Chapter {i} with feedback (attempt {feedback_loop_count + 1}) ===", flush=True)
                feedback_prompt = (
                    f"Please rewrite Chapter {i} incorporating this feedback:\n\n"
                    f"FEEDBACK: {fb}\n\n"
                    f"Previous chapter text:\n{final_text[:2000]}...\n\n"
                    "Rewrite the entire chapter with the feedback applied. "
                    "Maintain continuity with previous chapters. "
                    "Output only the revised chapter text with no commentary."
                )
                rewrite_messages = context_manager.build_context(i).copy()
                rewrite_messages.append({"role": "user", "content": chapter_prompt(i, context_manager.get_previous_chapter_ending(i))})
                rewrite_messages.append({"role": "user", "content": feedback_prompt})

                ch = chat_once(
                    client, model, rewrite_messages,
                    stream=stream, temperature=temperature,
                    on_token=token_handler if stream else None,
                    cache=cache,
                )
                if stream:
                    print("\n", flush=True)

                new_word_count = count_words(ch)
                if new_word_count >= CHAPTER_MIN_WORDS:
                    final_text = ch
                    last_word_count = new_word_count
                    print(f"Chapter {i} rewritten. Word count: {last_word_count}", flush=True)
                else:
                    print(f"Warning: Rewritten chapter has {new_word_count} words (needs >= {CHAPTER_MIN_WORDS})", flush=True)

                context_messages.append({"role": "user", "content": f"FEEDBACK FOR CHAPTER {i}:\n{fb}\nPlease apply this feedback."})
                context_messages.append({"role": "assistant", "content": final_text})

                feedback_loop_count += 1

        # Auto-review loop: LLM reviews and potentially rewrites chapter
        if always_autogen_chapters:
            print(f"\n=== Auto-Reviewing Chapter {i} ===", flush=True)
            review_count = 0
            max_auto_reviews = 2
            llm_approved = False
            while review_count < max_auto_reviews and not llm_approved:
                previous_ending = context_manager.get_previous_chapter_ending(i) if i > 1 else ""
                review_notes, reviewed_text, llm_approved = auto_review_chapter(
                    client, model, final_text, i, previous_ending, stream, temperature, cache
                )

                review_file = out_dir / "chapters" / f"chapter_{i:02d}_review.md"
                (out_dir / "chapters").mkdir(parents=True, exist_ok=True)
                review_file.write_text(review_notes, encoding="utf-8")

                if reviewed_text != final_text:
                    print(f"Chapter {i} auto-rewritten based on review.", flush=True)
                    final_text = reviewed_text
                    last_word_count = count_words(final_text)
                    print(f"New word count: {last_word_count}", flush=True)
                    review_count += 1
                else:
                    print(f"Chapter {i} review: No rewrite needed.", flush=True)
                    break

        write_chapter_output(out_dir, i, final_text)

        # Extract chapter ending before adding to context
        chapter_ending = extract_chapter_ending(final_text)

        # Generate chapter summary for context management
        if stream:
            print(f"\n=== Generating Chapter {i} Summary ===", flush=True)
        chapter_summary = generate_chapter_summary(
            client, model, final_text, i, summary_length, stream, temperature, cache
        )
        
        # Save summary to file for reference
        summary_file = out_dir / "chapters" / f"chapter_{i:02d}_summary.md"
        summary_file.write_text(chapter_summary, encoding="utf-8")
        
        # Add chapter to context manager with ending
        context_manager.add_chapter(i, final_text, chapter_summary, chapter_ending)

        if not always_autogen_chapters:
            fb = maybe_collect_feedback(f"chapter {i}")
            if fb:
                context_messages.append({"role": "user", "content": f"FEEDBACK AFTER CHAPTER {i}:\n{fb}\nPlease apply this feedback in the next chapter(s)."})


def get_env_var_for_arg(arg_name: str) -> Optional[str]:
    """Convert argument name to env var name and return its value."""
    env_var = f"EQ_AUTHOR_{arg_name.upper().replace('-', '_')}"
    return os.getenv(env_var)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EQ-Author DeepSeek Planner & Writer")
    p.add_argument("--story-file", type=str, help="Path to file containing the story idea", default=get_env_var_for_arg("story-file"))
    p.add_argument("--story-text", type=str, help="Story idea text (overrides file if provided)", default=get_env_var_for_arg("story-text"))
    p.add_argument("--n-chapters", type=int, help="Override: number of chapters to write (if omitted, model proposes)", default=None)
    env_n_chapters = get_env_var_for_arg("n-chapters")
    if env_n_chapters:
        try:
            p.set_defaults(n_chapters=int(env_n_chapters))
        except ValueError:
            pass
    p.add_argument("--output-dir", type=str, help="Base directory for outputs", default=get_env_var_for_arg("output-dir") or "outputs")
    p.add_argument("--api-key", type=str, help="DeepSeek API key (or set API_KEY or EQ_AUTHOR_API_KEY)", default=None)
    p.add_argument("--base-url", type=str, help="DeepSeek base URL", default=get_env_var_for_arg("base-url") or "https://api.deepseek.com")
    p.add_argument("--model", type=str, help="Model name", default=get_env_var_for_arg("model") or "deepseek-reasoner")
    p.add_argument("--stream", action="store_true", help="Stream responses (aggregated in output)", default=True)
    p.add_argument("--no-stream", action="store_true", help="Disable response streaming", default=False)
    env_no_stream = get_env_var_for_arg("no-stream")
    if env_no_stream and env_no_stream.lower() in ("true", "1", "yes"):
        p.set_defaults(no_stream=True)
    p.add_argument("--temperature", type=float, help=f"Sampling temperature (default: model-specific, {DEFAULT_TEMPERATURE} for most models, {DEEPSEEK_REASONER_TEMPERATURE} for deepseek-reasoner)", default=None)
    env_temp = get_env_var_for_arg("temperature")
    if env_temp:
        try:
            p.set_defaults(temperature=float(env_temp))
        except ValueError:
            pass
    p.add_argument("--non-interactive", action="store_true", help="Run without feedback prompts and chapter count confirmation", default=False)
    env_non_interactive = get_env_var_for_arg("non-interactive")
    if env_non_interactive and env_non_interactive.lower() in ("true", "1", "yes"):
        p.set_defaults(non_interactive=True)
    p.add_argument("--no-cache", action="store_true", help="Disable prompt caching", default=False)
    env_no_cache = get_env_var_for_arg("no-cache")
    if env_no_cache and env_no_cache.lower() in ("true", "1", "yes"):
        p.set_defaults(no_cache=True)
    p.add_argument(
        "--cache-dir",
        type=str,
        default=get_env_var_for_arg("cache-dir") or ".prompt_cache",
        help="Directory for prompt/response cache (ignored if --no-cache is used)",
    )
    
    # Context management options
    p.add_argument("--context-strategy", type=str, choices=["aggressive", "balanced"],
                   default=get_env_var_for_arg("context-strategy") or DEFAULT_CONTEXT_STRATEGY, help="Strategy for managing context window (default: aggressive)")
    p.add_argument("--summary-length", type=int, default=DEFAULT_SUMMARY_LENGTH,
                   help=f"Target word count for chapter summaries (default: {DEFAULT_SUMMARY_LENGTH})")
    env_summary = get_env_var_for_arg("summary-length")
    if env_summary:
        try:
            p.set_defaults(summary_length=int(env_summary))
        except ValueError:
            pass
    p.add_argument("--recent-chapters", type=int, default=DEFAULT_RECENT_CHAPTERS,
                   help=f"Number of recent full chapters to keep in context (default: {DEFAULT_RECENT_CHAPTERS})")
    env_recent = get_env_var_for_arg("recent-chapters")
    if env_recent:
        try:
            p.set_defaults(recent_chapters=int(env_recent))
        except ValueError:
            pass
    p.add_argument("--max-context-tokens", type=int, default=None,
                   help=f"Maximum context tokens to maintain (default: model-specific, {DEFAULT_MAX_CONTEXT_TOKENS} for most models, {DEEPSEEK_REASONER_DEFAULT_TOKENS} for deepseek-reasoner)")
    env_max_tokens = get_env_var_for_arg("max-context-tokens")
    if env_max_tokens:
        try:
            p.set_defaults(max_context_tokens=int(env_max_tokens))
        except ValueError:
            pass
    
    p.add_argument("--resume-from", type=str, default=None,
                   help="Resume from a previous run by providing the output directory path")
    
    p.add_argument("--skip-planning", action="store_true", default=False,
                   help="Skip planning steps (1-5) and use existing files from --output-dir for chapter generation")
    env_skip = get_env_var_for_arg("skip-planning")
    if env_skip and env_skip.lower() in ("true", "1", "yes"):
        p.set_defaults(skip_planning=True)
    
    p.add_argument("--always-autogen-chapters", action="store_true", default=False,
                   help="After completing planning steps (1-5), automatically generate all chapters without prompting for feedback between chapters")
    env_autogen = get_env_var_for_arg("always-autogen-chapters")
    if env_autogen and env_autogen.lower() in ("true", "1", "yes"):
        p.set_defaults(always_autogen_chapters=True)
    
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    # Load .env if present (root of project or current working dir)
    load_dotenv()

    args = parse_args(argv)

    # Handle resume mode
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        if not resume_dir.exists():
            print(f"Error: Resume directory not found: {resume_dir}", file=sys.stderr)
            return 2

        api_key = args.api_key or os.getenv("EQ_AUTHOR_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: Provide --api-key or set API_KEY/OPENAI_API_KEY", file=sys.stderr)
            return 2

        cache: Optional[PromptCache] = None
        if not args.no_cache:
            cache_dir = (args.cache_dir or "").strip()
            if cache_dir:
                try:
                    cache = PromptCache(Path(cache_dir))
                except Exception as exc:
                    print(f"Warning: prompt cache disabled ({exc})")
                    cache = None

        stream_enabled = args.stream and not args.no_stream
        temperature = args.temperature if args.temperature is not None else get_default_temperature(args.model)

        try:
            run_workflow_v2_resume(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                story_prompt="",  # Not needed for resume
                out_dir=resume_dir,
                stream=stream_enabled,
                temperature=temperature,
                cache=cache,
                context_strategy=args.context_strategy,
                summary_length=args.summary_length,
                recent_chapters=args.recent_chapters,
                max_context_tokens=args.max_context_tokens,
                always_autogen_chapters=args.always_autogen_chapters,
                n_chapters=args.n_chapters,
            )
        except Exception as e:
            print(f"Error during resume: {e}", file=sys.stderr)
            return 1

        print("Resume complete.")
        return 0

    api_key = args.api_key or os.getenv("EQ_AUTHOR_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
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
        
        # Use model-specific defaults if not provided
        temperature = args.temperature if args.temperature is not None else get_default_temperature(args.model)
        
        # Handle skip-planning mode
        if args.skip_planning:
            try:
                run_workflow_v2_skip_planning(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    out_dir=out_dir,
                    stream=stream_enabled,
                    temperature=temperature,
                    cache=cache,
                    context_strategy=args.context_strategy,
                    summary_length=args.summary_length,
                    recent_chapters=args.recent_chapters,
                    max_context_tokens=args.max_context_tokens,
                    always_autogen_chapters=args.always_autogen_chapters,
                )
            except Exception as e:
                print(f"Error during chapter generation: {e}", file=sys.stderr)
                return 1
        else:
            run_workflow_v2(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                story_prompt=story_prompt,
                n_chapters=args.n_chapters,
                interactive=not args.non_interactive,
                out_dir=out_dir,
                stream=stream_enabled,
                temperature=temperature,
                cache=cache,
                context_strategy=args.context_strategy,
                summary_length=args.summary_length,
                recent_chapters=args.recent_chapters,
                max_context_tokens=args.max_context_tokens,
                always_autogen_chapters=args.always_autogen_chapters,
            )
    except Exception as e:
        print(f"Error during workflow: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
