"""Context management for LLM conversations."""

from typing import List, Dict, Any
from .constants import (
    DEFAULT_CONTEXT_STRATEGY,
    DEFAULT_SUMMARY_LENGTH,
    DEFAULT_RECENT_CHAPTERS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    CONTEXT_WARNING_THRESHOLD,
    CONTEXT_CRITICAL_THRESHOLD,
)


class ContextManager:
    """Manages conversation context to prevent overflow in limited-context models."""

    def __init__(
        self,
        strategy: str = DEFAULT_CONTEXT_STRATEGY,
        summary_length: int = DEFAULT_SUMMARY_LENGTH,
        recent_chapters: int = DEFAULT_RECENT_CHAPTERS,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ):
        self.strategy = strategy
        self.summary_length = summary_length
        self.recent_chapters = recent_chapters
        self.max_context_tokens = max_context_tokens

        self.core_context = []
        self.chapter_summaries = []
        self.recent_full_chapters = []
        self.current_messages = []

    def add_core_context(self, messages: List[Dict[str, str]]) -> None:
        """Add core planning context that should always be preserved."""
        self.core_context = messages.copy()

    def add_chapter(self, chapter_num: int, chapter_text: str, summary: str, ending: str = "") -> None:
        """Add a new chapter and its summary to context manager."""
        self.chapter_summaries.append({"chapter": chapter_num, "summary": summary, "ending": ending})

        if self.strategy == "balanced":
            self.recent_full_chapters.append({"chapter": chapter_num, "text": chapter_text})
            if len(self.recent_full_chapters) > self.recent_chapters:
                self.recent_full_chapters.pop(0)

    def build_context(self, current_chapter: int) -> List[Dict[str, str]]:
        """Build context for current chapter generation."""
        context = []
        context.extend(self.core_context)

        if self.strategy == "aggressive":
            recent_summaries = self.chapter_summaries[-self.recent_chapters:]
            for summary_info in recent_summaries:
                context.append(
                    {
                        "role": "assistant",
                        "content": f"Chapter {summary_info['chapter']} Summary:\n{summary_info['summary']}",
                    }
                )
        elif self.strategy == "balanced":
            for summary_info in self.chapter_summaries:
                context.append(
                    {
                        "role": "assistant",
                        "content": f"Chapter {summary_info['chapter']} Summary:\n{summary_info['summary']}",
                    }
                )
            for chapter_info in self.recent_full_chapters:
                context.append(
                    {
                        "role": "assistant",
                        "content": f"Chapter {chapter_info['chapter']} Full Text:\n{chapter_info['text']}",
                    }
                )
        return context

    def estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count (approximately 4 characters per token)."""
        return len(text) // 4

    def get_previous_chapter_ending(self, current_chapter: int) -> str:
        """Get ending text of previous chapter."""
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
            "suggestions": suggestions,
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
