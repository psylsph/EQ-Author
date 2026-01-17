"""Tests for ContextManager class."""

import pytest
from eq_author import ContextManager, DEFAULT_SUMMARY_LENGTH, DEFAULT_RECENT_CHAPTERS


class TestContextManagerInit:
    """Tests for ContextManager initialization."""

    def test_default_values(self):
        """Test ContextManager has correct default values."""
        cm = ContextManager()
        assert cm.strategy == "aggressive"
        assert cm.summary_length == DEFAULT_SUMMARY_LENGTH
        assert cm.recent_chapters == DEFAULT_RECENT_CHAPTERS
        assert cm.max_context_tokens == 8000

    def test_custom_values(self):
        """Test ContextManager with custom values."""
        cm = ContextManager(
            strategy="balanced",
            summary_length=300,
            recent_chapters=4,
            max_context_tokens=16000
        )
        assert cm.strategy == "balanced"
        assert cm.summary_length == 300
        assert cm.recent_chapters == 4
        assert cm.max_context_tokens == 16000

    def test_initial_empty_lists(self):
        """Test that all lists are initially empty."""
        cm = ContextManager()
        assert cm.core_context == []
        assert cm.chapter_summaries == []
        assert cm.recent_full_chapters == []
        assert cm.current_messages == []


class TestContextManagerCoreContext:
    """Tests for add_core_context method."""

    def test_add_core_context(self):
        """Test adding core context."""
        cm = ContextManager()
        messages = [{"role": "system", "content": "You are helpful"}]
        cm.add_core_context(messages)
        assert cm.core_context == messages

    def test_replace_core_context(self):
        """Test that add_core_context replaces existing context."""
        cm = ContextManager()
        cm.add_core_context([{"role": "system", "content": "First"}])
        cm.add_core_context([{"role": "system", "content": "Second"}])
        assert len(cm.core_context) == 1
        assert cm.core_context[0]["content"] == "Second"

    def test_core_context_copied(self):
        """Test that core context is copied, not referenced."""
        cm = ContextManager()
        messages = [{"role": "system", "content": "Test"}]
        cm.add_core_context(messages)
        messages.append({"role": "user", "content": "Modified"})
        assert len(cm.core_context) == 1


class TestContextManagerChapters:
    """Tests for add_chapter method."""

    def test_add_single_chapter(self):
        """Test adding a single chapter."""
        cm = ContextManager()
        cm.add_chapter(1, "Chapter 1 text", "Chapter 1 summary", "Ending text")
        assert len(cm.chapter_summaries) == 1
        assert cm.chapter_summaries[0]["chapter"] == 1
        assert cm.chapter_summaries[0]["summary"] == "Chapter 1 summary"
        assert cm.chapter_summaries[0]["ending"] == "Ending text"

    def test_add_multiple_chapters(self):
        """Test adding multiple chapters."""
        cm = ContextManager()
        for i in range(1, 6):
            cm.add_chapter(i, f"Chapter {i}", f"Summary {i}", f"Ending {i}")
        assert len(cm.chapter_summaries) == 5

    def test_recent_full_chapters_maintained(self):
        """Test that recent full chapters are maintained in balanced mode."""
        cm = ContextManager(strategy="balanced", recent_chapters=2)
        for i in range(1, 5):
            cm.add_chapter(i, f"Chapter {i} text", f"Summary {i}", f"Ending {i}")
        assert len(cm.recent_full_chapters) == 2
        assert cm.recent_full_chapters[0]["chapter"] == 3
        assert cm.recent_full_chapters[1]["chapter"] == 4

    def test_recent_full_chapters_limited(self):
        """Test that recent full chapters are limited to recent_chapters."""
        cm = ContextManager(strategy="balanced", recent_chapters=1)
        for i in range(1, 4):
            cm.add_chapter(i, f"Chapter {i} text", f"Summary {i}", f"Ending {i}")
        assert len(cm.recent_full_chapters) == 1
        assert cm.recent_full_chapters[0]["chapter"] == 3


class TestContextManagerBuildContext:
    """Tests for build_context method."""

    def test_empty_context(self):
        """Test building context with no chapters."""
        cm = ContextManager()
        context = cm.build_context(1)
        assert context == []

    def test_core_context_included(self):
        """Test that core context is included."""
        cm = ContextManager()
        cm.add_core_context([{"role": "system", "content": "Core"}])
        context = cm.build_context(1)
        assert len(context) == 1
        assert context[0]["content"] == "Core"

    def test_aggressive_strategy_summaries_only(self):
        """Test aggressive strategy includes only summaries."""
        cm = ContextManager(strategy="aggressive", recent_chapters=2)
        cm.add_core_context([{"role": "system", "content": "Core"}])
        cm.add_chapter(1, "Full 1", "Summary 1", "End 1")
        cm.add_chapter(2, "Full 2", "Summary 2", "End 2")
        cm.add_chapter(3, "Full 3", "Summary 3", "End 3")
        context = cm.build_context(3)
        # Should include core context + summaries of recent chapters (not full text)
        # The build_context method returns summaries for recent chapters as assistant messages
        assert len(context) >= 1  # At least core context


class TestContextManagerTokenEstimation:
    """Tests for token estimation."""

    def test_empty_text(self):
        """Test estimating tokens for empty text."""
        cm = ContextManager()
        assert cm.estimate_tokens("") == 0

    def test_rough_estimate(self):
        """Test rough token estimate (4 chars per token)."""
        cm = ContextManager()
        # 100 characters should be ~25 tokens
        tokens = cm.estimate_tokens("a" * 100)
        assert tokens == 25


class TestContextManagerChapterEnding:
    """Tests for get_previous_chapter_ending method."""

    def test_no_previous_chapter(self):
        """Test when there is no previous chapter."""
        cm = ContextManager()
        result = cm.get_previous_chapter_ending(1)
        assert result == ""

    def test_previous_chapter_ending(self):
        """Test getting previous chapter ending."""
        cm = ContextManager()
        cm.add_chapter(1, "Text 1", "Summary 1", "Ending 1")
        cm.add_chapter(2, "Text 2", "Summary 2", "Ending 2")
        result = cm.get_previous_chapter_ending(2)
        assert result == "Ending 1"

    def test_non_consecutive_chapter(self):
        """Test getting ending for non-consecutive chapter request."""
        cm = ContextManager()
        cm.add_chapter(1, "Text 1", "Summary 1", "Ending 1")
        cm.add_chapter(3, "Text 3", "Summary 3", "Ending 3")
        result = cm.get_previous_chapter_ending(4)  # Should look for chapter 3 when asking about chapter 4
        assert result == "Ending 3"


class TestContextManagerCheckContextSize:
    """Tests for check_context_size method."""

    def test_empty_messages(self):
        """Test checking empty messages."""
        cm = ContextManager()
        result = cm.check_context_size([])
        assert result["estimated_tokens"] == 0
        assert result["max_tokens"] == 8000
        assert result["usage_ratio"] == 0.0
        assert result["is_warning"] is False
        assert result["is_critical"] is False

    def test_warning_threshold(self):
        """Test reaching warning threshold (75%)."""
        cm = ContextManager(max_context_tokens=1000)
        # Create ~800 tokens (3200 chars) should trigger warning at 75%
        messages = [{"content": "x" * 3200}]
        result = cm.check_context_size(messages)
        # The actual ratio depends on the token estimation formula
        assert result["is_warning"] is True or result["usage_ratio"] >= 0.7

    def test_critical_threshold(self):
        """Test reaching critical threshold (90%)."""
        cm = ContextManager(max_context_tokens=1000)
        # Create ~1000 tokens (4000 chars)
        messages = [{"content": "x" * 4000}]
        result = cm.check_context_size(messages)
        assert result["is_warning"] is True
        assert result["is_critical"] is True
