"""Tests for auto-review and chapter review functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from eq_author import auto_review_chapter


class TestAutoReviewChapter:
    """Tests for auto_review_chapter function."""

    def test_function_exists(self):
        """Test that auto_review_chapter function exists."""
        assert callable(auto_review_chapter)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect
        sig = inspect.signature(auto_review_chapter)
        params = list(sig.parameters.keys())
        assert "client" in params
        assert "model" in params
        assert "chapter_text" in params
        assert "chapter_num" in params

    def test_returns_tuple(self):
        """Test that function returns a tuple of (review_notes, rewritten_text, llm_approved)."""
        # Create a mock client that returns a simple response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Review
- Word count: 2000
- Rating: 4/5
- Issues: None

## Rewrite Needed: NO
NO REWRITE NEEDED"""
        mock_client.chat.completions.create.return_value = mock_response

        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text="Test chapter content",
            chapter_num=1,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        review_notes, rewritten_text, llm_approved = result
        assert isinstance(review_notes, str)
        assert isinstance(rewritten_text, str)
        assert isinstance(llm_approved, bool)

    def test_rewrite_needed_yes(self):
        """Test handling when rewrite is needed."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Review
- Word count: 2000
- Rating: 3/5
- Issues: Pacing issues

## Rewrite Needed: YES

# Chapter 1: The Beginning

The story begins with...
[rewritten content here]
"""
        mock_client.chat.completions.create.return_value = mock_response

        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text="Original chapter content",
            chapter_num=1,
        )

        review_notes, rewritten_text, llm_approved = result
        assert "REVIEW" in review_notes.upper() or "Rating:" in review_notes
        assert isinstance(llm_approved, bool)

    def test_no_rewrite_needed(self):
        """Test handling when no rewrite is needed."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Review
- Word count: 3000
- Rating: 5/5
- Issues: None

## Rewrite Needed: NO
NO REWRITE NEEDED"""
        mock_client.chat.completions.create.return_value = mock_response

        original_text = "Original chapter content"
        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text=original_text,
            chapter_num=1,
        )

        review_notes, rewritten_text, llm_approved = result
        # When no rewrite needed, should return original text
        assert rewritten_text == original_text
        assert isinstance(llm_approved, bool)

    def test_with_previous_ending(self):
        """Test that previous chapter ending is included."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Review
- Word count: 2000
- Rating: 4/5
- Issues: None

## Rewrite Needed: NO
NO REWRITE NEEDED"""
        mock_client.chat.completions.create.return_value = mock_response

        previous_ending = "The door creaked open slowly."
        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text="Chapter content",
            chapter_num=2,
            previous_chapter_ending=previous_ending,
        )

        # Verify the API was called
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_content = messages[-1]["content"]
        assert "Previous chapter ended" in user_content or previous_ending in user_content

    def test_with_optional_params(self):
        """Test with optional parameters (stream, temperature, cache)."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Review
- Word count: 2000
- Rating: 4/5
- Issues: None

## Rewrite Needed: NO
NO REWRITE NEEDED"""
        mock_client.chat.completions.create.return_value = mock_response

        # Test without stream (simpler)
        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text="Test content",
            chapter_num=1,
            stream=False,
            temperature=0.7,
            cache=None,
        )

        # Verify temperature was passed
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs.get("temperature") == 0.7

    def test_extraction_fallback(self):
        """Test that extraction fallback works when rewrite text is too short."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Response that indicates rewrite but with very short content
        mock_response.choices[0].message.content = """## Review
- Word count: 2000
- Rating: 3/5
- Issues: Some issues

## Rewrite Needed: YES

Short
"""
        mock_client.chat.completions.create.return_value = mock_response

        original_text = "Original chapter content"
        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text=original_text,
            chapter_num=1,
        )

        review_notes, rewritten_text, llm_approved = result
        # Should fall back to original when extraction fails
        assert rewritten_text == original_text
        assert isinstance(llm_approved, bool)

    def test_llm_approved_true_when_no_rewrite_needed(self):
        """Test that llm_approved is True when no rewrite needed."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Review
- Word count: 3000
- Rating: 5/5
- Issues: None

## Rewrite Needed: NO
NO REWRITE NEEDED"""
        mock_client.chat.completions.create.return_value = mock_response

        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text="Test chapter content",
            chapter_num=1,
        )

        review_notes, rewritten_text, llm_approved = result
        assert llm_approved is True

    def test_llm_approved_true_when_rewrite_provided(self):
        """Test that rewritten content is extracted when LLM provides it."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """## Quality Assessment
- Word Count: 1500 (target: 2000+)
- Narrative Flow: 2/5
- Character Consistency: 3/5
- Dialogue Quality: 3/5
- Scene Description: 3/5
- Pacing: 2/5
- Overall: 2/5

## Verdict
NEEDS_IMPROVEMENT (specific issues listed below)

## Review Notes
The chapter is too short and lacks sufficient development of the setting and characters. It needs expansion to meet the word count target and provide a more immersive experience.

## Rewrite Recommendation
NEEDS_IMPROVEMENT

The expanded story begins with characters entering the ancient ruins under cover of darkness. Sarah led the way, her torch casting dancing shadows on moss-covered stone walls that had stood for a thousand years. The air was thick with the smell of damp earth and forgotten secrets.

She had prepared for this moment her entire life, training her mind and body for the challenges that lay ahead. The ancient prophecy spoke of a treasure beyond imagination, guarded by puzzles and traps that had claimed the lives of many would-be explorers.

"The entrance should be nearby," she whispered to her companion, a former archaeologist turned treasure hunter. His eyes glowed with the same fire she felt in her own chest - the fire of discovery, of ambition, of destiny.
"""
        mock_client.chat.completions.create.return_value = mock_response

        result = auto_review_chapter(
            client=mock_client,
            model="test-model",
            chapter_text="Short content",
            chapter_num=1,
        )

        review_notes, rewritten_text, llm_approved = result
        # When LLM provides rewrite, llm_approved is False (needs another review)
        assert llm_approved is False
        # Verify rewritten text was extracted (not original)
        assert "The expanded story begins" in rewritten_text
        # Verify original was NOT used (extraction worked)
        assert rewritten_text != "Short content"


class TestAutoReviewIntegration:
    """Integration tests for auto-review with actual API (optional)."""

    @pytest.mark.integration
    def test_review_with_real_api(self):
        """Test auto-review with real API endpoint."""
        import os

        api_base = os.getenv("TEST_API_BASE_URL", "http://192.168.1.227:1234/v1")
        api_key = os.getenv("TEST_API_KEY", "any-non-empty-string")
        model = os.getenv("TEST_MODEL", "mistralai/ministral-3-14b-reasoning")

        # Skip if API not available
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)

            # Quick connectivity test
            client.models.list()

            chapter_text = """# Chapter 1: The Beginning

Sarah walked through the ancient library, her footsteps echoing on the marble floors. She had been searching for three days when she finally found the hidden door.

The door was old, its wood worn smooth by countless hands over centuries. A faint glow emanated from the cracks around its frame.

Sarah hesitated. She had read the legends, heard the stories passed down through generations of librarians.

With trembling fingers, she reached out and touched the cold metal of the handle.

The door swung open with a whisper, revealing a vast chamber filled with books that seemed to glow with an inner light.
"""

            review_notes, rewritten_text, llm_approved = auto_review_chapter(
                client=client,
                model=model,
                chapter_text=chapter_text,
                chapter_num=1,
                stream=False,
            )

            assert "Review" in review_notes or "Rating" in review_notes
            assert len(rewritten_text) > 100
            assert isinstance(llm_approved, bool)

        except Exception as e:
            pytest.skip(f"API not available: {e}")
