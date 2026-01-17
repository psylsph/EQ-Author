"""Tests for argument parsing and CLI functions."""

import pytest
import sys
from pathlib import Path
from eq_author import parse_args, get_env_var_for_arg


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        args = parse_args([])
        assert args.story_file is None
        assert args.story_text is None
        assert args.n_chapters is None
        assert args.output_dir == "outputs"
        assert args.api_key is None
        assert args.base_url == "https://api.deepseek.com"
        assert args.model == "deepseek-reasoner"
        assert args.stream is True
        assert args.no_stream is False
        assert args.temperature is None
        assert args.non_interactive is False
        assert args.no_cache is False
        assert args.cache_dir == ".prompt_cache"
        assert args.context_strategy == "aggressive"
        assert args.summary_length == 250
        assert args.recent_chapters == 2
        assert args.max_context_tokens is None

    def test_story_file_argument(self):
        """Test --story-file argument."""
        args = parse_args(["--story-file", "my_story.txt"])
        assert args.story_file == "my_story.txt"

    def test_story_text_argument(self):
        """Test --story-text argument."""
        args = parse_args(["--story-text", "A story about..."])
        assert args.story_text == "A story about..."

    def test_n_chapters_argument(self):
        """Test --n-chapters argument."""
        args = parse_args(["--n-chapters", "12"])
        assert args.n_chapters == 12

    def test_output_dir_argument(self):
        """Test --output-dir argument."""
        args = parse_args(["--output-dir", "custom_output"])
        assert args.output_dir == "custom_output"

    def test_api_key_argument(self):
        """Test --api-key argument."""
        args = parse_args(["--api-key", "my_key"])
        assert args.api_key == "my_key"

    def test_base_url_argument(self):
        """Test --base-url argument."""
        args = parse_args(["--base-url", "http://localhost:1234"])
        assert args.base_url == "http://localhost:1234"

    def test_model_argument(self):
        """Test --model argument."""
        args = parse_args(["--model", "gpt-4"])
        assert args.model == "gpt-4"

    def test_stream_arguments(self):
        """Test --stream and --no-stream arguments."""
        args_stream = parse_args(["--stream"])
        assert args_stream.stream is True

        args_no_stream = parse_args(["--no-stream"])
        assert args_no_stream.no_stream is True

    def test_temperature_argument(self):
        """Test --temperature argument."""
        args = parse_args(["--temperature", "0.7"])
        assert args.temperature == 0.7

    def test_non_interactive_argument(self):
        """Test --non-interactive argument."""
        args = parse_args(["--non-interactive"])
        assert args.non_interactive is True

    def test_no_cache_argument(self):
        """Test --no-cache argument."""
        args = parse_args(["--no-cache"])
        assert args.no_cache is True

    def test_cache_dir_argument(self):
        """Test --cache-dir argument."""
        args = parse_args(["--cache-dir", "custom_cache"])
        assert args.cache_dir == "custom_cache"

    def test_context_strategy_argument(self):
        """Test --context-strategy argument."""
        for strategy in ["aggressive", "balanced"]:
            args = parse_args([f"--context-strategy", strategy])
            assert args.context_strategy == strategy

    def test_summary_length_argument(self):
        """Test --summary-length argument."""
        args = parse_args(["--summary-length", "300"])
        assert args.summary_length == 300

    def test_recent_chapters_argument(self):
        """Test --recent-chapters argument."""
        args = parse_args(["--recent-chapters", "4"])
        assert args.recent_chapters == 4

    def test_max_context_tokens_argument(self):
        """Test --max-context-tokens argument."""
        args = parse_args(["--max-context-tokens", "64000"])
        assert args.max_context_tokens == 64000

    def test_resume_from_argument(self):
        """Test --resume-from argument."""
        args = parse_args(["--resume-from", "/path/to/previous/run"])
        assert args.resume_from == "/path/to/previous/run"

    def test_skip_planning_argument(self):
        """Test --skip-planning argument."""
        args = parse_args(["--skip-planning"])
        assert args.skip_planning is True

    def test_always_autogen_chapters_argument(self):
        """Test --always-autogen-chapters argument."""
        args = parse_args(["--always-autogen-chapters"])
        assert args.always_autogen_chapters is True

    def test_multiple_arguments(self):
        """Test parsing multiple arguments together."""
        args = parse_args([
            "--story-file", "story.txt",
            "--n-chapters", "10",
            "--output-dir", "my_output",
            "--temperature", "0.8",
            "--context-strategy", "balanced",
            "--non-interactive",
        ])
        assert args.story_file == "story.txt"
        assert args.n_chapters == 10
        assert args.output_dir == "my_output"
        assert args.temperature == 0.8
        assert args.context_strategy == "balanced"
        assert args.non_interactive is True


class TestGetEnvVarForArg:
    """Tests for get_env_var_for_arg function."""

    def test_converts_story_file(self, monkeypatch):
        """Test conversion of story-file to env var."""
        import os
        os.environ["EQ_AUTHOR_STORY_FILE"] = "/test/path"
        result = get_env_var_for_arg("story-file")
        assert result == "/test/path"
        del os.environ["EQ_AUTHOR_STORY_FILE"]

    def test_converts_n_chapters(self, monkeypatch):
        """Test conversion of n-chapters to env var."""
        import os
        os.environ["EQ_AUTHOR_N_CHAPTERS"] = "12"
        result = get_env_var_for_arg("n-chapters")
        assert result == "12"
        del os.environ["EQ_AUTHOR_N_CHAPTERS"]

    def test_converts_base_url(self, monkeypatch):
        """Test conversion of base-url to env var."""
        import os
        os.environ["EQ_AUTHOR_BASE_URL"] = "http://localhost:1234"
        result = get_env_var_for_arg("base-url")
        assert result == "http://localhost:1234"
        del os.environ["EQ_AUTHOR_BASE_URL"]

    def test_converts_max_context_tokens(self, monkeypatch):
        """Test conversion of max-context-tokens to env var."""
        import os
        os.environ["EQ_AUTHOR_MAX_CONTEXT_TOKENS"] = "64000"
        result = get_env_var_for_arg("max-context-tokens")
        assert result == "64000"
        del os.environ["EQ_AUTHOR_MAX_CONTEXT_TOKENS"]

    def test_converts_skip_planning(self, monkeypatch):
        """Test conversion of skip-planning to env var."""
        import os
        os.environ["EQ_AUTHOR_SKIP_PLANNING"] = "true"
        result = get_env_var_for_arg("skip-planning")
        assert result == "true"
        del os.environ["EQ_AUTHOR_SKIP_PLANNING"]

    def test_converts_always_autogen_chapters(self, monkeypatch):
        """Test conversion of always-autogen-chapters to env var."""
        import os
        os.environ["EQ_AUTHOR_ALWAYS_AUTOGEN_CHAPTERS"] = "true"
        result = get_env_var_for_arg("always-autogen-chapters")
        assert result == "true"
        del os.environ["EQ_AUTHOR_ALWAYS_AUTOGEN_CHAPTERS"]

    def test_returns_none_when_not_set(self):
        """Test that function returns None when env var is not set."""
        result = get_env_var_for_arg("nonexistent-arg")
        assert result is None
