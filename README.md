# EQ-Author DeepSeek Planner & Writer

A powerful CLI that reads a story idea and runs a 7-step planning/writing workflow using the DeepSeek API via the OpenAI Python client. Features intelligent context management for limited-context models and beautiful PDF generation.

## Features

- **7-Step Planning Workflow**: Brainstorming, intention setting, chapter planning, character development, and more
- **Narrative Overlap Prevention**: Intelligent chapter boundary management prevents repetition between chapters
- **Intelligent Context Management**: Automatically manages conversation context to prevent overflow in limited-context models
- **Interactive & Non-Interactive Modes**: Run with real-time feedback or fully automated
- **PDF Generation**: Convert your stories to professionally formatted PDFs with custom fonts
- **Prompt Caching**: Avoid repeat API calls with intelligent response caching
- **Multi-Model Support**: Works with DeepSeek, OpenRouter, LM Studio, and other OpenAI-compatible APIs

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - `openai>=1.30.0` - For API communication
  - `python-dotenv>=1.0.1` - For environment variable loading
  - `reportlab>=3.6.0` - For PDF generation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EQ-Author
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Authentication & Configuration

Create a `.env` file in the project root with:
```
API_KEY=your_deepseek_api_key
# Optional: OPENAI_API_KEY=your_key as fallback
```

The CLI auto-loads `.env` via python-dotenv. You can also pass configuration via command-line arguments.

### Configuration Options

- **API Key**: Set via `API_KEY` environment variable or `--api-key` flag
- **Base URL**: Default is `https://api.deepseek.com`, override with `--base-url`
- **Model**: Default is `deepseek-reasoner`, override with `--model`
- **Caching**: Enabled by default, stored under `.prompt_cache/`

## Usage

### Basic CLI Usage

Run with a story file (model proposes chapter count and prompts for feedback):
```bash
python eq_author.py --story-file ideas/my_story.md
```

Provide idea directly and specify chapter count:
```bash
python eq_author.py --story-text "A detective discovers a mysterious library" --n-chapters 5
```

Enable streaming with custom temperature:
```bash
python eq_author.py --story-file idea.txt --n-chapters 6 --stream --temperature 0.8
```

Note: Temperature and max_context_tokens defaults are model-specific:
- deepseek-reasoner: temperature=1.0, max_context_tokens=32000 (default output), max output 64K
- Other models: temperature=1.0, max_context_tokens=8000

### Non-Interactive Mode

Run without feedback prompts (ideal for automation):
```bash
python eq_author.py --story-file idea.txt --non-interactive --stream
```

### Context Management

For limited-context models (4K-8K tokens):
```bash
python eq_author.py --story-file idea.txt --context-strategy aggressive --summary-length 150 --recent-chapters 1 --max-context-tokens 6000
```

For medium-context models (16K-32K tokens):
```bash
python eq_author.py --story-file idea.txt --context-strategy balanced --summary-length 250 --recent-chapters 3 --max-context-tokens 24000
```

For large-context models (64K+ tokens):
```bash
python eq_author.py --story-file idea.txt --context-strategy full --summary-length 400 --recent-chapters 5 --max-context-tokens 60000
```

### Optimal Settings for DeepSeek

For DeepSeek models with 128K max context length:

**Recommended (Balanced approach):**
```bash
python eq_author.py --story-file idea.txt --context-strategy balanced --max-context-tokens 64000 --model deepseek-reasoner
```

**Maximum performance (Unlimited context):**
```bash
python eq_author.py --story-file idea.txt --unlimited-context --model deepseek-reasoner
```

For unlimited context (bypass all context management):
```bash
python eq_author.py --story-file idea.txt --unlimited-context
```

### Alternative API Providers

**OpenRouter:**
```bash
python eq_author.py --story-file idea.txt --base-url https://openrouter.ai/api/v1 --model moonshotai/kimi-k2-0905 --api-key sk-or-v1-your-key
```

**LM Studio (Local):**
```bash
python eq_author.py --story-file idea.txt --base-url http://127.0.0.1:11434/v1 --model mistral-nemo:12b
```

**Custom Server:**
```bash
python eq_author.py --story-file idea.txt --base-url http://192.168.1.227:1234/v1 --model gemma-3-12b-it
```

### Prompt Override for Adult Target

If your idea metadata indicates an adult target audience (for example a metadata line like `**Target Audience:** Adults` or a genre containing the word "Adult" or "Erotic"), EQ-Author will automatically attempt to apply a project-level prompt override.

- Place a file named `prompt_override.txt` in the project root (the same directory as `eq_author.py`).
- When an adult target is detected, the file contents are prepended as an initial `system` message before the default `You are a helpful assistant` system message.
- CLI indicators and trace files:
   - When the override is applied you will see a concise console message:
      - `[PROMPT OVERRIDE] Applied prompt_override.txt (adult target detected)`
   - A copy of the applied override is written to the run output directory as `prompt_override_used.txt` for traceability.
   - If an adult target is detected but `prompt_override.txt` is missing or empty you will see:
      - `[PROMPT OVERRIDE] Detected adult target but `prompt_override.txt` is missing or empty`
   - In that case a small marker file `prompt_override_missing.txt` is written to the run output directory.

This allows you to centralise any extra system-level instructions (for example stylistic constraints, safety or legal guidance, or account-specific instructions that should always be applied to adult pieces) without editing the code.

Example `prompt_override.txt` (very small example):
```
You are an adult-content editor. Follow the author's stylistic intentions while respecting local laws and content safety constraints. When asked to produce explicit content, ensure the output matches the author's explicit request and formatting instructions.
```

Tip: The override is applied based on a simple metadata check near the top of the idea file (case-insensitive). If your idea uses a different metadata format, add a clear `Target Audience` line near the top.

### Local API Providers — LM Studio & Ollama

When running models locally you typically talk to a local HTTP API. Common examples and defaults used in examples above:

- LM Studio (local server): `http://127.0.0.1:11434/v1` is a common default base URL used by LM Studio installations. Use the `:11434` port in examples above if your local LM Studio instance listens there.
- Ollama (local server): Ollama also exposes an HTTP-compatible API. The base URL and port can vary depending on your Ollama configuration; a common pattern is `http://127.0.0.1:<port>/v1` (replace `<port>` with your configured API port).

Notes and how to confirm the correct port:

- Check the local service UI or logs for the configured API port. Both LM Studio and Ollama surface their API settings in their local admin UI or startup logs.
- You can quickly verify connectivity and available models from your chosen server by running a small Python test or using `curl`. For example (adjust `BASE_URL` and `MODEL` for your setup):

```bash
# Quick connectivity test (list models) — Python (requires OpenAI-like client configured)
python - <<'PY'
from openai import OpenAI
client = OpenAI(api_key='any-non-empty-string', base_url='http://127.0.0.1:11434/v1')
print(client.models.list())
PY
```

Or with `curl` (if your server exposes a models endpoint):

```bash
curl "http://127.0.0.1:11434/v1/models" -H "Authorization: Bearer <API_KEY>"
```

If you run into connection errors, double-check the base URL and port, ensure the server is running and accessible from your machine, and confirm any required API key or auth settings for the server in question.

#### Example: LM Studio model with 12,000-token context window

For an LM Studio model that supports approximately 12,000 context tokens, use a balanced approach but reduce per-summary length and the number of recent full chapters to stay comfortably under the limit.

Recommended flags:

```bash
python eq_author.py \
   --story-file ideas/my_story.md \
   --base-url http://127.0.0.1:11434/v1 \
   --model <your-lmstudio-model-name> \
   --non-interactive \
   --stream \
   --context-strategy balanced \
   --max-context-tokens 12000 \
   --summary-length 150 \
   --recent-chapters 1 \
   --temperature 0.8
```

Why these choices:

- `--context-strategy balanced`: keeps full text for the most recent chapter(s) while retaining summaries for earlier chapters for continuity.
- `--max-context-tokens 12000`: tells the tool to target a 12k token budget when estimating context usage.
- `--summary-length 150`: shorter summaries reduce token usage while preserving plot signals.
- `--recent-chapters 1`: keep only the last chapter in full to save tokens while maintaining immediate continuity.
- `--temperature 0.8`: gives a slight reduction from the default to help with focused, coherent chapter generation.

If you see context warnings during a run, reduce `--summary-length` further (e.g., 100) or lower `--recent-chapters` to 0 to rely on summaries only.

### Prompt Caching

Enable caching (default):
```bash
python eq_author.py --story-file idea.txt --cache-dir .prompt_cache
```

Disable caching:
```bash
python eq_author.py --story-file idea.txt --cache-dir ""
```

## Context Management

The system includes intelligent context management to handle long stories with limited-context models:

### How Context Strategies Work

The context management system automatically maintains story continuity by tracking previous chapters while managing memory usage. Here's how each strategy works:

#### Summary Length and Recent Chapters

- **Summary Length**: Each chapter summary is approximately 250 words by default (controlled by `--summary-length`)
- **Recent Chapters**: The system keeps the full text of the most recent chapters (default is 2, controlled by `--recent-chapters`)

For example, with a 10-chapter story using default settings:
- **Aggressive strategy**: Keeps only summaries of chapters 8-10 (3 × 250 words = ~750 words total)
- **Balanced strategy**: Keeps summaries of chapters 1-8 (8 × 250 words = 2000 words) + full text of chapters 9-10
- **Full strategy**: Attempts to keep all chapters in full text (may cause overflow with limited models)

### Context Strategies

1. **Aggressive** (default): Only keeps summaries of recent chapters
   - Best for: Very limited context models (4K-8K tokens)
   - Memory usage: Lowest
   - Detail level: Summaries only
   - Example: For a 10-chapter story, keeps only the last 2-3 chapter summaries

2. **Balanced**: Keeps all summaries + full text of recent chapters
   - Best for: Medium context models (16K-32K tokens)
   - Memory usage: Medium
   - Detail level: Full recent chapters + all summaries
   - Example: For a 10-chapter story, keeps summaries of chapters 1-8 (2000 words) + full text of chapters 9-10
   - **Recommended for DeepSeek**: Provides good balance of context and performance

3. **Full**: Attempts to keep all chapter text
   - Best for: Large context models (64K+ tokens)
   - Memory usage: Highest
   - Detail level: Everything
   - Example: For a 10-chapter story, tries to keep all 10 chapters in full text

4. **Unlimited**: Bypasses all context management
   - Best for: Very large context models (128K+ tokens) or when context limits are not a concern
   - Memory usage: No limits
   - Detail level: Everything without summarization
   - Note: Enabled with --unlimited-context flag
   - **Optimal for DeepSeek**: Takes full advantage of 128K context window

### Context Management Options

- `--context-strategy {aggressive,balanced,full}`: Strategy for managing context window
- `--unlimited-context`: Bypass all context management and keep all full chapters in memory without summarization
- `--summary-length INTEGER`: Target word count for chapter summaries (default: 250)
  - Increase for more detailed summaries (e.g., 300-400 for complex stories)
  - Decrease for shorter summaries (e.g., 150-200 for simpler narratives)
- `--recent-chapters INTEGER`: Number of recent full chapters to keep (default: 2)
  - Increase for better continuity (e.g., 3-4 for complex character development)
  - Decrease to save memory (e.g., 1 for minimal context)
- `--max-context-tokens INTEGER`: Maximum context tokens to maintain (default: 8000)

The system automatically generates chapter summaries and monitors context usage, providing warnings when approaching limits. When using `--unlimited-context`, all chapter summarization is skipped and context warnings are disabled.

## Narrative Overlap Prevention

The system includes intelligent chapter boundary management to prevent narrative overlap between chapters:

### How It Works

1. **Chapter Ending Tracking**: Automatically extracts the last 1-2 sentences of each chapter to track where it ends
2. **Explicit Instructions**: Provides clear guidance to the AI to avoid repeating content from the previous chapter's ending
3. **Boundary Creation**: Helps the AI create distinct chapter transitions while maintaining story continuity

### Benefits

- **No More Repetitive Beginnings**: Each chapter starts with new content, not by restating the previous chapter's ending
- **Smooth Transitions**: Maintains narrative flow without redundancy
- **Clear Chapter Boundaries**: Each chapter feels distinct and self-contained
- **Preserved Continuity**: Story context is maintained without repetitive content

This feature works automatically with all context strategies and requires no additional configuration. It's especially useful for longer stories where maintaining distinct chapter boundaries is important for reader engagement.

## Command-Line Options

```
usage: eq_author.py [-h] [--story-file STORY_FILE] [--story-text STORY_TEXT]
                    [--n-chapters N_CHAPTERS] [--output-dir OUTPUT_DIR]
                    [--api-key API_KEY] [--base-url BASE_URL] [--model MODEL]
                    [--stream] [--no-stream] [--temperature TEMPERATURE]
                    [--non-interactive] [--no-cache] [--cache-dir CACHE_DIR]
                    [--context-strategy {aggressive,balanced,full,unlimited}]
                    [--unlimited-context] [--summary-length SUMMARY_LENGTH]
                    [--recent-chapters RECENT_CHAPTERS]
                    [--max-context-tokens MAX_CONTEXT_TOKENS]

options:
  -h, --help            show this help message and exit
  --story-file STORY_FILE
                        Path to file containing the story idea
  --story-text STORY_TEXT
                        Story idea text (overrides file if provided)
  --n-chapters N_CHAPTERS
                        Override: number of chapters to write (if omitted, model proposes)
  --output-dir OUTPUT_DIR
                        Base directory for outputs (default: outputs)
  --api-key API_KEY     DeepSeek API key (or set API_KEY)
  --base-url BASE_URL   DeepSeek base URL (default: https://api.deepseek.com)
  --model MODEL         Model name (default: deepseek-reasoner)
  --stream              Stream responses (aggregated in output) (default: True)
  --no-stream           Disable response streaming
  --temperature TEMPERATURE
                        Sampling temperature (default: 1.0)
  --non-interactive     Run without feedback prompts and chapter count confirmation
  --no-cache            Disable prompt caching
  --cache-dir CACHE_DIR
                        Directory for prompt/response cache (default: .prompt_cache)
  --context-strategy {aggressive,balanced,full,unlimited}
                        Strategy for managing context window (default: aggressive)
  --unlimited-context
                        Bypass all context management and keep all full chapters in memory without summarization
  --summary-length SUMMARY_LENGTH
                        Target word count for chapter summaries (default: 250)
  --recent-chapters RECENT_CHAPTERS
                        Number of recent full chapters to keep in context (default: 2)
  --max-context-tokens MAX_CONTEXT_TOKENS
                        Maximum context tokens to maintain (default: 128000)
```

## Project Structure

```
EQ-Author/
├── eq_author.py          # Main CLI application
├── publish_to_pdf.py     # PDF generation utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── Planning.md          # Detailed prompt templates
├── .env                 # Environment variables (create this)
├── .prompt_cache/       # Prompt/response cache (auto-created)
├── ideas/               # Sample story ideas
│   ├── Sample.md
│   └── unit985.md
├── outputs/             # Generated stories (auto-created)
│   └── run-YYYYmmdd-HHMMSS/
│       ├── 01_brainstorm_and_reflection.md
│       ├── 02_intention_and_chapter_planning.md
│       ├── 03_human_vs_llm_critique.md
│       ├── 04_final_plan.md
│       ├── 05_characters.md
│       └── chapters/
│           ├── chapter_01.md
│           ├── chapter_01_summary.md
│           ├── chapter_02.md
│           └── chapter_02_summary.md
└── story_output/        # Generated PDFs (auto-created)
```

## Output Files

Each run creates a timestamped directory under `outputs/run-YYYYmmdd-HHMMSS/` containing:

### Planning Files
- `01_brainstorm_and_reflection.md` - Initial ideas and proposed chapter count
- `02_intention_and_chapter_planning.md` - Story intentions and chapter outlines
- `03_human_vs_llm_critique.md` - Analysis of human vs AI writing approaches
- `04_final_plan.md` - Refined final story plan
- `05_characters.md` - Detailed character profiles

### Chapter Files
- `chapters/chapter_XX.md` - Individual chapter content
- `chapters/chapter_XX_summary.md` - Chapter summaries (when using context management)

## Troubleshooting

### Common Issues

**Connection Errors:**
- Ensure the API server is running and accessible
- Check if the URL is correct (e.g., `http://localhost:1234` for LM Studio)
- Verify no firewall is blocking the connection
- For LM Studio, make sure 'Server' mode is enabled

**Authentication Errors:**
- Check if API key is correct
- For LM Studio, API key may not be required (use any non-empty string)
- Ensure the API key has sufficient permissions

**Model Not Found:**
- Check if the model name is correct
- For LM Studio, ensure a model is loaded
- Use `client.models.list()` to see available models

**Context Overflow:**
- Enable context management with `--context-strategy aggressive`
- Reduce `--max-context-tokens` for your model
- Decrease `--recent-chapters` to keep fewer full chapters
- Lower `--summary-length` to create shorter summaries

**Timeout Errors:**
- Try again with a shorter prompt or simpler request
- Check if the server is overloaded
- Consider using streaming mode (`--stream`)

### Performance Tips

1. **Enable Caching**: Use prompt caching to avoid repeat API calls
2. **Optimize Context**: Choose the right context strategy for your model
3. **Use Streaming**: Enable streaming to see progress in real-time
4. **Batch Operations**: Run multiple chapters in non-interactive mode
5. **Monitor Usage**: Watch for context warnings and adjust settings

## Development

### Running Tests

Currently, there is no automated test suite. Manual verification is recommended:

1. Run the planner with a sample idea:
```bash
python eq_author.py --story-file ideas/Sample.md --n-chapters 2 --non-interactive --stream
```

2. Verify outputs in the latest `outputs/run-*` folder

3. Check the generated Markdown and PDFs for proper formatting

### Contributing

When contributing to the project:

1. Preserve existing prompt templates, step filenames, and directory layout
2. Maintain compatibility with `--non-interactive` flows for automation
3. Update documentation when adding new flags or changing defaults
4. Test with multiple API providers when making connection changes

## License

This project is licensed under the terms specified in the LICENSE file.

## Changelog

### Recent Updates
- Added intelligent context management for limited-context models
- Implemented automatic chapter summary generation
- Added PDF generation with custom fonts and professional layout
- Added support for multiple API providers (OpenRouter, LM Studio, etc.)
- Improved error handling and troubleshooting tips
- Added prompt caching to reduce API calls
- Enhanced CLI with more configuration options
- **Fixed narrative overlap between chapters**: Implemented chapter ending tracking and explicit instructions to prevent repetition at chapter transitions

## PDF Generation

Convert your stories to professionally formatted PDFs:

### CLI PDF Generation

Generate PDF from a specific chapter:
```bash
python publish_to_pdf.py outputs/run-20231001-120000/chapters/chapter_01.md -o story_output/MyStory.pdf -t "My Story"
```

Generate PDF from all chapters in a directory:
```bash
python publish_to_pdf.py outputs/run-20231001-120000/chapters -o story_output/CompleteStory.pdf -t "Complete Story"
```

### PDF Features

- **Custom Fonts**: Automatically downloads and registers beautiful fonts (Crimson Text, Cinzel)
- **Professional Layout**: Proper margins, spacing, and typography
- **Chapter Detection**: Automatically identifies chapters from Markdown files
- **Background Colors**: Eye-friendly paper-like background
- **Markdown Support**: Handles headings, lists, blockquotes, and code blocks
