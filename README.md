# EQ-Author DeepSeek Planner & Writer

A powerful CLI and Streamlit UI that reads a story idea and runs a 7-step planning/writing workflow using the DeepSeek API via the OpenAI Python client. Features intelligent context management for limited-context models and beautiful PDF generation.

## Features

- **7-Step Planning Workflow**: Brainstorming, intention setting, chapter planning, character development, and more
- **Intelligent Context Management**: Automatically manages conversation context to prevent overflow in limited-context models
- **Interactive & Non-Interactive Modes**: Run with real-time feedback or fully automated
- **Streamlit Web UI**: User-friendly browser interface with real-time streaming
- **PDF Generation**: Convert your stories to professionally formatted PDFs with custom fonts
- **Prompt Caching**: Avoid repeat API calls with intelligent response caching
- **Multi-Model Support**: Works with DeepSeek, OpenRouter, LM Studio, and other OpenAI-compatible APIs

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - `openai>=1.30.0` - For API communication
  - `python-dotenv>=1.0.1` - For environment variable loading
  - `reportlab>=3.6.0` - For PDF generation
  - `streamlit>=1.34.0` - For the web UI

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

### Prompt Caching

Enable caching (default):
```bash
python eq_author.py --story-file idea.txt --cache-dir .prompt_cache
```

Disable caching:
```bash
python eq_author.py --story-file idea.txt --cache-dir ""
```

## Streamlit Web UI

Launch the interactive planner in your browser:

```bash
streamlit run streamlit_app.py
```

The Streamlit UI provides:
- **Real-time Streaming**: Watch responses generate live
- **Step-by-Step Control**: Advance through planning stages at your own pace
- **Feedback Integration**: Add feedback at any step and apply it
- **PDF Export**: Generate and download PDFs directly from the UI
- **Model Selection**: Refresh and select from available models
- **Configuration Panel**: Adjust all settings from the sidebar

### UI Features

- **Configuration Sidebar**: Set API key, base URL, model, temperature, and streaming options
- **Story Setup**: Upload files or paste story ideas directly
- **Planning Steps**: Interactive expandable sections for each planning step
- **Chapter Management**: View chapters, provide feedback, and track progress
- **PDF Generation**: Create formatted PDFs with custom titles

## Context Management

The system includes intelligent context management to handle long stories with limited-context models:

### Context Strategies

1. **Aggressive** (default): Only keeps summaries of recent chapters
   - Best for: Very limited context models (4K-8K tokens)
   - Memory usage: Lowest
   - Detail level: Summaries only

2. **Balanced**: Keeps all summaries + full text of recent chapters
   - Best for: Medium context models (16K-32K tokens)
   - Memory usage: Medium
   - Detail level: Full recent chapters + all summaries

3. **Full**: Attempts to keep all chapter text
   - Best for: Large context models (64K+ tokens)
   - Memory usage: Highest
   - Detail level: Everything

### Context Management Options

- `--context-strategy {aggressive,balanced,full}`: Strategy for managing context window
- `--summary-length INTEGER`: Target word count for chapter summaries (default: 250)
- `--recent-chapters INTEGER`: Number of recent full chapters to keep (default: 2)
- `--max-context-tokens INTEGER`: Maximum context tokens to maintain (default: 8000)

The system automatically generates chapter summaries and monitors context usage, providing warnings when approaching limits.

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

## Command-Line Options

```
usage: eq_author.py [-h] [--story-file STORY_FILE] [--story-text STORY_TEXT]
                    [--n-chapters N_CHAPTERS] [--output-dir OUTPUT_DIR]
                    [--api-key API_KEY] [--base-url BASE_URL] [--model MODEL]
                    [--stream] [--no-stream] [--temperature TEMPERATURE]
                    [--non-interactive] [--no-cache] [--cache-dir CACHE_DIR]
                    [--context-strategy {aggressive,balanced,full}]
                    [--summary-length SUMMARY_LENGTH] [--recent-chapters RECENT_CHAPTERS]
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
  --context-strategy {aggressive,balanced,full}
                        Strategy for managing context window (default: aggressive)
  --summary-length SUMMARY_LENGTH
                        Target word count for chapter summaries (default: 250)
  --recent-chapters RECENT_CHAPTERS
                        Number of recent full chapters to keep in context (default: 2)
  --max-context-tokens MAX_CONTEXT_TOKENS
                        Maximum context tokens to maintain (default: 8000)
```

## Project Structure

```
EQ-Author/
├── eq_author.py          # Main CLI application
├── streamlit_app.py      # Streamlit web interface
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
- Enhanced Streamlit UI with real-time streaming and feedback integration
- Added support for multiple API providers (OpenRouter, LM Studio, etc.)
- Improved error handling and troubleshooting tips
- Added prompt caching to reduce API calls
- Enhanced CLI with more configuration options
