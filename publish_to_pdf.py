try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Preformatted,
        ListFlowable,
        ListItem,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except Exception:
    # Allow importing this module without ReportLab to test text/YAML utilities
    letter = None  # type: ignore
    SimpleDocTemplate = Paragraph = Spacer = PageBreak = ListFlowable = ListItem = object  # type: ignore
    getSampleStyleSheet = ParagraphStyle = None  # type: ignore
    inch = None  # type: ignore
    colors = None  # type: ignore
    pdfmetrics = None  # type: ignore
    TTFont = None  # type: ignore
    REPORTLAB_AVAILABLE = False
import re
import os
import urllib.request
import warnings
import argparse
import sys
from typing import List, Optional
from xml.sax.saxutils import escape as xml_escape

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Lazy optional; only needed for YAML sources

class PageTemplateWithBackground:
    """
    Custom page template that draws a background color
    """
    def __init__(self, background_color):
        self.background_color = background_color
    
    def on_page(self, canvas, doc):
        """Draw the background color on each page"""
        canvas.saveState()
        canvas.setFillColor(self.background_color)
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
        canvas.restoreState()

def download_font(font_name, font_url):
    """
    Download a font file if it doesn't exist locally.
    
    Args:
        font_name (str): Name of the font file
        font_url (str): URL to download the font from
    Returns:
        bool: True if font is available (downloaded or exists), False otherwise
    """
    if not os.path.exists(font_name):
        try:
            print(f"Downloading {font_name}...")
            urllib.request.urlretrieve(font_url, font_name)
            return True
        except Exception as e:
            warnings.warn(f"Failed to download {font_name}: {e}")
            return False
    return True

def register_fonts():
    """
    Register fonts for use in the PDF.
    Falls back to default fonts if custom fonts are unavailable.
    """
    font_urls = {
        'CrimsonText-Regular.ttf': 'https://raw.githubusercontent.com/google/fonts/main/ofl/crimsontext/CrimsonText-Regular.ttf',
        'CrimsonText-Bold.ttf': 'https://raw.githubusercontent.com/google/fonts/main/ofl/crimsontext/CrimsonText-Bold.ttf',
        'CrimsonText-Italic.ttf': 'https://raw.githubusercontent.com/google/fonts/main/ofl/crimsontext/CrimsonText-Italic.ttf',
        'Cinzel-Regular.ttf': 'https://github.com/NDISCOVER/Cinzel/raw/refs/heads/master/fonts/ttf/Cinzel-Regular.ttf',
        'Cinzel-Bold.ttf': 'https://github.com/NDISCOVER/Cinzel/raw/refs/heads/master/fonts/ttf/Cinzel-Bold.ttf'
    }
    
    # Try to download and register custom fonts
    fonts_available = all(download_font(name, url) for name, url in font_urls.items())
    
    try:
        if fonts_available:
            # Register custom fonts
            pdfmetrics.registerFont(TTFont('CrimsonText', 'CrimsonText-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('CrimsonText-Bold', 'CrimsonText-Bold.ttf'))
            pdfmetrics.registerFont(TTFont('CrimsonText-Italic', 'CrimsonText-Italic.ttf'))
            pdfmetrics.registerFont(TTFont('Cinzel', 'Cinzel-Regular.ttf'))
            pdfmetrics.registerFont(TTFont('Cinzel-Bold', 'Cinzel-Bold.ttf'))
            return True
    except Exception as e:
        warnings.warn(f"Failed to register custom fonts: {e}")
    
    return False

def collapse_duplicate_newlines(text: str) -> str:
    """Collapse sequences of 3+ newlines to double newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)

def clean_text(text: str, preserve_markup: bool = False) -> str:
    """Normalize whitespace while optionally leaving markup intact."""
    if text is None:
        return ""

    # Normalize newlines up-front so downstream processing stays predictable.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if preserve_markup:
        # Keep authoring markup but collapse long blank sections and trim edges.
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.replace("TERMINATE", "")
        return text.strip()

    # Legacy cleanup path for callers that expect aggressively normalized output.
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    text = text.replace("**", "")
    text = text.replace("TERMINATE", "")
    text = text.replace("=", "")
    return text


INLINE_CODE_RE = re.compile(r'`([^`]+)`')
LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
BOLD_RE = re.compile(r'(\*\*|__)(.+?)\1')
ITALIC_RE = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)')
ITALIC_UNDERSCORE_RE = re.compile(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)')
STRIKE_RE = re.compile(r'~~(.+?)~~')


def convert_inline_markup(text: str) -> str:
    """Convert a subset of Markdown inline markup into ReportLab-friendly tags."""
    if not text:
        return ""

    # Basic image handling: surface the alt text and URL inline.
    def _image_to_text(match: re.Match) -> str:
        alt = match.group(1).strip() or "Image"
        url = match.group(2).strip()
        return f"{alt} ({url})"

    text = IMAGE_RE.sub(_image_to_text, text)

    placeholders: dict[str, str] = {}
    counter = 0

    def store_placeholder(replacement: str) -> str:
        nonlocal counter
        token = f"§§TOKEN{counter}§§"
        placeholders[token] = replacement
        counter += 1
        return token

    def _code_repl(match: re.Match) -> str:
        code_text = xml_escape(match.group(1))
        return store_placeholder(f'<font face="Courier">{code_text}</font>')

    def _link_repl(match: re.Match) -> str:
        label_raw = match.group(1).strip()
        href = xml_escape(match.group(2).strip())
        label_markup = convert_inline_markup(label_raw)
        return store_placeholder(f'<link href="{href}">{label_markup}</link>')

    # Reserve placeholders for code spans and links before escaping everything else.
    text = INLINE_CODE_RE.sub(_code_repl, text)
    text = LINK_RE.sub(_link_repl, text)

    escaped = xml_escape(text)

    escaped = STRIKE_RE.sub(r'<strike>\1</strike>', escaped)
    escaped = BOLD_RE.sub(r'<b>\2</b>', escaped)
    escaped = ITALIC_RE.sub(r'<i>\1</i>', escaped)
    escaped = ITALIC_UNDERSCORE_RE.sub(r'<i>\1</i>', escaped)

    # Reinstate any placeholders with their rendered markup.
    for token, markup in placeholders.items():
        escaped = escaped.replace(token, markup)

    return escaped.replace('\n', '<br/>')


def build_flowables_from_markdown(
    markdown_text: str,
    *,
    body_style: 'ParagraphStyle',
    list_item_style: 'ParagraphStyle',
    code_style: 'ParagraphStyle',
    blockquote_style: 'ParagraphStyle',
    heading_styles: dict[int, 'ParagraphStyle']
) -> List[object]:
    """Parse Markdown-flavoured text into platypus flowables."""
    flowables: List[object] = []
    if not markdown_text:
        return flowables

    lines = markdown_text.splitlines()
    current_paragraph: List[str] = []
    list_items: List[ListItem] = []
    list_type: Optional[str] = None  # 'ul' or 'ol'
    list_start: Optional[int] = None
    in_code_block = False
    code_lines: List[str] = []
    i = 0

    def flush_paragraph() -> None:
        nonlocal current_paragraph
        if not current_paragraph:
            return
        text = "\n".join(current_paragraph).strip()
        current_paragraph = []
        if text:
            para_markup = convert_inline_markup(text)
            flowables.append(Paragraph(para_markup, body_style))

    def flush_list() -> None:
        nonlocal list_items, list_type, list_start
        if not list_items:
            return
        kwargs = {
            'leftIndent': list_item_style.leftIndent + 12,
            'bulletFontName': list_item_style.fontName,
            'bulletFontSize': list_item_style.fontSize,
            'spaceBefore': list_item_style.spaceBefore,
            'spaceAfter': list_item_style.spaceAfter,
        }
        if list_type == 'ol':
            kwargs['bulletType'] = '1'
            if list_start is not None:
                kwargs['start'] = list_start
        else:
            kwargs['bulletType'] = 'bullet'
        flowables.append(ListFlowable(list_items, **kwargs))
        list_items = []
        list_type = None
        list_start = None

    def flush_code_block() -> None:
        nonlocal code_lines
        if not code_lines:
            return
        code_text = "\n".join(code_lines).rstrip('\n')
        flowables.append(Preformatted(code_text, code_style))
        code_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_code_block:
            if stripped.startswith('```'):
                flush_code_block()
                in_code_block = False
            else:
                code_lines.append(line)
            i += 1
            continue

        if stripped.startswith('```'):
            flush_paragraph()
            flush_list()
            in_code_block = True
            code_lines = []
            i += 1
            continue

        if stripped == "":
            flush_paragraph()
            flush_list()
            i += 1
            continue

        heading_match = re.match(r'^\s{0,3}(#{1,6})\s+(.+)$', line)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = min(len(heading_match.group(1)), max(heading_styles.keys()))
            heading_style = heading_styles.get(level) or heading_styles[max(heading_styles.keys())]
            heading_text = heading_match.group(2).strip()
            flowables.append(Paragraph(convert_inline_markup(heading_text), heading_style))
            i += 1
            continue

        if re.match(r'^\s*(?:---|\*\*\*|___)\s*$', stripped):
            flush_paragraph()
            flush_list()
            flowables.append(Spacer(1, 0.15 * inch))
            i += 1
            continue

        if stripped.startswith('>'):
            flush_paragraph()
            flush_list()
            quote_lines: List[str] = []
            while i < len(lines):
                q_line = lines[i]
                if q_line.strip().startswith('>'):
                    quote_lines.append(re.sub(r'^\s*>\s?', '', q_line))
                    i += 1
                else:
                    break
            quote_text = "\n".join(quote_lines).strip()
            if quote_text:
                flowables.append(Paragraph(convert_inline_markup(quote_text), blockquote_style))
            continue

        unordered_match = re.match(r'^\s*[-*+]\s+(.*)$', line)
        if unordered_match:
            flush_paragraph()
            if list_type not in (None, 'ul'):
                flush_list()
            list_type = 'ul'
            item_lines = [unordered_match.group(1)]
            i += 1
            while i < len(lines):
                cont_line = lines[i]
                if cont_line.strip() == "":
                    item_lines.append('')
                    i += 1
                    continue
                if re.match(r'^\s{2,}(.*)$', cont_line) and not re.match(r'^\s*[-*+]\s+', cont_line) and not re.match(r'^\s*\d+\.\s+', cont_line):
                    item_lines.append(re.sub(r'^\s{2,}', '', cont_line))
                    i += 1
                    continue
                break
            item_text = "\n".join(item_lines).strip()
            item_markup = convert_inline_markup(item_text)
            list_items.append(ListItem(Paragraph(item_markup, list_item_style)))
            continue

        ordered_match = re.match(r'^\s*(\d+)\.\s+(.*)$', line)
        if ordered_match:
            flush_paragraph()
            if list_type not in (None, 'ol'):
                flush_list()
            if list_type != 'ol' or not list_items:
                list_type = 'ol'
                list_start = int(ordered_match.group(1))
            item_lines = [ordered_match.group(2)]
            i += 1
            while i < len(lines):
                cont_line = lines[i]
                if cont_line.strip() == "":
                    item_lines.append('')
                    i += 1
                    continue
                if re.match(r'^\s{2,}(.*)$', cont_line) and not re.match(r'^\s*[-*+]\s+', cont_line) and not re.match(r'^\s*\d+\.\s+', cont_line):
                    item_lines.append(re.sub(r'^\s{2,}', '', cont_line))
                    i += 1
                    continue
                break
            item_text = "\n".join(item_lines).strip()
            item_markup = convert_inline_markup(item_text)
            list_items.append(ListItem(Paragraph(item_markup, list_item_style)))
            continue

        if list_items:
            flush_list()

        current_paragraph.append(line)
        i += 1

    flush_paragraph()
    flush_list()
    if in_code_block:
        flush_code_block()

    return flowables

def parse_markdown_chapters(md_text: str) -> List[tuple]:
    """
    Parse Markdown text into chapters based on common chapter heading patterns,
    tailored for files like "Erotic Weekend at Blackwood Manor.md".

    Supports headings such as:
    - "# Chapter 1"
    - "## Chapter 2: Title"
    - Plain line: "Chapter 3" at start of a line

    Returns list of tuples: [(chapter_title, chapter_body), ...]
    """
    if not md_text:
        return []

    # Normalize line endings and whitespace similar to clean_text, but keep
    # content intact for chapter slicing
    text = re.sub(r'\r+', '\n', md_text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = text.strip()

    # Pattern 1: Markdown headings like '# Chapter 1', '## Chapter Two: Title'
    pat_md = re.compile(r"(?im)^(?:#{1,6})\s*chapter\s+([^\n]*)\n?")
    # Pattern 2: Plain line starting with 'Chapter 1' (no leading '#')
    pat_plain = re.compile(r"(?im)^chapter\s+([^\n]*)\n?")

    # Find all heading matches with start positions
    matches = []
    for m in pat_md.finditer(text):
        matches.append((m.start(), m.end(), m.group(1).strip()))
    # Add plain matches that are not already covered by md matches
    for m in pat_plain.finditer(text):
        # Skip if this position already captured by a md heading
        if any(s <= m.start() < e for s, e, _ in matches):
            continue
        matches.append((m.start(), m.end(), m.group(1).strip()))

    # Sort by position in text
    matches.sort(key=lambda t: t[0])
    chapters: List[tuple] = []
    if not matches:
        return chapters

    for idx, (start, end, tail) in enumerate(matches):
        next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        body = text[end:next_start].strip()
        # Build a nice chapter title. If tail starts with a number or text, keep it.
        title_tail = tail.strip()
        # Remove leading punctuation like ':' or '-' left after "Chapter X:"
        title_tail = re.sub(r"^[:\-\u2014\u2013\s]+", "", title_tail)
        ch_title = f"Chapter {title_tail or str(idx + 1)}"
        chapters.append((ch_title, body))

    return chapters

def _normalize_for_compare(s: str) -> str:
    """Normalize text for robust comparison across section boundaries.
    - Trim
    - Remove leading ellipses and dots
    - Strip leading quotes/dashes
    - Collapse internal whitespace
    - Drop trailing punctuation/quotes/dashes
    - Lowercase for case-insensitive compare
    """
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r'^(?:\u2026|\.{3,})\s*', '', s)  # leading ellipsis/dots
    s = s.lstrip('“”"\'\u2018\u2019\u2014\u2013- ')
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[\s\.,;:!\?\u2014\u2013\-"“”\'\u2018\u2019]+$', '', s)
    return s.lower()

def _normalize_line_for_overlap(line: str) -> str:
    return _normalize_for_compare(line)

def join_sections_with_overlap(sections: List[str]) -> str:
    """Join multiple text sections into a single string, removing any
    repeated overlap where the last line of one section is repeated as the
    first line of the next section, and dropping dash-led continuation lines.

    Implementation notes:
    - Determine overlap using original lines (before any newline collapsing).
    - If a section's first non-empty line begins with an em/en dash, that
      single line is dropped outright.
    - Only after removals do we apply dash/newline normalization.
    """
    out_parts: List[str] = []
    last_line_norm: Optional[str] = None

    for block in sections:
        if not block:
            continue

        # Work from original lines for overlap/dash-led detection
        orig_lines = block.splitlines()
        # indices of first/last non-empty lines
        orig_first_idx = 0
        while orig_first_idx < len(orig_lines) and str(orig_lines[orig_first_idx]).strip() == "":
            orig_first_idx += 1
        orig_last_idx = len(orig_lines) - 1
        while orig_last_idx >= 0 and str(orig_lines[orig_last_idx]).strip() == "":
            orig_last_idx -= 1

        # Prepare a working copy of lines
        working = list(orig_lines)

        # Drop first line if it's dash-led continuation
        drop_first = False
        if orig_first_idx < len(working):
            if re.match(r"^\s*[\u2014\u2013—]\s+", working[orig_first_idx] or ""):
                drop_first = True

        # Also drop first line if it duplicates previous section's last line
        if not drop_first and last_line_norm is not None and orig_first_idx < len(working):
            if _normalize_for_compare(working[orig_first_idx]) == last_line_norm:
                drop_first = True

        if drop_first and orig_first_idx < len(working):
            working.pop(orig_first_idx)

        # Now form block text and apply normalization transforms
        block_proc = "\n".join(working)
        # Remove standalone dash lines
        block_proc = re.sub(r"(?m)^\s*[\u2014\u2013—]\s*$\n?", "", block_proc)
        # Remove newlines before dashes
        block_proc = re.sub(r"\n+(?=\s*[\u2014\u2013—])", "", block_proc)
        # Replace dashes with space
        block_proc = re.sub(r"[\u2014\u2013—]", " ", block_proc)
        # Collapse single in-sentence newlines
        block_proc = re.sub(r"(?<!\n)\n(?!\n)", " ", block_proc)

        joined = block_proc.strip()
        if joined:
            if out_parts:
                out_parts.append("")
            out_parts.append(joined)

        # Update last_line_norm using the original last non-empty line
        if orig_last_idx >= 0:
            last_line_norm = _normalize_for_compare(orig_lines[orig_last_idx]) or None

    return "\n".join(out_parts).strip()

def parse_yaml_story(src_path: str) -> Optional[dict]:
    """Parse a YAML file and extract a simple story structure if present.

    Currently supports Prelude files with structure:
      prelude:
        title: <str>
        parts: [ { text: <str>, ... }, ... ]

    Returns a dict with keys: 'doc_title', 'chapters' where chapters is a
    list of tuples (chapter_title, chapter_text).
    """
    if yaml is None:
        raise RuntimeError(
            "YAML source provided but PyYAML is not installed. Install PyYAML or use a text/markdown source.")
    with open(src_path, 'r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        return None

    # Helper to extract from a container that has title + parts
    def _extract_from_container(container: dict, default_title: str = ""):
        c_title = str(container.get('title') or default_title or '').strip()
        texts: List[str] = []
        for part in container.get('parts', []):
            if isinstance(part, dict) and isinstance(part.get('text'), str):
                texts.append(part['text'])
        if texts:
            combined = join_sections_with_overlap(texts)
            ch_title = c_title or default_title or 'Chapter'
            return c_title or default_title, [(ch_title, combined)]
        return None

    # Prelude support
    prelude = data.get('prelude') if isinstance(data.get('prelude'), dict) else None
    if prelude and isinstance(prelude.get('parts'), list):
        res = _extract_from_container(prelude, default_title='Prelude')
        if res:
            doc_t, chapters = res
            chapter_title = chapters[0][0]
            # Prefix with Prelude: if not already
            if not chapter_title.lower().startswith('prelude'):
                chapter_title = f"Prelude: {chapter_title}"
            return {
                'doc_title': doc_t or 'Prelude',
                'chapters': [(chapter_title, chapters[0][1])]
            }

    # Top-level title + parts support (e.g., Test.yaml)
    if isinstance(data.get('parts'), list):
        res = _extract_from_container(data, default_title=str(data.get('title') or '').strip() or 'Untitled')
        if res:
            doc_t, chapters = res
            return {
                'doc_title': doc_t,
                'chapters': [(chapters[0][0], chapters[0][1])]
            }

    # Chapter support: chapter: { title, number, parts: [...] }
    if isinstance(data.get('chapter'), dict):
        chapter = data['chapter']
        res = _extract_from_container(chapter, default_title=str(chapter.get('title') or '').strip() or 'Chapter')
        if res:
            doc_t, chapters = res
            num = chapter.get('number')
            ch_title = chapters[0][0]
            if num is not None:
                ch_title = f"{num}: {ch_title}"
            return {
                'doc_title': doc_t,
                'chapters': [(ch_title, chapters[0][1])]
            }

    return None

def process_story_to_pdf(input_text, output_filename='story.pdf', title='My Story'):
    """
    Process text into chapters and create a beautifully formatted PDF Story.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed; cannot build PDF. Install requirements or run non-PDF tests only.")
    # Try to register custom fonts
    custom_fonts = register_fonts()
    
    # Clean the input text
    input_text = collapse_duplicate_newlines(input_text)
    input_text = clean_text(input_text, preserve_markup=True)
    
    # Define eye-friendly colors
    background_color = colors.HexColor('#FBFAF5')  # Softer neutral background
    text_color = colors.Color(0.133, 0.133, 0.133)  # Soft black
    
    class DocumentWithBackground(SimpleDocTemplate):
        def handle_pageBegin(self):
            self.canv.saveState()
            self.canv.setFillColor(background_color)
            self.canv.rect(0, 0, letter[0], letter[1], fill=1)
            self.canv.restoreState()
            super().handle_pageBegin()
    
    # Create the PDF document
    doc = DocumentWithBackground(
        output_filename,
        pagesize=letter,
        rightMargin=1.25*inch,
        leftMargin=1.25*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles with appropriate fonts
    title_style = ParagraphStyle(
        'StoryTitle',
        fontName='Cinzel-Bold' if custom_fonts else 'Times-Bold',
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        leading=36,
        textColor=text_color
    )
    
    chapter_style = ParagraphStyle(
        'ChapterTitle',
        fontName='Cinzel' if custom_fonts else 'Times-Bold',
        fontSize=18,
        spaceAfter=30,
        spaceBefore=30,
        alignment=1,
        leading=28,
        textColor=text_color
    )
    
    body_style = ParagraphStyle(
        'StoryBody',
        fontName='CrimsonText' if custom_fonts else 'Times-Roman',
        fontSize=12,
        leading=18,
        spaceAfter=12,
        firstLineIndent=24,
        alignment=0,
        textColor=text_color
    )

    list_item_style = ParagraphStyle(
        'StoryListItem',
        parent=body_style,
        firstLineIndent=0,
        leftIndent=18,
        spaceBefore=0,
        spaceAfter=6,
    )

    blockquote_style = ParagraphStyle(
        'StoryBlockQuote',
        parent=body_style,
        leftIndent=24,
        rightIndent=12,
        firstLineIndent=0,
        spaceBefore=12,
        spaceAfter=12,
        textColor=colors.Color(0.25, 0.25, 0.25)
    )

    code_style = ParagraphStyle(
        'StoryCodeBlock',
        parent=body_style,
        fontName='Courier',
        fontSize=10,
        leading=14,
        firstLineIndent=0,
        leftIndent=18,
        backColor=colors.Color(0.95, 0.95, 0.92),
        spaceBefore=6,
        spaceAfter=12,
    )

    heading_styles = {
        1: ParagraphStyle(
            'StoryHeading1',
            parent=body_style,
            fontName=chapter_style.fontName,
            fontSize=16,
            leading=24,
            firstLineIndent=0,
            spaceBefore=18,
            spaceAfter=12,
        ),
        2: ParagraphStyle(
            'StoryHeading2',
            parent=body_style,
            fontName=chapter_style.fontName,
            fontSize=14,
            leading=20,
            firstLineIndent=0,
            spaceBefore=16,
            spaceAfter=10,
        ),
        3: ParagraphStyle(
            'StoryHeading3',
            parent=body_style,
            fontName=chapter_style.fontName,
            fontSize=13,
            leading=18,
            firstLineIndent=0,
            spaceBefore=14,
            spaceAfter=8,
        ),
    }
    
    # Store the elements that will make up our document
    elements = []
    
    # Add title page
    elements.append(Spacer(1, 3*inch))
    elements.append(Paragraph(xml_escape(title), title_style))
    elements.append(PageBreak())
    
    # Extract chapters from Markdown/text; fallback to whole text
    parsed_chapters = parse_markdown_chapters(input_text)
    fallback_single = False
    if not parsed_chapters:
        fallback_single = True
        parsed_chapters = [("Chapter 1", clean_text(input_text, preserve_markup=True))]

    # Process each chapter
    for i, (chapter_title, content) in enumerate(parsed_chapters, 1):
        if fallback_single:
            chapter_title = f"Chapter {i}"

        # Add chapter title with proper spacing
        elements.append(Spacer(1, inch))
        elements.append(Paragraph(xml_escape(chapter_title), chapter_style))
        elements.append(Spacer(1, 0.5*inch))

        # Process chapter content - strip any chapter heading that might be in the body
        if content:
            # Remove chapter heading patterns from content to prevent double-titling
            content_clean = re.sub(r'(?im)^#{1,6}\s*chapter\s+[^\n]*\n*', '', content)
            content_clean = re.sub(r'(?im)^chapter\s+[^\n]*\n*', '', content_clean)
            
            chapter_flowables = build_flowables_from_markdown(
                clean_text(content_clean, preserve_markup=True),
                body_style=body_style,
                list_item_style=list_item_style,
                code_style=code_style,
                blockquote_style=blockquote_style,
                heading_styles=heading_styles,
            )
            elements.extend(chapter_flowables)
        
        # Add page break after each chapter
        elements.append(PageBreak())
    
    # Build the PDF document
    doc.build(elements)
    
    return output_filename



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a markdown/text story or a directory of chapter files to a nicely formatted PDF.')
    parser.add_argument('source', help='Path to the source markdown/text file, YAML file, or a directory containing chapter_XX.md files')
    parser.add_argument('-o', '--output', help='Output PDF path (overrides automatic naming). Example: story_output/mystory.pdf')
    parser.add_argument('-t', '--title', help='Title to use in the PDF. Required when source is a directory; otherwise defaults to source filename')
    args = parser.parse_args()

    src = args.source
    if not os.path.exists(src):
        print(f"Error: source file does not exist: {src}")
        sys.exit(2)

    story_from_yaml = None
    input_text = None
    doc_title = None

    # If source is a directory, gather ordered chapter files: chapter_01.md, chapter_02.md, ...
    if os.path.isdir(src):
        if not args.title or not str(args.title).strip():
            print("Error: --title is required when source is a directory.")
            sys.exit(2)

        dir_entries = []
        try:
            dir_entries = os.listdir(src)
        except Exception as e:
            print(f"Failed to list directory: {e}")
            sys.exit(1)

        # Match files named chapter_XX.md (case-insensitive), XX may be 1 or more digits
        pat = re.compile(r"^chapter_(\d+)\.md$", re.IGNORECASE)
        matches = []
        for name in dir_entries:
            m = pat.match(name)
            if m:
                try:
                    num = int(m.group(1))
                except Exception:
                    continue
                matches.append((num, name))

        if not matches:
            print("Error: No chapter files found in directory. Expected files named like 'chapter_01.md'.")
            sys.exit(2)

        matches.sort(key=lambda t: t[0])
        parts: List[str] = []
        for num, fname in matches:
            fpath = os.path.join(src, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    content = fh.read()
            except Exception as e:
                print(f"Failed to read chapter file '{fname}': {e}")
                sys.exit(1)
            content = collapse_duplicate_newlines(content)
            # Build a combined text that our splitter understands
            parts.append(f"Chapter {num}\n\n{content}\n")

        input_text = collapse_duplicate_newlines("\n\n".join(parts))
        doc_title = args.title.strip()

        # Determine output path defaults for directory case
        if args.output:
            out_path = args.output
        else:
            out_dir = os.path.join(src, 'story_output')
            os.makedirs(out_dir, exist_ok=True)
            safe_name = f"{doc_title}.pdf"
            out_path = os.path.join(out_dir, safe_name)

        # Basic cleanup used previously
        input_text = clean_text(collapse_duplicate_newlines(input_text), preserve_markup=True)

        out_file = process_story_to_pdf(input_text, out_path, doc_title)
        print(f"Story saved to {out_file}")
        sys.exit(0)

    ext = os.path.splitext(src)[1].lower()
    if ext in ('.yaml', '.yml'):
        try:
            story_from_yaml = parse_yaml_story(src)
        except Exception as e:
            print(f"Failed to parse YAML: {e}")
            sys.exit(1)

        if story_from_yaml:
            # Build a minimal input_text that our existing chapter splitter understands
            chapters = story_from_yaml['chapters']
            contents: List[str] = []
            for ch_title, ch_text in chapters:
                # Ensure we mark the chapter for the splitter
                contents.append(f"Chapter {ch_title}\n\n{ch_text}\n")
            input_text = collapse_duplicate_newlines("\n\n".join(contents))
            doc_title = story_from_yaml.get('doc_title')
        else:
            # Fallback: treat as plain text if no known structure
            try:
                with open(src, 'r', encoding='utf-8') as fh:
                    input_text = collapse_duplicate_newlines(fh.read())
            except Exception as e:
                print(f"Failed to read source file: {e}")
                sys.exit(1)
    else:
        # Read the source file (markdown/text)
        try:
            with open(src, 'r', encoding='utf-8') as fh:
                input_text = collapse_duplicate_newlines(fh.read())
        except Exception as e:
            print(f"Failed to read source file: {e}")
            sys.exit(1)

    # Basic cleanup used previously
    input_text = clean_text(collapse_duplicate_newlines(input_text), preserve_markup=True)

    # Determine title and output path
    title = args.title if args.title else (doc_title or os.path.splitext(os.path.basename(src))[0])
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(os.path.dirname(src), 'story_output')
        os.makedirs(out_dir, exist_ok=True)
        safe_name = f"{title}.pdf"
        out_path = os.path.join(out_dir, safe_name)

    out_file = process_story_to_pdf(input_text, out_path, title)
    print(f"Story saved to {out_file}")
