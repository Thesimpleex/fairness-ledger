"""HTML to text with a character-offset map back to the source HTML.

The text is the single coordinate system of the dataset: every provenance span is a
``[start, end)`` interval of :attr:`Document.text`. Because the conversion is
deterministic, the same cached filing always yields the same text, which
:attr:`Document.sha256` fingerprints. :meth:`Document.html_offset` maps a text
position back to the byte-for-byte location in the original HTML.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from array import array
from dataclasses import dataclass, field
from functools import cached_property
from html.entities import name2codepoint
from html.parser import HTMLParser

_BLOCK = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "tr",
        "ul",
    ]
)
_VOID = frozenset(
    [
        "br",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "area",
        "base",
        "col",
        "embed",
        "source",
        "track",
        "wbr",
    ]
)
_SKIP = frozenset({"script", "style", "head", "title"})
_BOLD_STYLE = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)", re.I)
_TRANSLATE = str.maketrans(
    {
        "\xa0": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        " ": " ",
        "​": "",
        "­": "",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)
_SPACE_RUN = re.compile(r"\s+|\S+")


@dataclass(frozen=True)
class Line:
    """A text line (one block element) with simple typographic cues."""

    start: int
    end: int
    bold: float
    linked: float


@dataclass
class Document:
    """Plain-text view of one filing with offsets back to the HTML source."""

    accession: str
    text: str
    html_offsets: array
    lines: list[Line] = field(default_factory=list)

    @cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def html_offset(self, position: int) -> int:
        """Offset in the source HTML of the character at text ``position``."""
        if not 0 <= position < len(self.text):
            raise IndexError(position)
        return int(self.html_offsets[position])

    def snippet(self, start: int, end: int) -> str:
        return self.text[start:end]

    def line_index(self, position: int) -> int:
        """Index of the line containing ``position``."""
        starts = [ln.start for ln in self.lines]
        return max(0, bisect.bisect_right(starts, position) - 1)


_PAGE_ARTIFACT = re.compile(
    r"^(?:\d{1,3}|[ivxlc]{1,6}|[A-Z]-\d{1,3}|-\s*\d{1,3}\s*-|table\s+of\s+contents|"
    r"index\s+to\s+(?:consolidated\s+)?financial\s+statements)$",
    re.I,
)
_SENTENCE_END = (".", ";", ":", "!", "?", '"', ")")


def _drop_page_artifacts(
    text: str, offsets: array, lines: list[Line]
) -> tuple[str, array, list[Line]]:
    """Remove page numbers and ``Table of Contents`` lines that interrupt running text.

    A paragraph that was split by a page break (``... ranging`` / ``101`` / ``Table of
    Contents`` / ``from 9.0% ...``) is joined into one line again.
    """
    pieces: list[str] = []
    new_offsets = array("I")
    new_lines: list[Line] = []
    length = 0
    dropped_since_kept = False
    for line in lines:
        raw = text[line.start : line.end].strip()
        if _PAGE_ARTIFACT.match(raw):
            dropped_since_kept = True
            continue
        if new_lines:
            joined = dropped_since_kept and not new_lines_text_ends_sentence(pieces)
            sep = " " if joined else "\n"
            pieces.append(sep)
            new_offsets.append(offsets[line.start - 1] if line.start > 0 else 0)
            length += 1
            if joined:
                prev = new_lines.pop()
                start = prev.start
                chunk = text[line.start : line.end]
                pieces.append(chunk)
                new_offsets.extend(offsets[line.start : line.end])
                width_prev, width_new = prev.end - prev.start, line.end - line.start
                total = width_prev + width_new
                new_lines.append(
                    Line(
                        start,
                        length + len(chunk),
                        (prev.bold * width_prev + line.bold * width_new) / total,
                        (prev.linked * width_prev + line.linked * width_new) / total,
                    )
                )
                length += len(chunk)
                dropped_since_kept = False
                continue
        chunk = text[line.start : line.end]
        pieces.append(chunk)
        new_offsets.extend(offsets[line.start : line.end])
        new_lines.append(Line(length, length + len(chunk), line.bold, line.linked))
        length += len(chunk)
        dropped_since_kept = False
    pieces.append("\n")
    new_offsets.append(offsets[-1] if len(offsets) else 0)
    return "".join(pieces), new_offsets, new_lines


def new_lines_text_ends_sentence(pieces: list[str]) -> bool:
    for chunk in reversed(pieces):
        stripped = chunk.rstrip()
        if stripped:
            return stripped.endswith(_SENTENCE_END)
    return True


class _TextBuilder(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", source)]
        self._parts: list[str] = []
        self.offsets = array("I")
        self._length = 0
        self._skip = 0
        self._stack: list[tuple[str, bool, bool]] = []
        self._line_start = 0
        self._bold = 0
        self._linked = 0
        self.lines: list[Line] = []

    # -- helpers ----------------------------------------------------------

    def _here(self) -> int:
        line, col = self.getpos()
        return self._line_starts[line - 1] + col

    def _emit(self, chunk: str, offsets: range | list[int]) -> None:
        if not chunk:
            return
        chunk_t = chunk.translate(_TRANSLATE)
        if len(chunk_t) != len(chunk):
            kept = [o for c, o in zip(chunk, offsets, strict=True) if c not in "​­"]
            offsets = kept
            chunk = chunk_t
        else:
            chunk = chunk_t
        self._parts.append(chunk)
        self.offsets.extend(offsets)
        self._length += len(chunk)

    def _text(self) -> str:
        return "".join(self._parts)

    def _break(self, source_pos: int) -> None:
        if self._length and not self._at_line_start():
            self._close_line()
            self._emit("\n", [source_pos])
            self._line_start = self._length

    def _at_line_start(self) -> bool:
        return self._length == self._line_start

    def _close_line(self) -> None:
        start, end = self._line_start, self._length
        if end > start:
            width = end - start
            self.lines.append(Line(start, end, self._bold / width, self._linked / width))
        self._bold = 0
        self._linked = 0

    def _in_bold(self) -> bool:
        return any(entry[1] for entry in self._stack)

    def _in_link(self) -> bool:
        return any(entry[2] for entry in self._stack)

    def _add_text(self, raw: str, base: int) -> None:
        pieces: list[str] = []
        offs: list[int] = []
        for match in _SPACE_RUN.finditer(raw):
            token = match.group()
            if token[0].isspace():
                last = pieces[-1] if pieces else (self._parts[-1] if self._parts else "\n")
                if last.endswith((" ", "\n")):
                    continue
                pieces.append(" ")
                offs.append(base + match.start())
            else:
                pieces.append(token)
                offs.extend(range(base + match.start(), base + match.end()))
        chunk = "".join(pieces)
        if not chunk:
            return
        before = self._length
        self._emit(chunk, offs)
        added = self._length - before
        if self._in_bold():
            self._bold += added
        if self._in_link():
            self._linked += added

    # -- HTMLParser hooks --------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        pos = self._here()
        if tag in _BLOCK or tag == "br":
            self._break(pos)
        elif tag in ("td", "th") and self._length and not self._at_line_start():
            self._add_text(" | ", pos)
        if tag in _VOID:
            return
        style = dict(attrs).get("style") or ""
        bold = tag in ("b", "strong", "th") or bool(_BOLD_STYLE.search(style))
        link = tag == "a" and dict(attrs).get("href") is not None
        self._stack.append((tag, bold, link))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP or self._skip:
            return
        if tag in _BLOCK or tag == "br":
            self._break(self._here())

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break
        if tag in _BLOCK:
            self._break(self._here())

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._add_text(data, self._here())

    def handle_entityref(self, name: str) -> None:
        if self._skip:
            return
        code = name2codepoint.get(name)
        char = chr(code) if code else f"&{name};"
        self._add_text(char, self._here())

    def handle_charref(self, name: str) -> None:
        if self._skip:
            return
        try:
            code = int(name[1:], 16) if name[:1] in "xX" else int(name)
            char = bytes([code]).decode("cp1252", "replace") if 128 <= code <= 159 else chr(code)
        except (ValueError, OverflowError):
            return
        self._add_text(char, self._here())

    def finish(self) -> tuple[str, array, list[Line]]:
        self._close_line()
        return _drop_page_artifacts(self._text(), self.offsets, self.lines)


def html_to_document(accession: str, raw: bytes | str) -> Document:
    """Convert filing HTML to a :class:`Document`.

    Block elements become line breaks, table cells are joined with `` | ``, runs of
    whitespace collapse to one space, typographic quotes are straightened. Every
    output character keeps the offset of its source character.

    Parameters
    ----------
    accession
        Accession number, stored as the document id.
    raw
        HTML bytes (decoded as UTF-8 with Windows-1252 fallback) or text.
    """
    if isinstance(raw, bytes):
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            source = raw.decode("cp1252", "replace")
    else:
        source = raw
    builder = _TextBuilder(source)
    builder.feed(source)
    builder.close()
    text, offsets, lines = builder.finish()
    if len(offsets) != len(text):
        raise RuntimeError("offset map out of sync with text")
    return Document(accession=accession, text=text, html_offsets=offsets, lines=lines)
