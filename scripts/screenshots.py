"""Product screenshots rendered from real program output.

Terminal shots run a command, capture its ANSI output and draw it inside a
macOS-style window. Report shots open a generated HTML file and capture it.
Both are rendered by the locally installed Google Chrome through Playwright at
2x device scale, on a transparent background so they sit well on light and
dark GitHub themes.

Requires the ``docs`` extra (``rich``, ``playwright``) and Google Chrome.
"""

from __future__ import annotations

import html
import io
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _terminal_theme(
    bg: str, fg: str, normal: Sequence[str], bright: Sequence[str]
) -> TerminalTheme:
    return TerminalTheme(_rgb(bg), _rgb(fg), [_rgb(c) for c in normal], [_rgb(c) for c in bright])


WINDOW_THEMES = {
    "dark": {
        "window": "#111111",
        "bar": "#1A1A1A",
        "ink": "#F5F5F5",
        "muted": "#9E9E9E",
        "ring": "rgba(255,255,255,0.10)",
        "prompt": "#FF4D5E",
        "dot": "#3A3A3A",
        "shadow": "0 1px 2px rgba(0,0,0,.30), 0 8px 24px rgba(0,0,0,.35)",
        "ansi": _terminal_theme(
            "#111111",
            "#F5F5F5",
            [
                "#9E9E9E",
                "#FF4D5E",
                "#3FB27F",
                "#E3A008",
                "#F5F5F5",
                "#FF4D5E",
                "#CFCFCF",
                "#9E9E9E",
            ],
            [
                "#CFCFCF",
                "#FF7A86",
                "#5CC896",
                "#F0B429",
                "#FFFFFF",
                "#FF7A86",
                "#E6E6E6",
                "#FFFFFF",
            ],
        ),
    },
    "light": {
        "window": "#FFFFFF",
        "bar": "#F5F5F5",
        "ink": "#111111",
        "muted": "#6B6B6B",
        "ring": "rgba(0,0,0,0.10)",
        "prompt": "#C8102E",
        "dot": "#D0D0D0",
        "shadow": "0 1px 2px rgba(0,0,0,.12), 0 2px 6px rgba(0,0,0,.08)",
        "ansi": _terminal_theme(
            "#FFFFFF",
            "#111111",
            [
                "#111111",
                "#C8102E",
                "#1E7B4F",
                "#9A6700",
                "#111111",
                "#C8102E",
                "#3D3D3D",
                "#6B6B6B",
            ],
            [
                "#3D3D3D",
                "#E0283F",
                "#2A9D63",
                "#B07C00",
                "#111111",
                "#E0283F",
                "#111111",
                "#111111",
            ],
        ),
    },
}

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
html, body {{ margin: 0; background: transparent; }}
.stage {{ display: inline-block; padding: 40px 48px 64px; }}
.window {{
  display: inline-block; min-width: 560px; border-radius: 12px; overflow: hidden;
  background: {window};
  box-shadow: 0 0 0 1px {ring}, {shadow};
}}
.bar {{ height: 38px; display: flex; align-items: center; padding: 0 14px;
  background: {bar}; position: relative; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }}
.title {{ position: absolute; left: 0; right: 0; text-align: center; pointer-events: none;
  font: 400 12.5px "Helvetica Neue", Helvetica, sans-serif; color: {muted}; }}
pre {{ margin: 0; padding: 18px 24px 24px; color: {ink}; white-space: pre;
  font: 13px/1.38 ui-monospace, "SF Mono", Menlo, monospace; }}
.prompt {{ color: {prompt}; }}
.cmd {{ font-weight: 600; }}
</style></head><body><div class="stage" id="stage"><div class="window">
<div class="bar"><span class="dot" style="background:{dot}"></span><span class="dot"
style="background:{dot}"></span><span class="dot" style="background:{dot}"></span>
<div class="title">{title}</div></div>
<pre>{prompt_line}{body}</pre></div></div></body></html>
"""


def capture_command(
    command: Sequence[str] | str, columns: int = 100, cwd: str | Path | None = None
) -> str:
    """Run ``command`` with colour forced on and return its ANSI output."""
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    env = {**os.environ, "FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": str(columns)}
    env.pop("NO_COLOR", None)
    result = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout


def terminal_html(
    ansi_output: str,
    command: str | None = None,
    title: str = "",
    theme: str = "dark",
    columns: int = 100,
) -> str:
    """Wrap ANSI output in a self-contained HTML page showing a terminal window."""
    t = WINDOW_THEMES[theme]
    console = Console(
        record=True,
        width=columns,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
    )
    console.print(Text.from_ansi(ansi_output.rstrip("\n")), end="")
    body = console.export_html(theme=t["ansi"], inline_styles=True, code_format="{code}")
    prompt_line = (
        f'<span class="prompt">$</span> <span class="cmd">{html.escape(command)}</span>\n'
        if command
        else ""
    )
    return _PAGE.format(
        title=html.escape(title),
        prompt_line=prompt_line,
        body=body,
        **{k: v for k, v in t.items() if k != "ansi"},
    )


def _shoot(
    page_html: str | None,
    out: Path,
    url: str | None = None,
    selector: str | None = None,
    width: int = 1400,
    full_page: bool = True,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(device_scale_factor=2, viewport={"width": width, "height": 900})
        if url is not None:
            page.goto(url)
        else:
            page.set_content(page_html or "")
        page.evaluate("document.fonts.ready")
        if selector:
            page.locator(selector).screenshot(path=str(out), omit_background=True)
        else:
            page.screenshot(path=str(out), full_page=full_page)
        browser.close()
    return out


def terminal_screenshot(
    command: Sequence[str] | str,
    out: str | Path,
    title: str = "",
    display_command: str | None = None,
    theme: str = "dark",
    columns: int = 100,
    cwd: str | Path | None = None,
) -> Path:
    """Run a real command and save a PNG of its output in a terminal window."""
    ansi = capture_command(command, columns=columns, cwd=cwd)
    shown = display_command or (command if isinstance(command, str) else shlex.join(command))
    page_html = terminal_html(ansi, command=shown, title=title, theme=theme, columns=columns)
    return _shoot(page_html, Path(out), selector="#stage")


def report_screenshot(
    report: str | Path, out: str | Path, width: int = 1280, selector: str | None = None
) -> Path:
    """Save a PNG of a generated HTML report (full page, or one element via ``selector``)."""
    return _shoot(
        None, Path(out), url=Path(report).resolve().as_uri(), selector=selector, width=width
    )
