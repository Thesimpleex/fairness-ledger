"""Repository banner: name, claim, one evidenced number and the signature chart.

The layout follows the key-figure cards of bcg.com (one large number with a short,
specific explanation) and the editorial restraint of The Economist's charts: white
or near-black surface, black and grey type in Helvetica Neue, one red accent, no
gradients, glows or shadows. The banner is rendered at 1280 x 640 CSS pixels, the
size GitHub uses for social previews, at 2x device scale by the locally installed
Google Chrome through Playwright.

Requires the ``docs`` extra (``playwright``) and Google Chrome.
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

_CSS = """
:root{--bg:#FFFFFF;--ink:#111111;--ink2:#3D3D3D;--muted:#6B6B6B;--rule:#D9D9D9;--accent:#C8102E}
.dark{--bg:#0F0F0F;--ink:#F5F5F5;--ink2:#CFCFCF;--muted:#9E9E9E;--rule:#333333;--accent:#FF4D5E}
html,body{margin:0;background:transparent}
.card{width:1280px;height:640px;box-sizing:border-box;padding:60px 72px 92px;
  background:var(--bg);color:var(--ink);position:relative;display:grid;
  grid-template-columns:548px 1fr;column-gap:48px;
  font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.left{display:flex;flex-direction:column;min-height:0}
.mark{width:32px;height:3px;background:var(--accent);margin:4px 0 16px}
.kicker{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  font-weight:500}
h1{font-weight:300;font-size:{name_px}px;line-height:1;letter-spacing:-.045em;margin:20px 0 0}
.claim{font-size:22px;line-height:1.36;color:var(--ink2);margin:20px 0 0;font-weight:400;
  letter-spacing:-.006em}
.stat{margin-top:auto}
.value{font-size:56px;font-weight:300;color:var(--accent);letter-spacing:-.035em;line-height:1}
.label{font-size:14.5px;line-height:1.45;color:var(--ink2);margin-top:10px;max-width:470px}
.right{display:flex;align-items:center;justify-content:center;min-width:0}
.right img{max-width:100%;max-height:452px}
.foot{position:absolute;left:72px;right:72px;bottom:30px;border-top:1px solid var(--rule);
  padding-top:13px;display:flex;justify-content:space-between;font-size:12.5px;
  color:var(--muted);letter-spacing:.01em}
"""


def _page(
    theme: str,
    name: str,
    claim: str,
    kicker: str,
    value: str,
    label: str,
    chart: Path,
    footer_left: str,
    footer_right: str,
) -> str:
    name_px = 92 if len(name) <= 11 else max(64, int(92 * 11 / len(name)))
    css = _CSS.replace("{name_px}", str(name_px))
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>"
        f"<div class='card {theme}' id='card'><div class='left'>"
        f"<div class='mark'></div><div class='kicker'>{html.escape(kicker)}</div>"
        f"<h1>{html.escape(name)}</h1><p class='claim'>{html.escape(claim)}</p>"
        f"<div class='stat'><div class='value'>{html.escape(value)}</div>"
        f"<div class='label'>{html.escape(label)}</div></div></div>"
        f"<div class='right'><img src='{chart.resolve().as_uri()}' alt=''></div>"
        f"<div class='foot'><span>{html.escape(footer_left)}</span>"
        f"<span>{html.escape(footer_right)}</span></div></div></body></html>"
    )


def render_banner(
    stem: str | Path,
    *,
    name: str,
    claim: str,
    kicker: str,
    value: str,
    label: str,
    chart_light: str | Path,
    chart_dark: str | Path,
    footer_left: str,
    footer_right: str,
) -> list[Path]:
    """Write ``<stem>-light.png`` and ``<stem>-dark.png`` (2560 x 1280 pixels).

    ``value`` and ``label`` must be a real, reproducible result of the repository;
    the chart images are rendered beforehand with ``viz.save_figure``.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(device_scale_factor=2, viewport={"width": 1280, "height": 640})
        for theme, chart in (("light", Path(chart_light)), ("dark", Path(chart_dark))):
            doc = Path(tmp) / f"banner-{theme}.html"
            doc.write_text(
                _page(theme, name, claim, kicker, value, label, chart, footer_left, footer_right),
                encoding="utf-8",
            )
            page.goto(doc.as_uri())
            page.evaluate("document.fonts.ready")
            path = stem.with_name(f"{stem.name}-{theme}.png")
            page.locator("#card").screenshot(path=str(path))
            out.append(path)
        browser.close()
    return out
