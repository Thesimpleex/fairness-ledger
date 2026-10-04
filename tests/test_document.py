from __future__ import annotations

import pytest

from fairness_ledger.document import html_to_document


def test_text_blocks_cells_entities_and_whitespace():
    d = html_to_document(
        "x",
        "<p>This summary  highlights\n  selected &amp; more&nbsp;text.</p>"
        "<table><tr><td>7.5%</td><td>to</td><td>9.5%</td></tr></table><p>&#8220;Quote&#8221;</p>",
    )
    assert d.text.splitlines() == [
        "This summary highlights selected & more text.",
        "7.5% | to | 9.5%",
        '"Quote"',
    ]


def test_offsets_map_back_to_html_source():
    source = "<html><body><p>Discount rates ranging from 7.5% to 9.5%.</p></body></html>"
    d = html_to_document("x", source)
    i = d.text.index("7.5%")
    j = d.html_offset(i)
    assert source[j : j + 4] == "7.5%"
    with pytest.raises(IndexError):
        d.html_offset(len(d.text))


def test_cp1252_character_references_and_fallback_decoding():
    d = html_to_document("x", "<p>8.0x&#150;10.0x &#147;ok&#148;</p>")
    assert d.text.startswith("8.0x–10.0x")
    raw = b"<p>a \x96 b</p>"
    assert "–" in html_to_document("x", raw).text


def test_script_and_style_are_dropped_and_lines_flag_bold_and_links():
    d = html_to_document(
        "x",
        "<style>p{}</style><script>var a=1;</script>"
        "<p><b>Opinion of X</b></p><p><a href='#t'>Link text</a></p><p>plain</p>",
    )
    assert "var" not in d.text
    bold, linked, plain = d.lines
    assert bold.bold == 1.0 and linked.linked == 1.0 and plain.bold == 0.0


def test_page_numbers_and_table_of_contents_lines_do_not_split_a_sentence():
    d = html_to_document(
        "x",
        "<p>using discount rates ranging</p><p>101</p><p><a href='#t'>Table of Contents</a></p>"
        "<p>from 9.0% to 10.0% based on</p><p>Next paragraph.</p><p>12</p><p>Another</p>",
    )
    assert (
        d.text
        == "using discount rates ranging from 9.0% to 10.0% based on\nNext paragraph.\nAnother\n"
    )
    assert len(d.html_offsets) == len(d.text)


def test_hash_is_deterministic():
    a = html_to_document("x", "<p>abc</p>")
    b = html_to_document("y", "<p>abc</p>")
    assert a.sha256 == b.sha256
