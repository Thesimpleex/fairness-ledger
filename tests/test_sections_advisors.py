from __future__ import annotations

from itertools import pairwise

from fairness_ledger.advisors import UNATTRIBUTED, MentionIndex, canonical_house, find_mentions
from fairness_ledger.document import html_to_document
from fairness_ledger.sections import SegmentIndex, find_headings, segment_document


def test_house_lexicon_and_aliases():
    text = "Morgan Stanley & Co. LLC and J.P. Morgan Securities advised; Evercore Group and PJT Partners."
    houses = [m.house for m in find_mentions(text)]
    assert houses == ["Morgan Stanley", "J.P. Morgan", "Evercore", "PJT Partners"]
    assert canonical_house("Goldman Sachs") == "Goldman Sachs"
    assert canonical_house("Nobody & Sons") is None


def test_textbook_titles_are_not_advisor_mentions():
    assert find_mentions("see Duff & Phelps Cost of Capital Navigator (2019)") == []


def test_mention_index_lookups():
    text = "Lazard did x. Later, Centerview did y."
    index = MentionIndex.build(text)
    assert [m.house for m in index.between(0, len(text))] == ["Lazard", "Centerview"]
    assert index.last_before(15).house == "Lazard"
    assert index.last_before(len(text)).house == "Centerview"
    assert index.last_before(15, floor=5) is None


HTML = (
    "<p><a href='#o'>Opinion of Lazard Frères &amp; Co. LLC</a> 57</p>"  # table of contents entry
    "<p>Background of the Merger</p>"
    "<p>The parties spoke on many days.</p>"
    "<p><b>Opinion of Lazard Frères &amp; Co. LLC</b></p>"
    "<p>Lazard used discount rates ranging from 7.0% to 8.0%.</p>"
    "<p><b>Certain Unaudited Prospective Financial Information</b></p>"
    "<p>Management projections follow.</p>"
)


def test_headings_drop_table_of_contents_and_find_segments():
    doc = html_to_document("s", HTML)
    heads = find_headings(doc)
    assert [h.kind for h in heads] == ["background", "opinion", "end"]
    index = MentionIndex.build(doc.text)
    segments = segment_document(doc, index)
    seg_index = SegmentIndex(segments)
    pos = doc.text.index("discount rates ranging")
    inside = seg_index.at(pos)
    assert inside.kind == "opinion" and inside.advisor == "Lazard"
    after = seg_index.at(doc.text.index("Management projections"))
    assert after.kind == "other" and after.advisor == UNATTRIBUTED
    assert seg_index.at(doc.text.index("many days")).kind == "background"
    for a, b in pairwise(segments):
        assert a.end == b.start
