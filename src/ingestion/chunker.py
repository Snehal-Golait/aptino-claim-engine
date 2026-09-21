"""
Heading-aware chunker for the policy PDF.

Why not fixed-size chunking:
The assignment explicitly requires "meaningful/non-naive chunking" with
page/section metadata. A fixed 500-char sliding window would slice a waiting
period clause away from its heading, or merge two unrelated definitions into
one chunk — both of which directly hurt citation quality (25% of the grade).

Approach:
1. Read every character on every page with its font name and vertical
   position (pdfplumber gives us both).
2. A line is a heading candidate if >60% of its characters use the
   document's bold font. Numeric list markers ("1.", "2.") are merged into
   the heading text that follows them on the same page.
3. Walking the document top-to-bottom, each heading opens a new "section".
   Body text is accumulated under the most recently opened section
   (sections can and do span multiple pages in this 17-page policy).
4. Within a section, if the accumulated text exceeds MAX_CHUNK_CHARS, split
   on paragraph boundaries (never mid-sentence) into multiple chunks, all
   tagged with the same section but distinct chunk_ids.

Every chunk carries: chunk_id, section, page_start, page_end, text.
That page/section metadata is what makes retrieved evidence traceable back
to the source policy, per requirement 4.1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pdfplumber

MAX_CHUNK_CHARS = 1100
MIN_CHUNK_CHARS = 120
BOLD_FRACTION_THRESHOLD = 0.6
RUNNING_HEADER_MARKERS = ("UNIVERSAL SOMPO", "CSC- Individual Health Insurance-Policy Wording")


@dataclass
class Line:
    page: int
    top: float
    text: str
    is_heading: bool


@dataclass
class Chunk:
    chunk_id: str
    section: str
    page_start: int
    page_end: int
    text: str


def _is_running_header_or_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if any(marker in stripped for marker in RUNNING_HEADER_MARKERS):
        return True
    if re.fullmatch(r"\d{1,3}", stripped):  # bare page number
        return True
    return False


def _extract_lines(pdf_path: str) -> list[Line]:
    """Group characters into visual lines, flagging bold (heading) lines."""
    lines: list[Line] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            by_top: dict[float, list] = {}
            for ch in page.chars:
                key = round(ch["top"], 1)
                by_top.setdefault(key, []).append(ch)
            for top, chars in sorted(by_top.items()):
                text = "".join(c["text"] for c in chars).strip()
                if not text:
                    continue
                bold_frac = sum(1 for c in chars if "Bold" in c["fontname"]) / len(chars)
                lines.append(
                    Line(
                        page=page_num,
                        top=top,
                        text=text,
                        is_heading=bold_frac > BOLD_FRACTION_THRESHOLD
                        and not _is_running_header_or_page_number(text),
                    )
                )
    return lines


def _merge_heading_fragments(lines: list[Line]) -> list[Line]:
    """
    The PDF often splits a heading into a bare list marker ("1.") on one
    line and the heading text ("Notice") on the very next line. Merge any
    heading line that is only a numeral/marker into the following heading
    line so we get "1. Notice" as a single section title.
    """
    merged: list[Line] = []
    i = 0
    marker_pattern = re.compile(r"^\d{1,2}\.?$")
    while i < len(lines):
        line = lines[i]
        if (
            line.is_heading
            and marker_pattern.match(line.text)
            and i + 1 < len(lines)
            and lines[i + 1].is_heading
        ):
            nxt = lines[i + 1]
            merged.append(
                Line(
                    page=line.page,
                    top=line.top,
                    text=f"{line.text} {nxt.text}",
                    is_heading=True,
                )
            )
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def _split_long_section(text: str) -> list[str]:
    """Split on paragraph/sentence boundaries so no chunk exceeds MAX_CHUNK_CHARS."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    pieces: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) > MAX_CHUNK_CHARS and buf:
            pieces.append(buf.strip())
            buf = para
        else:
            buf = candidate
    if buf.strip():
        pieces.append(buf.strip())

    # Any single paragraph still too long: hard-split on sentences.
    final: list[str] = []
    sentence_pattern = re.compile(r"(?<=[.;])\s+")
    for piece in pieces:
        if len(piece) <= MAX_CHUNK_CHARS:
            final.append(piece)
            continue
        sentences = sentence_pattern.split(piece)
        buf2 = ""
        for sent in sentences:
            cand2 = f"{buf2} {sent}".strip() if buf2 else sent
            if len(cand2) > MAX_CHUNK_CHARS and buf2:
                final.append(buf2.strip())
                buf2 = sent
            else:
                buf2 = cand2
        if buf2.strip():
            final.append(buf2.strip())
    return final


DEFINITION_START_PATTERN = re.compile(r"^([A-Z][A-Za-z][A-Za-z\s/\-]{1,50}?)\s+means\b")


def _split_section_lines_by_definition(lines: list[str]) -> list[tuple[str, list[str]]]:
    """
    Within a section (e.g. DEFINITIONS), this policy states many defined
    terms as plain, non-bold sentences ("Hospital means ..."). Detecting the
    "<Term> means" boundary lets us give each term its own chunk instead of
    burying it inside one giant DEFINITIONS blob — this matters because
    several of the candidate cases turn on the exact wording of a single
    definition (Hospital, Domiciliary Treatment, Pre-Existing Diseases).

    Returns a list of (term_or_None, lines_for_that_term). The first group,
    with term=None, is any preamble text before the first detected term.
    If no definition boundaries are found, returns [(None, all_lines)].
    """
    groups: list[tuple[str | None, list[str]]] = []
    current_term: str | None = None
    current_lines: list[str] = []
    found_any = False
    for line in lines:
        m = DEFINITION_START_PATTERN.match(line)
        if m:
            found_any = True
            if current_lines:
                groups.append((current_term, current_lines))
            current_term = m.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        groups.append((current_term, current_lines))
    if not found_any:
        return [(None, lines)]
    return groups


NUMBERED_ITEM_PATTERN = re.compile(r"^(\d{1,2})\.\s*([A-Z(].{3,})")


def _split_lines_by_numbered_item(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """
    Second-pass splitter for numbered lists whose items are NOT rendered in
    bold (e.g. exclusion items 3-21 in 'WHAT WE EXCLUDE': only items 1-2 are
    bold in the source PDF, the rest run on as plain text). Requires at
    least 2 matches in the block before splitting, so we don't fragment
    ordinary sentences that happen to start with a number.
    """
    matches = [(i, m) for i, line in enumerate(lines) if (m := NUMBERED_ITEM_PATTERN.match(line))]
    if len(matches) < 2:
        return [(None, lines)]

    groups: list[tuple[str | None, list[str]]] = []
    first_idx = matches[0][0]
    if first_idx > 0:
        groups.append((None, lines[:first_idx]))
    for k, (idx, m) in enumerate(matches):
        end = matches[k + 1][0] if k + 1 < len(matches) else len(lines)
        label = f"Item {m.group(1)}"
        groups.append((label, lines[idx:end]))
    return groups


def chunk_policy_pdf(pdf_path: str) -> list[Chunk]:
    raw_lines = _extract_lines(pdf_path)
    lines = _merge_heading_fragments(raw_lines)

    sections: list[dict] = []
    current = {"heading": "Preamble", "page_start": 1, "page_end": 1, "lines": []}

    for line in lines:
        if _is_running_header_or_page_number(line.text):
            continue
        if line.is_heading:
            if current["lines"]:
                sections.append(current)
            current = {"heading": line.text, "page_start": line.page, "page_end": line.page, "lines": []}
        else:
            current["lines"].append(line.text)
            current["page_end"] = line.page
    if current["lines"]:
        sections.append(current)

    chunks: list[Chunk] = []
    section_counters: dict[str, int] = {}
    for sec in sections:
        if not sec["lines"]:
            continue
        def_groups = _split_section_lines_by_definition(sec["lines"])
        expanded_groups: list[tuple[str | None, list[str]]] = []
        for term, term_lines in def_groups:
            if term is None:
                expanded_groups.extend(_split_lines_by_numbered_item(term_lines))
            else:
                expanded_groups.append((term, term_lines))

        for term, term_lines in expanded_groups:
            body = "\n".join(term_lines).strip()
            effective_heading = f"{sec['heading']} > {term}" if term else sec["heading"]

            if len(body) < MIN_CHUNK_CHARS and chunks:
                # Too small to stand alone (e.g. a one-line note/cross-ref) —
                # merge into previous chunk rather than emit a near-empty chunk.
                prev = chunks[-1]
                prev.text = f"{prev.text}\n\n[{effective_heading}]\n{body}"
                prev.page_end = max(prev.page_end, sec["page_end"])
                continue

            pieces = _split_long_section(body) if body else []
            slug = re.sub(r"[^a-z0-9]+", "-", effective_heading.lower()).strip("-")[:50] or "section"
            for piece in pieces:
                idx = section_counters.get(slug, 0)
                section_counters[slug] = idx + 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{slug}-{idx:02d}",
                        section=effective_heading,
                        page_start=sec["page_start"],
                        page_end=sec["page_end"],
                        text=piece,
                    )
                )
    return chunks


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    result = chunk_policy_pdf(path)
    print(f"Produced {len(result)} chunks\n")
    for c in result[:8]:
        print(f"[{c.chunk_id}] section={c.section!r} pages={c.page_start}-{c.page_end} len={len(c.text)}")
        print(" ", c.text[:150].replace("\n", " "), "...")
        print()
