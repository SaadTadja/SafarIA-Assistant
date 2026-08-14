"""One-off script: extract and clean Rules 85/87/90 (Schedules & Delays, Denied Boarding
Compensation, Refunds) from the official ATPCO tariff PDF into a clean .txt file for RAG
ingestion. Run once; not part of the app's runtime.
"""

import re
from pathlib import Path

from pypdf import PdfReader

SRC = Path("Docs(for retrieving)/Refund policy/ATPCO 02.pdf")
OUT = Path("Docs(for retrieving)/Refund policy/ATPCO_rules_85_87_90_extract.txt")

NOISE_RE = re.compile(r"^\d+\s*\|\s*P\s*a\s*g\s*e", re.IGNORECASE)


def clean_page(text: str) -> str:
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("Tariff:") or s.startswith("Carrier:"):
            continue
        if NOISE_RE.match(s):
            continue
        kept.append(s)
    return "\n".join(kept)


RULE_HEADER_RE = re.compile(r"^Rule (\d+) (.+)$")


def split_into_rule_blocks(combined: str) -> list[tuple[str, str]]:
    """Split into (title, body) per rule.

    Each page repeats its rule title as a running header, so splitting on any header line
    would yield one spurious block per page. Track the rule number and split only when it
    changes; the header line is dropped either way.
    """
    current_rule_num = None
    current_title = None
    blocks: list[list[str]] = []
    titles: list[str] = []

    for line in combined.splitlines():
        m = RULE_HEADER_RE.match(line)
        if m:
            rule_num = m.group(1)
            if rule_num != current_rule_num:
                current_rule_num = rule_num
                current_title = f"Rule {m.group(1)} - {m.group(2)}"
                titles.append(current_title)
                blocks.append([])
            continue  # never keep the header line itself as content

        if blocks:
            blocks[-1].append(line)

    return list(zip(titles, ["\n".join(b) for b in blocks]))


# "(A)  CONDITIONS FOR PAYMENT" is a heading; "(1)  PASSENGER HOLDING A TICKET FOR..."
# continuing into prose is not. Both use the same marker, so headings are told apart by
# being short, standalone and free of sentence punctuation.
SUBHEADING_RE = re.compile(r"^\([A-Za-z0-9]+\)\s{2,}[A-Z][A-Z0-9 ,/\-]*$")


def split_rule_into_subsections(title: str, body: str, max_len: int = 1200) -> list[str]:
    """Further split one rule's body on its internal sub-headings, so a ~9,600-character
    rule becomes several ~600-1,200 character chunks instead of one oversized blob that
    would dilute its embedding and dump a wall of legal text on the LLM if retrieved."""
    lines = body.splitlines()
    sub_blocks: list[list[str]] = [[]]
    sub_titles: list[str] = [""]

    for line in lines:
        if len(line) <= 70 and SUBHEADING_RE.match(line):
            sub_titles.append(line.strip())
            sub_blocks.append([])
            continue
        sub_blocks[-1].append(line)

    pieces = []
    for sub_title, sub_lines in zip(sub_titles, sub_blocks):
        text = "\n".join(sub_lines).strip()
        if not text:
            continue
        heading = f"{title}{' - ' + sub_title if sub_title else ''}"
        pieces.append(f"{heading}\n{text}")

    if not pieces:
        return [f"{title}\n{body}".strip()]

    # Merge tiny adjacent pieces: the goal is well-sized chunks, not maximal splitting.
    # Single newline, never blank - a blank line is the loader's chunk boundary.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) <= max_len:
            merged[-1] = merged[-1] + "\n" + piece
        else:
            merged.append(piece)
    return merged


def main():
    reader = PdfReader(SRC)
    pages = range(46, 57)  # Rule 85 (idx 46) through end of Rule 90 (idx 56)
    cleaned = [clean_page(reader.pages[i].extract_text()) for i in pages]
    combined = "\n".join(cleaned)

    header = (
        "Source: Airline Tariff Publishing Company (ATPCO), Passenger Fares and Rules "
        "Tariff No. AT-1, filed on behalf of Royal Air Maroc with the US Department of "
        "Transportation (DOT No. 558). Rules 85 (Schedules, Delays and Cancellations), "
        "87 (Denied Boarding Compensation), and 90 (Refunds).\n\n"
    )

    rule_blocks = split_into_rule_blocks(combined)

    all_pieces: list[str] = []
    for title, body in rule_blocks:
        all_pieces.extend(split_rule_into_subsections(title, body))

    # Blank line between pieces = chunk boundary for the app's paragraph-based chunker.
    output = header + "\n\n".join(all_pieces)
    OUT.write_text(output, encoding="utf-8")
    print(f"Wrote {len(output)} chars in {len(all_pieces)} chunks to {OUT}")
    for piece in all_pieces:
        print(f"  - {len(piece)} chars: {piece.splitlines()[0][:80]}")


if __name__ == "__main__":
    main()
