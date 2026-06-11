"""
Batch PII Masker Runner
========================
Reads english_queries.txt, sends all queries to the /mask/batch endpoint,
and saves the masked results to masked_output.txt and masked_output.csv.

Usage
-----
    python run_pii_masker.py

    # Custom file or API URL:
    python run_pii_masker.py --input english_queries.txt --url http://localhost:8000

Requirements
------------
    pip install requests
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import requests


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "english_queries.txt"
DEFAULT_URL   = "http://localhost:8000"
TXT_OUTPUT    = "masked_output.txt"
CSV_OUTPUT    = "masked_output.csv"


# ── Parse queries from file ───────────────────────────────────────────────────

def parse_queries(filepath: str) -> list[dict]:
    """
    Parses the text file into a list of {grievance_no, text} dicts.
    Each entry is separated by a blank line and formatted as:
        GRIEVANCENO: description text
    """
    content = Path(filepath).read_text(encoding="utf-8")
    blocks  = [b.strip() for b in content.split("\n\n") if b.strip()]

    queries = []
    for block in blocks:
        if ": " in block:
            grievance_no, _, text = block.partition(": ")
            queries.append({"grievance_no": grievance_no.strip(), "text": text.strip()})
        else:
            print(f"  [WARN] Skipping unrecognised block: {block[:60]}...")

    return queries


# ── Call API ──────────────────────────────────────────────────────────────────

def call_batch_api(texts: list[str], base_url: str) -> list[str]:
    url     = f"{base_url.rstrip('/')}/mask/batch"
    payload = {"texts": texts, "monitor": False}

    print(f"  Sending {len(texts)} queries to {url} ...")
    try:
        resp = requests.post(url, json=payload, timeout=400)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to {url}")
        print("  Make sure your API is running:  uvicorn api:app --reload --port 8000")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] API returned an error: {e}")
        print(f"  Response: {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()
    return [item["masked_text"] for item in data["results"]]


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_txt(queries: list[dict], masked_texts: list[str], filepath: str):
    lines = []
    for q, masked in zip(queries, masked_texts):
        lines.append(f"{q['grievance_no']}: {masked}")
        lines.append("")          # blank line separator
    Path(filepath).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved text output  → {filepath}")


def save_csv(queries: list[dict], masked_texts: list[str], filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["GrievanceNo", "OriginalText", "MaskedText"])
        for q, masked in zip(queries, masked_texts):
            writer.writerow([q["grievance_no"], q["text"], masked])
    print(f"  Saved CSV output   → {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all queries through the PII masker batch API.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to english_queries.txt")
    parser.add_argument("--url",   default=DEFAULT_URL,   help="Base URL of the PII masker API")
    args = parser.parse_args()

    print(f"\n[1/4] Reading queries from: {args.input}")
    queries = parse_queries(args.input)
    print(f"      Found {len(queries)} queries.")

    print(f"\n[2/4] Calling /mask/batch ...")
    texts        = [q["text"] for q in queries]
    masked_texts = call_batch_api(texts, args.url)
    print(f"      Done. {len(masked_texts)} responses received.")

    print(f"\n[3/4] Saving outputs ...")
    save_txt(queries, masked_texts, TXT_OUTPUT)
    save_csv(queries, masked_texts, CSV_OUTPUT)

    print(f"\n[4/4] All done! ✓")
    print(f"      {len(queries)} queries masked successfully.\n")


if __name__ == "__main__":
    main()