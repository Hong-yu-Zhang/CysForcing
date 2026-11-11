#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download AlphaFold PDB files for seq_id(s) listed in a TSV (parallel),
and verify existing files are valid (not empty / not HTML / looks like PDB).

URL template:
https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.pdb

Output:
{out_dir}/{seq_id}.pdb
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

URL_TMPL = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.pdb"


# -------------------------
# TSV parsing
# -------------------------
def read_seq_ids_from_tsv(tsv_path: str) -> list[str]:
    """Read UniProt accessions from a TSV with a seq_id column, or from a
    headerless file with one accession per line."""
    seq_ids: list[str] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Empty TSV or missing header row.")

        colmap = {name.strip().lower(): name for name in reader.fieldnames}
        if "seq_id" not in colmap:
            if len(reader.fieldnames) != 1:
                raise ValueError(
                    f"TSV header missing 'seq_id'. Got columns: {reader.fieldnames}")
            # Single unnamed column: a plain accession list, header row included.
            seq_ids.append(reader.fieldnames[0].strip())
            seq_ids.extend(
                (row.get(reader.fieldnames[0]) or "").strip() for row in reader)
            seq_ids = [u for u in seq_ids if u]
        else:
            seq_col = colmap["seq_id"]
            for row in reader:
                uid = (row.get(seq_col) or "").strip()
                if uid:
                    seq_ids.append(uid)

    seen = set()
    uniq: list[str] = []
    for x in seq_ids:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# -------------------------
# File validation
# -------------------------
PDB_HINT_PREFIXES = (b"HEADER", b"TITLE", b"ATOM  ", b"HETATM", b"MODEL", b"REMARK", b"TER", b"END")


def is_valid_pdb_file(path: Path, min_size: int = 1024) -> tuple[bool, str]:
    """
    Heuristics to detect bad/empty downloads:
    - file exists and size >= min_size
    - first bytes are not HTML
    - first line looks like PDB-ish record types
    """
    if not path.exists():
        return (False, "missing")

    try:
        size = path.stat().st_size
    except OSError as e:
        return (False, f"stat_error: {e}")

    if size == 0:
        return (False, "empty")
    if size < min_size:
        # Many valid PDBs are larger than 1KB; small files are suspicious (often error pages/truncation)
        # Still allow passing if content clearly looks like PDB.
        pass

    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError as e:
        return (False, f"read_error: {e}")

    head_strip = head.lstrip()
    if head_strip.startswith(b"<!DOCTYPE html") or head_strip.startswith(b"<html") or b"<head" in head[:2000].lower():
        return (False, "html_content")

    # Check first non-empty line prefix
    first_line = head.splitlines()[0].strip() if head.splitlines() else b""
    if first_line.startswith(PDB_HINT_PREFIXES):
        if size < min_size:
            return (True, f"ok_but_small({size})")
        return (True, "ok")

    # Sometimes file begins with many REMARK lines; already covered. If not, treat as suspicious.
    if size >= min_size:
        return (False, "unknown_header")
    return (False, f"too_small({size})")


# -------------------------
# Download logic
# -------------------------
def download_pdb(uid: str, out_dir: Path, overwrite: bool = False, timeout: int = 60):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uid}.pdb"

    url = URL_TMPL.format(uid=uid)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return ("FAIL", uid, f"HTTP {r.status_code} {url}")

            ctype = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                return ("FAIL", uid, f"Got HTML content-type: {url}")

            tmp_path = out_path.with_suffix(".pdb.part")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)

            os.replace(tmp_path, out_path)
            return ("OK", uid, str(out_path))

    except requests.RequestException as e:
        return ("FAIL", uid, f"{type(e).__name__}: {e}")


def _worker(uid: str, out_dir_str: str, overwrite: bool, timeout: int, per_task_delay: float, min_size: int, retries: int):
    out_dir = Path(out_dir_str)
    out_path = out_dir / f"{uid}.pdb"

    # 1) If file exists and overwrite is False, validate it; if bad, redownload.
    if out_path.exists() and not overwrite:
        valid, reason = is_valid_pdb_file(out_path, min_size=min_size)
        if valid:
            return ("SKIP", uid, f"{out_path} ({reason})")
        # invalid: try to redownload (and overwrite)
        overwrite = True

    if per_task_delay and per_task_delay > 0:
        time.sleep(per_task_delay)

    last = None
    for attempt in range(retries + 1):
        status, uid2, msg = download_pdb(uid, out_dir=out_dir, overwrite=True, timeout=timeout)
        last = (status, uid2, msg)

        if status == "OK":
            # verify after download
            valid, reason = is_valid_pdb_file(out_path, min_size=min_size)
            if valid:
                return ("OK", uid, f"{out_path} ({reason})")
            else:
                # remove bad file to avoid keeping junk
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                last = ("FAIL", uid, f"Downloaded but invalid ({reason}): {out_path}")

        # small backoff between retries
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))

    return last


def main():
    ap = argparse.ArgumentParser(description="Parallel download AlphaFold PDBs and validate existing files.")
    ap.add_argument(
        "-i",
        "--input",
        default="data/ood/pred_exp_inter.tsv",
        help="Input TSV/txt/CSV with UniProt accessions (header column seq_id, "
             "or one accession per line).",
    )
    ap.add_argument(
        "-o",
        "--out",
        default="data/ood/af2_pdb",
        help="Output directory.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite even if local file looks valid.")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds (default: 60).")

    ap.add_argument("-j", "--jobs", type=int, default=8, help="Number of parallel workers (default: 8).")
    ap.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Per-task start delay in seconds (default: 0.0).",
    )

    ap.add_argument(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum expected PDB file size in bytes used for validation (default: 1024).",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries when download fails or validation fails (default: 2).",
    )
    ap.add_argument("--print-limit", type=int, default=50, help="Print seq_id list if <= this limit.")
    args = ap.parse_args()

    seq_ids = read_seq_ids_from_tsv(args.input)
    print(f"Found {len(seq_ids)} unique seq_id(s).")
    if len(seq_ids) <= args.print_limit:
        print("seq_id:", ", ".join(seq_ids))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Validation: min_size={args.min_size} bytes; retries={args.retries}")

    ok = fail = skip = 0

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = [
            ex.submit(
                _worker,
                uid,
                str(out_dir),
                args.overwrite,
                args.timeout,
                args.delay,
                args.min_size,
                args.retries,
            )
            for uid in seq_ids
        ]

        for fut in as_completed(futures):
            status, uid, msg = fut.result()
            print(f"[{status}] {uid} -> {msg}")
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                fail += 1

    print(f"Done. OK={ok}, SKIP={skip}, FAIL={fail}")


if __name__ == "__main__":
    main()