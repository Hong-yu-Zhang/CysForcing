#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Join prob values from result.json into an input TSV.

- Input TSV columns must include: seq_id, posi (others kept as-is)
- result.json format: a list of objects, each like:
    {"A2A432.pdb": {"C_32": [0.0647], ..., "Sequence": "..."}}

Mapping rule:
- seq_id -> key "{seq_id}.pdb" in JSON
- posi -> key "C_{posi}" in that PDB dict
- value -> first element of the list (e.g. [0.0647] -> 0.0647)

Output:
- A new TSV that contains all original columns, and a column `prob`
  filled from JSON. If `prob` already exists, it will be replaced.

Also adds an optional `prob_status` column if you pass --with-status.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_prob_index(json_path: str) -> dict[tuple[str, int], float]:
    """
    Build an index: (seq_id, posi) -> prob
    from the JSON described above.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("result.json must be a list at the top level.")

    idx: dict[tuple[str, int], float] = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        for pdb_name, pdb_dict in item.items():
            if not isinstance(pdb_name, str) or not pdb_name.endswith(".pdb"):
                continue
            if not isinstance(pdb_dict, dict):
                continue

            seq_id = pdb_name[:-4]  # strip ".pdb"
            for k, v in pdb_dict.items():
                if not isinstance(k, str) or not k.startswith("C_"):
                    continue
                if k == "Sequence":
                    continue
                # parse position
                try:
                    posi = int(k.split("_", 1)[1])
                except Exception:
                    continue

                # value expected like [0.123]
                prob = None
                if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], (int, float)):
                    prob = float(v[0])
                elif isinstance(v, (int, float)):
                    # tolerate scalar
                    prob = float(v)

                if prob is not None:
                    idx[(seq_id, posi)] = prob

    return idx


def normalize_header_map(fieldnames: list[str]) -> dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames}


def main():
    ap = argparse.ArgumentParser(description="Add/replace prob column in TSV using result.json")
    ap.add_argument("-i", "--tsv", required=True, help="Input TSV path")
    ap.add_argument("-j", "--json", required=True, help="Input result.json path")
    ap.add_argument("-o", "--out", required=True, help="Output TSV path")
    ap.add_argument(
        "--with-status",
        action="store_true",
        help="Add prob_status column (ok / missing_seq / missing_pos / bad_posi)",
    )
    ap.add_argument(
        "--na",
        default="NA",
        help="Value to write when prob is missing (default: NA)",
    )
    args = ap.parse_args()

    prob_idx = load_prob_index(args.json)

    in_path = Path(args.tsv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Input TSV missing header.")

        colmap = normalize_header_map(reader.fieldnames)
        if "seq_id" not in colmap or "posi" not in colmap:
            raise ValueError(f"Input TSV must include 'seq_id' and 'posi'. Got: {reader.fieldnames}")

        seq_col = colmap["seq_id"]
        pos_col = colmap["posi"]

        # prepare output columns: keep original order, ensure prob exists, optionally prob_status
        out_fields = [c for c in reader.fieldnames if c.strip().lower() != "prob" and c.strip().lower() != "prob_status"]
        out_fields.append("prob")
        if args.with_status:
            out_fields.append("prob_status")

        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields, delimiter="\t")
            writer.writeheader()

            # For status reporting
            # set of seq_ids existing in JSON (helps to tell missing_seq vs missing_pos)
            json_seq_ids = {seq for (seq, _pos) in prob_idx.keys()}

            for row in reader:
                uid = (row.get(seq_col) or "").strip()
                pos_raw = (row.get(pos_col) or "").strip()

                status = "ok"
                prob_val: Any = args.na

                try:
                    posi = int(float(pos_raw))  # tolerate "142.0"
                except Exception:
                    posi = None
                    status = "bad_posi"

                if uid and posi is not None:
                    posi_json = posi + 1 
                    key = (uid, posi_json)
                    if key in prob_idx:
                        prob_val = prob_idx[key]
                        status = "ok"
                    else:
                        if uid not in json_seq_ids:
                            status = "missing_seq"
                        else:
                            status = "missing_pos"
                else:
                    if status != "bad_posi":
                        status = "missing_seq" if not uid else "bad_posi"

                # write output row (preserve other columns)
                out_row = {k: row.get(k, "") for k in out_fields}
                out_row["prob"] = prob_val
                if args.with_status:
                    out_row["prob_status"] = status

                writer.writerow(out_row)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()