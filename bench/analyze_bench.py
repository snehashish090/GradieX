#!/usr/bin/env python3
"""Summarise bench/results_all.csv.

    python3 bench/analyze_bench.py bench/results_all.csv

Prints, for each optimisation setting, one row per architecture with the
microseconds-per-sample figure for each engine and the cross-version speedups.
The comparison is only drawn where the engines can express the same architecture:
v0.0.1 has uniform hidden widths, so 784->128x2->10 appears for v0.0.2/v0.0.3 only.

Microseconds come from the harness's own timing loop, which reports a minimum, a
median and a maximum over its trials. The minimum is what the tables compare --
interference only ever adds time -- and the spread column shows how much
interference there was, so a number that moved for reasons other than the code is
visible rather than hidden.
"""

import csv
import sys
from collections import OrderedDict

SETTING_ORDER = [
    "-O2,1",
    "-O2,32",
    "-O2 -fopenmp-simd,1",
    "-O2 -fopenmp-simd,32",
    "-O3,1",
    "-O3,32",
    "-O3 -fopenmp-simd,1",
    "-O3 -fopenmp-simd,32",
]

ARCH_ORDER = [
    "2->8x2->1",
    "20->32x2->3",
    "64->128x3->10",
    "784->128x2->10",
]


def load(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            row["batch"] = int(row["batch"])
            row["params"] = int(row["params"])
            for field in ("min_us", "median_us", "max_us"):
                row[field] = float(row[field])
            rows.append(row)
    return rows


def fmt(x, places=None):
    if x is None:
        return "—"
    if places is not None:
        return f"{x:,.{places}f}"
    if x >= 100:
        return f"{x:,.1f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def faster(baseline, candidate):
    """How many times faster `candidate` is than `baseline` (>1 means faster)."""
    if baseline is None or candidate is None or candidate == 0:
        return None
    return baseline / candidate


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/results_all.csv"
    rows = load(path)

    by_key = {}
    settings = OrderedDict()
    for row in rows:
        key = (row["flags"], row["batch"])
        settings.setdefault(key, None)
        by_key[(key, row["arch"], row["version"])] = row

    ordered = [k for k in SETTING_ORDER if k in settings]
    for k in settings:
        if k not in ordered:
            ordered.append(k)

    print(f"# Three-way benchmark — `{path}`\n")
    print(
        "Microseconds per sample for forward + backward + update; the **minimum** of "
        "the harness's trials, because interference only ever adds time. "
        "`spread` is max/min for that row — a large value means the machine was "
        "noisy while it ran, so treat small differences in that row as unresolved.\n"
    )

    for flags, batch in ordered:
        print(f"## `{flags}`, batch {batch}\n")
        print(
            "| Architecture | Params | v0.0.1 | v0.0.2 | v0.0.3 "
            "| v0.0.2 vs v0.0.1 | v0.0.3 vs v0.0.2 | v0.0.3 vs v0.0.1 | spread |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for arch in ARCH_ORDER:
            v1 = by_key.get(((flags, batch), arch, "0.0.1"))
            v2 = by_key.get(((flags, batch), arch, "0.0.2"))
            v3 = by_key.get(((flags, batch), arch, "0.0.3"))
            if not (v1 or v2 or v3):
                continue
            a1 = v1["min_us"] if v1 else None
            a2 = v2["min_us"] if v2 else None
            a3 = v3["min_us"] if v3 else None
            params = (v1 or v2 or v3)["params"]
            present = [r for r in (v1, v2, v3) if r]
            spread = max(r["max_us"] / r["min_us"] for r in present)
            print(
                f"| `{arch}` | {params:,} | {fmt(a1)} | {fmt(a2)} | {fmt(a3)} "
                f"| {fmt(faster(a1, a2), 2)}× | {fmt(faster(a2, a3), 2)}× "
                f"| {fmt(faster(a1, a3), 2)}× | {spread:.2f}× |"
            )
        print()

    print("## What the flag and the batch buy, v0.0.3\n")
    print("Each column is how many times faster than the `-O2`, batch-1 baseline.\n")
    print(
        "| Architecture | `-O2` b1 | `-O2+simd` b1 | SIMD gain | `-O2+simd` b32 "
        "| batch-32 gain |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for arch in ARCH_ORDER:
        base = by_key.get((("-O2", 1), arch, "0.0.3"))
        if not base:
            continue
        simd = by_key.get((("-O2 -fopenmp-simd", 1), arch, "0.0.3"))
        batched = by_key.get((("-O2 -fopenmp-simd", 32), arch, "0.0.3"))
        s = simd["min_us"] if simd else None
        b = batched["min_us"] if batched else None
        print(
            f"| `{arch}` | {fmt(base['min_us'])} | {fmt(s)} "
            f"| {fmt(faster(base['min_us'], s), 2)}× | {fmt(b)} "
            f"| {fmt(faster(base['min_us'], b), 2)}× |"
        )

    print("\n## What the flag and the batch buy, v0.0.1\n")
    print("v0.0.1 has no SIMD directives, so the flag column is a control: it "
          "should sit at 1.00×.\n")
    print("| Architecture | `-O2` b1 | `-O2+simd` b1 | flag effect | `-O2` b32 | batch-32 gain |")
    print("|---|---:|---:|---:|---:|---:|")
    for arch in ARCH_ORDER:
        base = by_key.get((("-O2", 1), arch, "0.0.1"))
        if not base:
            continue
        simd = by_key.get((("-O2 -fopenmp-simd", 1), arch, "0.0.1"))
        batched = by_key.get((("-O2", 32), arch, "0.0.1"))
        s = simd["min_us"] if simd else None
        b = batched["min_us"] if batched else None
        print(
            f"| `{arch}` | {fmt(base['min_us'])} | {fmt(s)} "
            f"| {fmt(faster(base['min_us'], s), 2)}× | {fmt(b)} "
            f"| {fmt(faster(base['min_us'], b), 2)}× |"
        )


if __name__ == "__main__":
    main()
