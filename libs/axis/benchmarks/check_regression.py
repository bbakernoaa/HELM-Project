#!/usr/bin/env python3
"""Compare AXIS benchmark results against a stored baseline.

Fails (exit 1) if any benchmark case regresses by more than the regression
threshold relative to its baseline wall-clock time.

Usage:
    check_regression.py <baseline.json> <results.json> [threshold]

Arguments:
    baseline.json   Stored baseline results (list of {name, wall_clock_ms}).
    results.json    Current run results (same schema).
    threshold       Optional fractional regression threshold (default 0.15).

Exit codes:
    0   All benchmarks within threshold (or no baseline present).
    1   One or more benchmarks regressed beyond the threshold.
    2   Usage error (missing/invalid arguments).
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            f"usage: {argv[0]} <baseline.json> <results.json> [threshold]",
            file=sys.stderr,
        )
        return 2

    baseline_path = argv[1]
    results_path = argv[2]
    threshold = float(argv[3]) if len(argv) > 3 else 0.15

    # Load baseline (absent baseline is a soft pass — nothing to compare against).
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print("WARNING: No baseline found — skipping regression check")
        return 0

    with open(results_path) as f:
        results = json.load(f)

    baseline_map = {b["name"]: b["wall_clock_ms"] for b in baseline}

    regressions = []
    print(f"{'Case':<40} {'Baseline (ms)':<15} {'Current (ms)':<15} {'Delta':<10} {'Status'}")
    print("-" * 95)

    for result in results:
        name = result["name"]
        current_ms = result["wall_clock_ms"]

        if name not in baseline_map:
            print(f"{name:<40} {'N/A':<15} {current_ms:<15.2f} {'NEW':<10} SKIP")
            continue

        baseline_ms = baseline_map[name]
        if baseline_ms <= 0:
            print(f"{name:<40} {baseline_ms:<15.2f} {current_ms:<15.2f} {'N/A':<10} SKIP (zero baseline)")
            continue

        delta = (current_ms - baseline_ms) / baseline_ms
        status = "PASS" if delta <= threshold else "FAIL"

        if delta > threshold:
            regressions.append(
                {
                    "name": name,
                    "baseline_ms": baseline_ms,
                    "current_ms": current_ms,
                    "regression_pct": delta * 100,
                }
            )

        print(f"{name:<40} {baseline_ms:<15.2f} {current_ms:<15.2f} {delta * 100:>+7.1f}%   {status}")

    print()

    if regressions:
        print(f"REGRESSION DETECTED: {len(regressions)} benchmark(s) exceeded {threshold * 100:.0f}% threshold")
        for r in regressions:
            print(f"  - {r['name']}: {r['current_ms']:.2f}ms vs baseline {r['baseline_ms']:.2f}ms (+{r['regression_pct']:.1f}%)")
        return 1

    print("All benchmarks within regression threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
