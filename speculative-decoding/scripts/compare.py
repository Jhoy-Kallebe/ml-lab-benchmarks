#!/usr/bin/env python3
"""Compare ALL speculative decoding benchmark runs, preserving K and provenance.

No third-party dependencies. Reads the original experiment layout:
results/speculative-decoding/<mode>/<timestamp>/{summary.csv,selected-metrics.csv,deployment.json}
"""
import argparse
import csv
import json
from pathlib import Path

MODES = ("baseline", "ngram", "ngram-mod", "draft")


def rows_csv(path):
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def tokens_drafted(deployment_path):
    """Read actual K from the manifest saved by benchmark.sh; do not guess."""
    if not deployment_path.is_file():
        return None
    with deployment_path.open(encoding="utf-8") as fp:
        deployment = json.load(fp)
    containers = deployment.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    target = next((c for c in containers if c.get("name") == "llama-spec"), None)
    if not target:
        return None
    arguments = target.get("args", [])
    if "--spec-draft-n-max" not in arguments:
        return 0 if "--spec-type" in arguments and arguments[arguments.index("--spec-type") + 1] == "none" else None
    pos = arguments.index("--spec-draft-n-max") + 1
    try:
        return int(arguments[pos])
    except (ValueError, IndexError):
        return None


def load(root):
    records = []
    for mode in MODES:
        mode_dir = root / mode
        for run in sorted(p for p in mode_dir.glob("*") if p.is_dir() and (p / "summary.csv").exists()):
            k = tokens_drafted(run / "deployment.json")
            supplement = (run / "selected-metrics.csv")
            gpu_by_concurrency = {int(r["concurrency"]): r for r in rows_csv(supplement)} if supplement.is_file() else {}
            for s in rows_csv(run / "summary.csv"):
                conc = int(s["concurrency"])
                m = gpu_by_concurrency.get(conc, {})
                records.append({
                    "mode": mode,
                    "k": k,
                    "run": run.name,
                    "concurrency": conc,
                    "baseline_run": None,
                    "throughput_tokens_s": to_float(s.get("throughput_tokens_s")),
                    "speedup_vs_baseline": None,
                    "e2e_mean_s": to_float(s.get("e2e_mean_s")),
                    "e2e_p95_s": to_float(s.get("e2e_p95_s")),
                    "e2e_p95_change_pct": None,
                    "ttft_mean_ms": to_float(s.get("ttft_mean_ms")),
                    "itl_mean_ms": to_float(s.get("itl_mean_ms")),
                    "itl_p95_ms": to_float(m.get("itl_ms_p95")),
                    "gpu_memory_mib_max": to_float(m.get("gpu_memory_mib_max")),
                    "gpu_utilization_pct_mean": to_float(m.get("gpu_utilization_pct_mean")),
                    "run_dir": str(run),
                })
    return records


def select_reference(records, tested):
    candidates = [r for r in records if r["mode"] == "baseline" and r["concurrency"] == tested["concurrency"]]
    if not candidates:
        return None
    before = [r for r in candidates if r["run"] <= tested["run"]]
    return max(before, key=lambda r: r["run"]) if before else min(candidates, key=lambda r: r["run"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/speculative-decoding"))
    parser.add_argument("--latest-only", action="store_true", help="Show only latest run per (mode, concurrency, K), still keeping distinct K values")
    arguments = parser.parse_args()
    rows = load(arguments.root)
    if not rows:
        parser.error("Nenhum benchmark encontrado em --root")
    for record in rows:
        baseline = select_reference(rows, record)
        if not baseline:
            continue
        record["baseline_run"] = baseline["run"]
        speed, reference = record["throughput_tokens_s"], baseline["throughput_tokens_s"]
        if speed is not None and reference and reference > 0:
            record["speedup_vs_baseline"] = round(speed / reference, 4)
        e2e, e2e_baseline = record["e2e_p95_s"], baseline["e2e_p95_s"]
        if e2e is not None and e2e_baseline and e2e_baseline > 0:
            record["e2e_p95_change_pct"] = round((e2e / e2e_baseline - 1) * 100, 2)

    rows.sort(key=lambda r: (r["concurrency"], MODES.index(r["mode"]), r["k"] if r["k"] is not None else -1, r["run"]))
    if arguments.latest_only:
        latest = {(r["mode"], r["concurrency"], r["k"]): r for r in rows}
        rows = list(latest.values())

    output = arguments.root / "comparison.csv"
    arguments.root.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparativo exportado: {output}")
    print("modo       K  conc.  run                  tokens/s  speedup  E2E P95  Δ E2E P95")
    for r in rows:
        def fmt(x, form=".2f"):
            return format(x, form) if x is not None else "N/A"
        print(f"{r['mode']:<10} {str(r['k']):>2}  {r['concurrency']:>4}   {r['run']:<20} {fmt(r['throughput_tokens_s']):>8}  {fmt(r['speedup_vs_baseline'],'.3f'):>7}  {fmt(r['e2e_p95_s']):>7}  {fmt(r['e2e_p95_change_pct']):>9}%")

if __name__ == "__main__":
    main()
