#!/usr/bin/env python3

import json
import csv
import sys

from pathlib import Path


def load_benchmark(path):
    """Carrega um arquivo JSON do GuideLLM."""

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def get_mean(metrics, metric_name):
    """Extrai a média de uma métrica."""

    return metrics[metric_name]["successful"]["mean"]


def get_percentile(metrics, metric_name, percentile="p95"):
    """Extrai um percentil de uma métrica."""

    return metrics[metric_name]["successful"]["percentiles"][percentile]


def extract_metrics(path):
    """Extrai os indicadores de um experimento."""

    data = load_benchmark(path)

    benchmark = data["benchmarks"][0]

    metrics = benchmark["metrics"]

    concurrency = int(path.stem.split("-")[-1])

    requests = metrics["request_totals"]

    return {
        "concurrency": concurrency,

        "requests_successful": requests["successful"],
        "requests_errored": requests["errored"],
        "requests_incomplete": requests["incomplete"],

        "ttft_mean_ms": get_mean(
            metrics, "time_to_first_token_ms"
        ),

        "ttft_p95_ms": get_percentile(
            metrics, "time_to_first_token_ms"
        ),

        "itl_mean_ms": get_mean(
            metrics, "inter_token_latency_ms"
        ),

        "e2e_mean_s": get_mean(
            metrics, "request_latency"
        ),

        "e2e_p95_s": get_percentile(
            metrics, "request_latency"
        ),

        "tpot_mean_ms": get_mean(
            metrics, "time_per_output_token_ms"
        ),

        "throughput_tokens_s": get_mean(
            metrics, "output_tokens_per_second"
        ),

        "throughput_requests_s": get_mean(
            metrics, "requests_per_second"
        ),

        "output_tokens_mean": get_mean(
            metrics, "output_token_count"
        )
    }


def main():

    if len(sys.argv) != 2:
        print("Uso: python3 summarize.py <diretorio>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.is_dir():
        print("[ERRO] Diretório não encontrado.")
        sys.exit(1)

    files = sorted(
        results_dir.glob("concurrency-*.json"),
        key=lambda p: int(p.stem.split("-")[-1])
    )

    if not files:
        print("[ERRO] Nenhum benchmark encontrado.")
        sys.exit(1)

    print("[INFO] Processando resultados...")

    results = []

    for file in files:

        result = extract_metrics(file)

        results.append(result)

        print(
            f"[OK] C={result['concurrency']} "
            f"| TTFT={result['ttft_mean_ms']:.2f} ms "
            f"| E2E={result['e2e_mean_s']:.2f} s "
            f"| Throughput={result['throughput_tokens_s']:.2f} tok/s"
        )

    output_file = results_dir / "summary.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print(f"[OK] Relatório gerado: {output_file}")


if __name__ == "__main__":
    main()