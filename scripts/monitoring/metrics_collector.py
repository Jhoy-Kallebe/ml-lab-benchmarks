#!/usr/bin/env python3
"""Consolida métricas de benchmark geradas por GuideLLM e monitoramento GPU.

O script percorre um diretório contendo arquivos JSON de benchmark e arquivos
CSV com amostras de GPU, extrai as estatísticas relevantes e gera um único CSV
final com uma linha por nível de concorrência. Esse arquivo é útil para comparar
como diferentes configurações impactam latência, throughput e uso de recursos.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path


# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================

PERCENTILES = ("p50", "p90", "p95", "p99")


def load_json(path):
    """Carrega um arquivo JSON de benchmark a partir do caminho informado."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def valid_number(value):
    """Converte valores numéricos válidos para float.

    A função aceita valores string, numéricos e também ignora entradas inválidas,
    vazias ou infinitas. Isso é essencial para CSVs de GPU que podem incluir
    unidades como `%`, `MiB` e `W`.
    """

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def calculate_percentile(values, percentile):
    """Calcula um percentil usando interpolação linear.

    Args:
        values: Lista de números a partir da qual o percentil será calculado.
        percentile: Percentil desejado, em formato de inteiro (50, 90, 95, 99).

    Returns:
        Valor interpolado do percentil ou ``None`` se a lista estiver vazia.
    """

    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * percentile / 100

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def calculate_statistics(values):
    """Calcula média, máximo e percentis para uma amostra de valores."""

    if not values:
        return {
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    result = {
        "mean": sum(values) / len(values),
        "max": max(values),
    }

    for percentile in (50, 90, 95, 99):
        result[f"p{percentile}"] = calculate_percentile(
            values,
            percentile
        )

    return result


# ==========================================
# 2. MÉTRICAS DO GUIDELLM
# ==========================================

def extract_guidellm_metrics(path):
    """Extrai métricas de latência de um JSON do GuideLLM.

    O arquivo exportado pelo benchmark possui uma estrutura em ``benchmarks``;
    cada benchmark contém um bloco ``metrics`` com os resultados de cada
    categoria. A função seleciona apenas a seção ``successful`` para obter as
    estatísticas centrais e os percentis necessários.
    """

    data = load_json(path)

    benchmarks = data.get("benchmarks", [])

    if not benchmarks:
        raise ValueError(f"Sem benchmarks: {path}")

    if len(benchmarks) != 1:
        raise ValueError(
            f"Esperado um benchmark por arquivo: {path}"
        )

    metrics = benchmarks[0]["metrics"]

    result = {}

    metric_mapping = {
        "itl_ms": "inter_token_latency_ms",
        "e2e_s": "request_latency",
    }

    for prefix, metric_name in metric_mapping.items():

        metric = metrics[metric_name]["successful"]

        result[f"{prefix}_mean"] = metric["mean"]

        for percentile in PERCENTILES:
            result[f"{prefix}_{percentile}"] = (
                metric["percentiles"][percentile]
            )

    return result


# ==========================================
# 3. MÉTRICAS DA GPU
# ==========================================

def read_gpu_metrics(path):
    """Lê um CSV de amostras da GPU e retorna estatísticas resumidas.

    O CSV é esperado conter colunas padronizadas como ``utilization.gpu [%]``,
    ``memory.used [MiB]`` e ``power.draw [W]``. A função normaliza os valores
    removendo unidades de medida e calcula apenas média e pico para cada métrica.
    """

    columns = {
        "gpu_utilization_pct": "utilization.gpu [%]",
        "gpu_memory_mib": "memory.used [MiB]",
        "gpu_power_w": "power.draw [W]",
    }

    samples = {
        key: [] for key in columns
    }

    with path.open(encoding="utf-8-sig", newline="") as file:

        reader = csv.DictReader(
            file,
            skipinitialspace=True
        )

        if not reader.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {path}")

        missing = set(columns.values()) - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Colunas ausentes em {path}: {missing}"
            )

        for row in reader:

            for key, column in columns.items():

                raw = row[column]

                if raw is None:
                    continue

                # Compatível com CSVs antigos que contêm unidades.
                cleaned = (
                    raw.strip()
                    .replace("MiB", "")
                    .replace("%", "")
                    .replace("W", "")
                    .strip()
                )

                value = valid_number(cleaned)

                if value is not None:
                    samples[key].append(value)

    result = {}

    for name, values in samples.items():

        stats = calculate_statistics(values)

        result[f"{name}_mean"] = stats["mean"]
        result[f"{name}_max"] = stats["max"]

    return result


# ==========================================
# 4. CONSOLIDAÇÃO
# ==========================================

def collect_metrics(results_dir):
    """Agrupa todos os resultados de um diretório em uma tabela final.

    A função identifica automaticamente arquivos de benchmark do GuideLLM no
    formato ``concurrency-N.json`` e procura o CSV correspondente de uso de GPU,
    no padrão ``gpu-concurrency-N.csv``. Cada concorrência produz uma linha da
    tabela final, facilitando comparação visual ou posterior análise em planilhas.
    """

    results = []

    benchmark_files = sorted(
        results_dir.glob("concurrency-*.json"),
        key=lambda p: int(p.stem.split("-")[-1])
    )

    if not benchmark_files:
        raise ValueError("Nenhum JSON do GuideLLM encontrado.")

    for benchmark_file in benchmark_files:

        match = re.fullmatch(
            r"concurrency-(\d+)",
            benchmark_file.stem
        )

        if not match:
            continue

        concurrency = int(match.group(1))

        gpu_file = (
            results_dir
            / f"gpu-concurrency-{concurrency}.csv"
        )

        print(f"[INFO] Processando concorrência {concurrency}")

        result = {
            "concurrency": concurrency
        }

        # Métricas de inferência
        result.update(
            extract_guidellm_metrics(benchmark_file)
        )

        # Métricas da GPU
        if gpu_file.exists():

            result.update(
                read_gpu_metrics(gpu_file)
            )

        else:

            print(
                f"[AVISO] Arquivo de GPU não encontrado: "
                f"{gpu_file.name}"
            )

            for name in (
                "gpu_utilization_pct",
                "gpu_memory_mib",
                "gpu_power_w"
            ):
                result[f"{name}_mean"] = None
                result[f"{name}_max"] = None

        # KV cache requer telemetria do runtime.
        result["kv_cache_usage"] = "N/A"

        results.append(result)

    return results


# ==========================================
# 5. EXPORTAÇÃO
# ==========================================

def save_csv(results, output_file):
    """Salva a tabela consolidada em um CSV final."""

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    """Ponto de entrada da consolidação de métricas."""

    parser = argparse.ArgumentParser(
        description="Consolida métricas de inferência e GPU."
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Diretório dos resultados do benchmark."
    )

    args = parser.parse_args()

    if not args.results_dir.is_dir():
        parser.error("Diretório não encontrado.")

    results = collect_metrics(args.results_dir)

    output_file = (
        args.results_dir / "selected-metrics.csv"
    )

    save_csv(results, output_file)

    print()
    print("[OK] Coleta consolidada concluída.")
    print(f"[OK] Arquivo: {output_file}")


if __name__ == "__main__":
    main()