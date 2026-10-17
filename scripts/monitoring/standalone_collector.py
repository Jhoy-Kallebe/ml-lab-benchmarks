#!/usr/bin/env python3
"""Coletor de métricas de benchmark para inferência em LLMs.

Este script dispara requisições em paralelo para um endpoint compatível com
OpenAI, mede a latência de cada chunk do streaming e acompanha o uso da GPU
via ``nvidia-smi`` durante a execução. Os resultados são exportados em JSON e
CSV em um diretório informado pela CLI.

O objetivo principal é gerar dados de desempenho sem depender de um
orquestrador específico, permitindo comparar diferentes modelos ou backends
em um ambiente controlado.
"""

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


# ==========================================
# ESTATÍSTICAS
# ==========================================

def percentile(values, p):
    """Calcula o percentil p de uma lista de valores numéricos.

    A implementação usa interpolação linear entre os valores vizinhos quando o
    percentil cai entre duas posições, preservando a semântica de uma mediana e
    percentis mais altos para relatórios de latência e desempenho.
    """
    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * p / 100

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def statistics(values):
    """Agrupa medidas em um resumo estatístico de uso para benchmarking.

    Args:
        values: Lista de valores numéricos, como latências em milissegundos ou
            uso de GPU.

    Returns:
        Dicionário com contagem, média, máximo e percentis p50, p90, p95 e p99.
        Quando a lista está vazia, todos os valores são retornados como ``None``
        para evitar erros de cálculo em execuções sem dados.
    """
    if not values:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


# ==========================================
# MONITORAMENTO GPU
# ==========================================

def monitor_gpu(stop_event, samples, interval):
    """Coleta estatísticas periódicas da GPU em uma thread separada.

    A função invoca o comando ``nvidia-smi`` a cada intervalo e armazena as
    métricas em ``samples`` para posterior análise. Se a ferramenta não estiver
    disponível ou falhar, o erro é registrado sem interromper a execução do
    benchmark principal.
    """

    fields = [
        "utilization.gpu",
        "memory.used",
        "power.draw",
    ]

    while not stop_event.is_set():

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )

            values = result.stdout.strip().split(",")

            if len(values) == 3:
                samples.append({
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "gpu_utilization_pct": float(values[0]),
                    "gpu_memory_mib": float(values[1]),
                    "gpu_power_w": float(values[2]),
                })

        except (
            subprocess.SubprocessError,
            ValueError,
            OSError,
        ) as error:
            print(f"[AVISO] Falha na coleta GPU: {error}")

        stop_event.wait(interval)


# ==========================================
# INFERÊNCIA
# ==========================================

def run_request(api_url, model, prompt, max_tokens, timeout):
    """Envia uma requisição a um provedor compatível com OpenAI e mede latência.

    O endpoint responde em streaming, emitindo eventos em formato SSE. Esse
    fluxo é processado linha a linha para detectar chunks de resposta e calcular o
    intervalo entre um fragmento e outro (ITL, inter-token latency).

    Args:
        api_url: URL base do provedor, por exemplo ``http://localhost:11435``.
        model: Nome do modelo a ser usado na inferência.
        prompt: Texto enviado ao modelo como mensagem do usuário.
        max_tokens: Limite máximo de tokens gerados.
        timeout: Tempo máximo de espera em segundos por requisição.

    Returns:
        Dicionário contendo sucesso da chamada, tempo total, latência inter-token,
        número de fragments e mensagem de erro, quando aplicável.
    """

    start = time.perf_counter()

    previous_event = None
    itl_samples = []

    fragments = 0
    completed = False

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
    }

    endpoint = api_url.rstrip("/") + "/v1/chat/completions"

    try:

        with requests.post(
            endpoint,
            json=payload,
            stream=True,
            timeout=(10, timeout),
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                line = line.decode("utf-8")

                if not line.startswith("data:"):
                    continue

                content = line[5:].strip()

                if content == "[DONE]":
                    completed = True
                    break

                try:
                    event = json.loads(content)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices", [])

                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                text = delta.get("content")

                if not text:
                    continue

                now = time.perf_counter()

                fragments += 1

                if previous_event is not None:
                    itl_samples.append(
                        (now - previous_event) * 1000
                    )

                previous_event = now

        end = time.perf_counter()

        return {
            "success": completed,
            "e2e_ms": (end - start) * 1000,
            "itl_ms": itl_samples,
            "fragments": fragments,
            "error": None if completed else "Streaming incompleto",
        }

    except requests.RequestException as error:

        return {
            "success": False,
            "e2e_ms": None,
            "itl_ms": [],
            "fragments": fragments,
            "error": str(error),
        }


# ==========================================
# BENCHMARK
# ==========================================

def run_benchmark(args):
    """Executa o benchmark completo e persiste os resultados em disco.

    O fluxo de execução envolve quatro etapas principais: (1) iniciar a coleta
    paralela de métricas de GPU, (2) disparar múltiplas inferências em paralelo, (3)
    compactar todos os resultados em estatísticas agregadas e (4) exportar
    arquivos JSON/CSV para análise posterior.
    """

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_samples = []
    stop_event = threading.Event()

    print("[INFO] Iniciando monitoramento GPU...")

    monitor = threading.Thread(
        target=monitor_gpu,
        args=(
            stop_event,
            gpu_samples,
            args.interval,
        ),
        daemon=True,
    )

    monitor.start()

    results = []

    print("[INFO] Executando inferências...")

    try:
        with ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:

            futures = [
                executor.submit(
                    run_request,
                    args.api_url,
                    args.model,
                    args.prompt,
                    args.max_tokens,
                    args.timeout,
                )
                for _ in range(args.requests)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    finally:
        stop_event.set()
        monitor.join(timeout=10)

    successful = [
        result for result in results
        if result["success"]
    ]

    e2e = [
        result["e2e_ms"]
        for result in successful
    ]

    itl = [
        sample
        for result in successful
        for sample in result["itl_ms"]
    ]

    gpu_utilization = [
        sample["gpu_utilization_pct"]
        for sample in gpu_samples
    ]

    gpu_memory = [
        sample["gpu_memory_mib"]
        for sample in gpu_samples
    ]

    gpu_power = [
        sample["gpu_power_w"]
        for sample in gpu_samples
    ]

    summary = {
        "model": args.model,
        "concurrency": args.concurrency,
        "requests_total": len(results),
        "requests_successful": len(successful),
        "requests_failed": len(results) - len(successful),

        "itl_ms": statistics(itl),
        "e2e_ms": statistics(e2e),

        "gpu_utilization_pct": statistics(
            gpu_utilization
        ),

        "gpu_memory_mib": statistics(
            gpu_memory
        ),

        "gpu_power_w": statistics(
            gpu_power
        ),

        "kv_cache_usage": None,
    }

    # Exportação JSON
    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Exportação CSV resumido
    row = {
        "model": args.model,
        "concurrency": args.concurrency,
        "requests_successful": len(successful),
        "requests_failed": len(results) - len(successful),
    }

    for metric in (
        "itl_ms",
        "e2e_ms",
        "gpu_utilization_pct",
        "gpu_memory_mib",
        "gpu_power_w",
    ):
        for key, value in summary[metric].items():
            row[f"{metric}_{key}"] = value

    row["kv_cache_usage"] = None

    with (output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        writer.writeheader()
        writer.writerow(row)

    # Exportação de amostras GPU
    if gpu_samples:
        with (output_dir / "gpu_samples.csv").open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=gpu_samples[0].keys()
            )

            writer.writeheader()
            writer.writerows(gpu_samples)

    print()
    print("[OK] Benchmark concluído.")
    print(f"[OK] Resultados: {output_dir}")

    print()
    print(json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ))


# ==========================================
# CLI
# ==========================================

def main():
    """Ponto de entrada da CLI do benchmark independente.

    A interface aceita parâmetros para conectar a um provedor local/remote,
    controlar concorrência e exportar os artefatos gerados. Os valores padrão
    foram escolhidos para facilitar a execução rápida em ambientes locais de
    desenvolvimento e benchmarking inicial.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark independente do GuideLLM"
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:11435"
    )

    parser.add_argument(
        "--model",
        default="qwen2.5:3b"
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=30
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300
    )

    parser.add_argument(
        "--prompt",
        default=(
            "Explique os principais componentes "
            "de uma arquitetura Kubernetes."
        )
    )

    parser.add_argument(
        "--output",
        default="results/standalone"
    )

    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests e --concurrency devem ser positivos.")

    if args.interval <= 0 or args.timeout <= 0:
        parser.error("--interval e --timeout devem ser positivos.")

    run_benchmark(args)


if __name__ == "__main__":
    main()