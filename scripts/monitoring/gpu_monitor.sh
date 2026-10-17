#!/usr/bin/env bash

set -euo pipefail

OUTPUT_FILE="${1:-gpu_metrics.csv}"
INTERVAL="${2:-1}"

echo "[INFO] Iniciando monitoramento da GPU..."
echo "[INFO] Arquivo: $OUTPUT_FILE"
echo "[INFO] Intervalo: ${INTERVAL}s"

exec nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,nounits \
    --loop-ms="$((INTERVAL * 1000))" \
    > "$OUTPUT_FILE"

# Conceda permissão de execução:
# chmod +x scripts/monitoring/gpu_monitor.sh
# Execute:
# ./scripts/monitoring/gpu_monitor.sh \
#     results/gpu-test.csv 1