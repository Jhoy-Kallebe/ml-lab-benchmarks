#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 1. CARREGAMENTO DAS CONFIGURAÇÕES
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="$PROJECT_DIR/configs/benchmark.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERRO] Arquivo de configuração não encontrado."
    exit 1
fi

source "$CONFIG_FILE"

echo "=========================================="
echo "       MLOps - LLM Benchmark"
echo "=========================================="

echo "Modelo: $MODEL"
echo "Endpoint: $API_URL"
echo "Requisições: $REQUESTS"
echo "Concorrências: $CONCURRENCIES"

# ==========================================
# 2. VERIFICAÇÃO DO SERVIDOR
# ==========================================

echo
echo "[INFO] Verificando disponibilidade da API..."

if ! curl --fail --silent --show-error \
    --connect-timeout 5 \
    --max-time 10 \
    "$API_URL/v1/models" > /dev/null; then

    echo "[ERRO] Não foi possível acessar a API do Ollama."
    echo
    echo "Verifique se o port-forward está ativo:"
    echo
    echo "minikube kubectl -p gpu-lab -- port-forward -n ml-lab service/ollama 11435:11434"

    exit 1
fi

echo "[OK] API do Ollama acessível."

# ==========================================
# 3. WARMUP DO MODELO
# ==========================================

echo
echo "[INFO] Iniciando warmup do modelo..."

WARMUP_RESPONSE=$(curl --fail --silent --show-error \
    --connect-timeout 5 \
    --max-time 300 \
    "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(MODEL="$MODEL" WARMUP_TOKENS="$WARMUP_TOKENS" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [
        {
            "role": "user",
            "content": "Explique brevemente o que é Kubernetes."
        }
    ],
    "max_tokens": int(os.environ["WARMUP_TOKENS"]),
    "stream": False
}))
PY
)")

# Verifica se a resposta contém uma geração válida.
if ! printf '%s' "$WARMUP_RESPONSE" | python3 -c '
import json
import sys

try:
    response = json.load(sys.stdin)
    choices = response.get("choices", [])

    if not choices:
        raise ValueError("Resposta sem choices")

    content = choices[0].get("message", {}).get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Resposta sem conteúdo gerado")

except (ValueError, KeyError, TypeError) as error:
    print(f"[ERRO] Warmup inválido: {error}")
    sys.exit(1)
'; then
    echo "[ERRO] Falha na validação do warmup."
    exit 1
fi

echo "[OK] Warmup concluído com sucesso."

# ==========================================
# 4. PREPARAÇÃO DOS RESULTADOS
# ==========================================

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_DIR="$PROJECT_DIR/results/runs/$TIMESTAMP"

mkdir -p "$RUN_DIR"

# Preserva a configuração original do experimento.
cp "$CONFIG_FILE" "$RUN_DIR/config.env"

echo
echo "[INFO] Diretório de resultados: $RUN_DIR"

# ==========================================
# 5. EXECUÇÃO DOS BENCHMARKS
# ==========================================

echo
echo "[INFO] Iniciando benchmarks..."

# ==========================================
# CONTROLE DO MONITORAMENTO
# ==========================================

GPU_MONITOR_PID=""

GPU_MONITOR_PID=""

stop_gpu_monitor() {

    if [[ -n "$GPU_MONITOR_PID" ]]; then

        echo "[INFO] Encerrando GPU Monitor (PID: $GPU_MONITOR_PID)..."

        kill "$GPU_MONITOR_PID" 2>/dev/null || true

        wait "$GPU_MONITOR_PID" 2>/dev/null || true

        GPU_MONITOR_PID=""

        echo "[OK] GPU Monitor encerrado."

    fi
}

trap stop_gpu_monitor EXIT

for CONCURRENCY in $CONCURRENCIES; do

    echo
    echo "=========================================="
    echo " Benchmark: Concorrência $CONCURRENCY"
    echo "=========================================="

    # Iniciar monitoramento
    GPU_OUTPUT="$RUN_DIR/gpu-concurrency-$CONCURRENCY.csv"

    "$SCRIPT_DIR/monitoring/gpu_monitor.sh" \
        "$GPU_OUTPUT" 1 &

    GPU_MONITOR_PID=$!

    sleep 2

    if ! kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        echo "[ERRO] GPU Monitor não iniciou."
        exit 1
    fi

    # Executar benchmark
    docker run --rm \
        --network host \
        -v "$RUN_DIR:/results" \
        "$DOCKER_IMAGE" run \
        --backend "kind=openai_http,target=$API_URL,model=$MODEL,request_format=/v1/chat/completions,validate_backend=false" \
        --tokenizer "kind=huggingface_auto,model=$TOKENIZER" \
        --profile "kind=concurrent,streams=$CONCURRENCY" \
        --constraint "kind=max_requests,count=$REQUESTS" \
        --data "kind=synthetic_text,prompt_tokens=$INPUT_TOKENS,output_tokens=$OUTPUT_TOKENS" \
        --output "kind=json,path=/results/concurrency-$CONCURRENCY.json" \
        --output "kind=csv,path=/results/concurrency-$CONCURRENCY.csv"

    # Encerrar monitoramento
    stop_gpu_monitor

    echo "[OK] Benchmark C=$CONCURRENCY concluído."

done

echo
echo "=========================================="
echo " Todos os benchmarks foram concluídos!"
echo "=========================================="

echo "Resultados disponíveis em:"
echo "$RUN_DIR"

echo
echo "[INFO] Gerando relatório comparativo..."

python3 "$SCRIPT_DIR/summarize.py" "$RUN_DIR"

echo "[OK] Relatório comparativo gerado."

# ==========================================
# 3. EXECUÇÃO DO BENCHMARK
# ==========================================
# 1 - Tornar o script executável
# chmod +x scripts/benchmark.sh
# 2 - Executar
# ./scripts/benchmark.sh