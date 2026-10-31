#!/usr/bin/env bash
# Reaproveita os coletores e o perfil GuideLLM do laboratorio original.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PROJECT="$(dirname "$ROOT")"
K=(minikube -p gpu-lab kubectl --)
MODE="${1:-}"
case "$MODE" in baseline|ngram|ngram-mod|draft) ;; *) echo "Uso: $0 {baseline|ngram|ngram-mod|draft}" >&2; exit 2 ;; esac
# Este arquivo e separado de configs/benchmark.env (Ollama).
# shellcheck disable=SC1091
source "$ROOT/configs/benchmark.env"

ACTUAL_MODE="$("${K[@]}" get deployment/llama-spec -n ml-lab -o json | python3 -c '
import json, sys
args=json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0]["args"]
print(args[args.index("--spec-type")+1])
')"
case "$MODE" in baseline) EXPECTED=none;; ngram) EXPECTED=ngram-simple;; ngram-mod) EXPECTED=ngram-mod;; draft) EXPECTED=draft-simple;; esac
[[ "$ACTUAL_MODE" == "$EXPECTED" ]] || { echo "[ERRO] Implantacao $ACTUAL_MODE; benchmark solicitado $EXPECTED" >&2; exit 1; }
curl --fail --silent --show-error --max-time 10 "$API_URL/health" >/dev/null
curl --fail --silent --show-error --max-time 10 "$API_URL/v1/models" >/dev/null

RUN_DIR="$PROJECT/results/speculative-decoding/$MODE/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR"
cp "$ROOT/configs/benchmark.env" "$RUN_DIR/config.env"
"${K[@]}" get deployment/llama-spec -n ml-lab -o json > "$RUN_DIR/deployment.json"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"

# Warmup fora da janela de medicao.
MODEL="$MODEL" WARMUP_TOKENS="$WARMUP_TOKENS" API_URL="$API_URL" python3 - <<'PY'
import json, os, urllib.request
payload=json.dumps({"model":os.environ["MODEL"], "messages":[{"role":"user","content":"Explique brevemente o que e Kubernetes."}], "max_tokens":int(os.environ["WARMUP_TOKENS"]), "temperature":0, "stream":False}).encode()
req=urllib.request.Request(os.environ["API_URL"].rstrip('/')+'/v1/chat/completions',data=payload,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=300) as response:
    body=json.load(response)
    assert body.get('choices'), 'Warmup sem resposta'
PY
MONITOR_PID=""
stop_monitor() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
trap stop_monitor EXIT
for C in $CONCURRENCIES; do
  echo "[INFO] Benchmark $MODE: concorrencia $C"
  "$PROJECT/scripts/monitoring/gpu_monitor.sh" "$RUN_DIR/gpu-concurrency-$C.csv" 1 &
  MONITOR_PID=$!
  sleep 2
  docker run --rm --network host -v "$RUN_DIR:/results" "$DOCKER_IMAGE" run \
    --backend "kind=openai_http,target=$API_URL,model=$MODEL,request_format=/v1/chat/completions,validate_backend=false" \
    --tokenizer "kind=huggingface_auto,model=$TOKENIZER" \
    --profile "kind=concurrent,streams=$C" \
    --constraint "kind=max_requests,count=$REQUESTS" \
    --data "kind=synthetic_text,prompt_tokens=$INPUT_TOKENS,output_tokens=$OUTPUT_TOKENS" \
    --output "kind=json,path=/results/concurrency-$C.json" \
    --output "kind=csv,path=/results/concurrency-$C.csv"
  stop_monitor
done
python3 "$PROJECT/scripts/summarize.py" "$RUN_DIR"
python3 "$PROJECT/scripts/monitoring/metrics_collector.py" "$RUN_DIR"
"${K[@]}" logs -n ml-lab deployment/llama-spec --tail=500 --timestamps > "$RUN_DIR/server.log" || true
curl --silent --max-time 10 "$API_URL/metrics" > "$RUN_DIR/server-metrics.prom" || true
echo "[OK] Resultados em $RUN_DIR"
