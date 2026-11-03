#!/usr/bin/env bash
# Reconfigura um único servidor llama.cpp; NÃO interrompe Ollama automaticamente.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
K=(minikube -p gpu-lab kubectl --)
MODE="${1:-}"
K_TOKENS="${2:-4}"
case "$MODE" in baseline|ngram|ngram-mod|draft) ;; *) echo "Uso: $0 {baseline|ngram|ngram-mod|draft} [2|4|6|8]" >&2; exit 2;; esac
[[ "$K_TOKENS" =~ ^(1|2|4|6|8)$ ]] || { echo "K invalido (1,2,4,6,8)" >&2; exit 2; }
REPLICAS="$("${K[@]}" get deployment ollama -n ml-lab -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
if [[ "$REPLICAS" != "0" ]]; then
  echo "[ERRO] Ollama ainda solicita a unica GPU. Antes de prosseguir:" >&2
  echo "  minikube -p gpu-lab kubectl -- scale deployment/ollama -n ml-lab --replicas=0" >&2
  exit 1
fi

ARGS=(
  --model /models/qwen2.5-3b-instruct-q4_k_m.gguf
  --alias qwen2.5-3b-instruct
  --ctx-size 2048
  --n-gpu-layers 99
  --parallel 1
  --threads 3
  --threads-batch 3
  --batch-size 128
  --ubatch-size 128
  --host 0.0.0.0
  --port 8080
  --metrics
)
# Configuracao opcional do Flash Attention.
# Sem LLAMA_FA, preserva o comportamento original.
if [[ -n "${LLAMA_FA:-}" ]]; then
  case "$LLAMA_FA" in
    on|off|auto) ;;
    *)
      echo "[ERRO] LLAMA_FA deve ser on, off ou auto" >&2
      exit 2
      ;;
  esac

  ARGS+=(--flash-attn "$LLAMA_FA")
fi

case "$MODE" in
  baseline) ARGS+=(--spec-type none);;
  ngram) ARGS+=(--spec-type ngram-simple --spec-draft-n-max "$K_TOKENS");;
  ngram-mod) ARGS+=(--spec-type ngram-mod --spec-draft-n-max "$K_TOKENS");;
  draft) ARGS+=(--spec-type draft-simple \
    --model-draft /models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --spec-draft-ngl all --spec-draft-n-max "$K_TOKENS" --spec-draft-n-ubatch 32);;
esac

# A implantação base tem replicas=0; patch antes de iniciar, evitando boot duplicado.
"${K[@]}" apply -f "$ROOT/k8s/pvc.yaml"
"${K[@]}" apply -f "$ROOT/k8s/service.yaml"
"${K[@]}" apply -f "$ROOT/k8s/deployment.yaml"
PATCH="$(python3 - "${ARGS[@]}" <<'PY'
import json, sys
print(json.dumps({"spec": {"template": {"spec": {"containers": [
    {"name": "llama-spec", "args": sys.argv[1:]}
]}}}}))
PY
)"
"${K[@]}" patch deployment llama-spec -n ml-lab --type strategic -p "$PATCH"
"${K[@]}" scale deployment/llama-spec -n ml-lab --replicas=1
"${K[@]}" rollout status deployment/llama-spec -n ml-lab --timeout=600s
printf '[OK] Modo ativo: %s (K=%s)\n' "$MODE" "$K_TOKENS"
echo 'Execute em outro terminal: minikube -p gpu-lab kubectl -- port-forward -n ml-lab service/llama-spec 11436:8080'
