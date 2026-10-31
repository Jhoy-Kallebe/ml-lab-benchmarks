#!/usr/bin/env bash
# Baixa GGUF verificando SHA-256 e envia a um PVC separado, sem tocar no PVC do Ollama.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
CACHE="${MODEL_CACHE:-$HOME/.cache/ml-lab-benchmarks/speculative-decoding}"
K=(minikube -p gpu-lab kubectl --)
ACTION="${1:-}"
WHAT="${2:-target}"

case "$WHAT" in
  target) FILE="qwen2.5-3b-instruct-q4_k_m.gguf"; REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"; HASH="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d" ;;
  draft) FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"; REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"; HASH="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db" ;;
  *) echo "Uso: $0 {download|stage|remove-loader} {target|draft}" >&2; exit 2 ;;
esac
mkdir -p "$CACHE"
LOCAL="$CACHE/$FILE"

verify_local() { echo "$HASH  $LOCAL" | sha256sum -c -; }

case "$ACTION" in
  download)
    if [[ -f "$LOCAL" ]] && verify_local; then
      echo "[OK] Modelo ja verificado: $LOCAL"
      exit 0
    fi
    echo "[INFO] Download/reinicio de $FILE (~2,1 GB target; ~491 MB draft)"
    curl --fail --location --retry 4 --retry-delay 3 --continue-at - \
      --output "$LOCAL" "https://huggingface.co/$REPO/resolve/main/$FILE"
    verify_local
    echo "[OK] Download integro: $LOCAL"
    ;;
  stage)
    [[ -f "$LOCAL" ]] || { echo "[ERRO] Execute: $0 download $WHAT" >&2; exit 1; }
    verify_local
    "${K[@]}" apply -f "$ROOT/k8s/pvc.yaml"
    "${K[@]}" apply -f "$ROOT/k8s/model-loader.yaml"
    "${K[@]}" wait --for=condition=Ready pod/llama-spec-model-loader -n ml-lab --timeout=180s
    if "${K[@]}" exec -n ml-lab llama-spec-model-loader -- sh -c "echo '$HASH  /models/$FILE' | sha256sum -c -"; then
      echo "[OK] O modelo ja esta no PVC."
    else
      "${K[@]}" cp "$LOCAL" "ml-lab/llama-spec-model-loader:/models/$FILE"
      "${K[@]}" exec -n ml-lab llama-spec-model-loader -- sh -c "echo '$HASH  /models/$FILE' | sha256sum -c -"
    fi
    echo "[OK] Modelo copiado e validado no PVC: $FILE"
    ;;
  remove-loader)
    "${K[@]}" delete pod llama-spec-model-loader -n ml-lab --ignore-not-found --wait=true
    ;;
  *) echo "Uso: $0 {download|stage|remove-loader} {target|draft}" >&2; exit 2 ;;
esac
