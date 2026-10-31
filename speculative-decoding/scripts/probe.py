#!/usr/bin/env python3
"""Probes deterministas de QA, repeticao e codigo para observar aceitacao."""
import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPEAT = "O Kubernetes distribui pods, reinicia processos e controla replicas. "
CODE = "def soma(a, b):\n    return a + b\n"
PROMPTS = {
    "qa": "Explique como o scheduler do Kubernetes decide onde executar um Pod. Responda detalhadamente.",
    "repeticao": "Repita literalmente o texto a seguir, preservando a ordem, sem adicionar comentarios:\n" + REPEAT * 9,
    "codigo": "Mostre 10 testes pytest separados para a funcao abaixo, com codigo completo:\n" + CODE,
}

def run(url, model, prompt, max_tokens):
    data = json.dumps({
        "model": model,
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions', data=data, headers={"Content-Type":"application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    elapsed = time.perf_counter() - start
    tokens = payload.get('usage',{}).get('completion_tokens')
    return {
        "elapsed_s": round(elapsed,3),
        "output_tokens": tokens,
        "output_tokens_per_e2e_s": round(tokens/elapsed,3) if isinstance(tokens,int) and elapsed>0 else None,
        "response": payload.get('choices',[{}])[0].get('message',{}).get('content',''),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:11436')
    p.add_argument('--model', default='qwen2.5-3b-instruct')
    p.add_argument('--mode', required=True, choices=['baseline','ngram','ngram-mod','draft'])
    p.add_argument('--output-dir', default='results/speculative-decoding/probes')
    p.add_argument('--max-tokens', type=int, default=192)
    args = p.parse_args()
    result = {"mode":args.mode,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"prompts":{}}
    for label,prompt in PROMPTS.items():
        result["prompts"][label]=run(args.url,args.model,prompt,args.max_tokens)
        x=result["prompts"][label]
        print(f"{label}: {x['output_tokens']} tokens / {x['elapsed_s']} s = {x['output_tokens_per_e2e_s']} tokens/s E2E")
    output = Path(args.output_dir) / f"{args.mode}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"Probes salvos em {output}")

if __name__=='__main__':
    main()
