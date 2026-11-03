#!/usr/bin/env python3
"""Controlled greedy-output equivalence tests for ml-lab-benchmarks llama.cpp server.

Copy beside probe.py in speculative-decoding/scripts/ and run from repository root.
Does not change the server deployment or overwrite prior measurements.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from probe import PROMPTS
except ImportError:
    sys.exit("Copy this script beside speculative-decoding/scripts/probe.py before running.")

ROOT = Path('results/speculative-decoding/equivalence')
MODEL = 'qwen2.5-3b-instruct'
MODES = ('baseline', 'ngram', 'draft-k1', 'draft-k2', 'draft-k4')


def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def request(url, payload):
    req = urllib.request.Request(
        url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    return result, elapsed


def run(args):
    now = datetime.now(timezone.utc)
    payload_base = {
        'model': args.model,
        'max_tokens': args.max_tokens,
        'temperature': 0,
        'top_k': 1,
        'top_p': 1.0,
        'seed': 42,
        'cache_prompt': args.cache == 'on',
        'stream': False,
    }
    dataset = {
        'mode': args.mode,
        'created_utc': now.isoformat(),
        'settings': payload_base,
        'prompts': PROMPTS,
        'runs': {},
    }
    for name, prompt in PROMPTS.items():
        dataset['runs'][name] = []
        for iteration in range(1, args.repeats + 1):
            req = {**payload_base, 'messages': [{'role': 'user', 'content': prompt}]}
            try:
                response, elapsed = request(args.url, req)
            except (urllib.error.URLError, ValueError) as e:
                sys.exit(f'Failed: {name} repeat={iteration}: {e}')
            choice = response.get('choices', [{}])[0]
            content = choice.get('message', {}).get('content') or ''
            times = response.get('timings') or {}
            usage = response.get('usage') or {}
            item = {
                'iteration': iteration,
                'response': content,
                'sha256': digest(content),
                'elapsed_s': round(elapsed, 4),
                'completion_tokens': usage.get('completion_tokens'),
                'finish_reason': choice.get('finish_reason'),
                'cache_n': times.get('cache_n'),
                'prompt_n': times.get('prompt_n'),
                'prompt_ms': times.get('prompt_ms'),
                'predicted_ms': times.get('predicted_ms'),
            }
            dataset['runs'][name].append(item)
            print(f'{args.mode:8} {name:10} rep={iteration} sha={item["sha256"][:12]} '
                  f'tok={item["completion_tokens"]} cache_n={item["cache_n"]} '
                  f'E2E={item["elapsed_s"]:.3f}s')
            if args.cache == 'off' and isinstance(item['cache_n'], int) and item['cache_n'] > 0:
                print('WARNING: server reported cached prompt tokens despite cache_prompt=false')
    folder = Path(args.output_dir) / args.mode
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{args.mode}-{now.strftime("%Y%m%d_%H%M%S_%f")}.json'
    path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSaved: {path}')


def newest(path, mode):
    candidates = sorted((path / mode).glob(f'{mode}-*.json'))
    return candidates[-1] if candidates else None


def mismatch_offset(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if a != b else None


def compare(args):
    folder = Path(args.output_dir)
    base_path = newest(folder, 'baseline')
    if base_path is None:
        sys.exit('No controlled baseline results. Run baseline first.')
    base = json.loads(base_path.read_text(encoding='utf-8'))
    print(f'BASELINE: {base_path}')
    for mode in ('ngram', 'draft-k1', 'draft-k2', 'draft-k4'):
        candidate_path = newest(folder, mode)
        if not candidate_path:
            print(f'{mode}: no controlled results yet')
            continue
        cand = json.loads(candidate_path.read_text(encoding='utf-8'))
        if base['settings'] != cand['settings']:
            print(f'WARNING {mode}: sampling settings differ, do not compare equivalence as controlled')
        print(f'\n{mode.upper()}: {candidate_path}')
        for workload in PROMPTS:
            b_samples = base['runs'][workload]
            c_samples = cand['runs'][workload]
            b0, c0 = b_samples[0]['response'], c_samples[0]['response']
            b_unique = len({r['sha256'] for r in b_samples})
            c_unique = len({r['sha256'] for r in c_samples})
            same = b0 == c0
            offset = mismatch_offset(b0, c0)
            print(f'  {workload:10}: exact_match={same} first_diff_char={offset} '
                  f'baseline_unique={b_unique}/{len(b_samples)} '
                  f'{mode}_unique={c_unique}/{len(c_samples)}')
            if not same:
                lo = max(0, offset - 25)
                hi = offset + 65
                print('    baseline: ', repr(b0[lo:hi]))
                print('    candidate:', repr(c0[lo:hi]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='Run controlled probes against active deployment')
    r.add_argument('--mode', choices=MODES, required=True)
    r.add_argument('--url', default='http://127.0.0.1:11436')
    r.add_argument('--model', default=MODEL)
    r.add_argument('--max-tokens', type=int, default=192)
    r.add_argument('--repeats', type=int, default=2)
    r.add_argument('--cache', choices=['on', 'off'], default='off')
    r.add_argument('--output-dir', default=str(ROOT))
    c = sub.add_parser('compare', help='Compare latest controlled runs to baseline')
    c.add_argument('--output-dir', default=str(ROOT))
    args = parser.parse_args()
    if args.action == 'run':
        if args.repeats < 1:
            parser.error('--repeats must be >= 1')
        run(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()
