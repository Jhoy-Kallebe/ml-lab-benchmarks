import re
from pathlib import Path

path = Path(
    "results/speculative-decoding/draft-k2-repeticoes.log"
)

pattern = re.compile(
    r"draft acceptance = [\d.]+ "
    r"\(\s*(\d+) accepted /\s*(\d+) generated\)"
)

matches = [
    (int(a), int(g))
    for a, g in pattern.findall(path.read_text())
]

print(f"Registros encontrados: {len(matches)}")

if len(matches) != 18:
    raise SystemExit(
        "Quantidade inesperada. Verifique os registros antes de calcular."
    )

# Ignorar os três registros do aquecimento
matches = matches[3:]

workloads = ["qa", "repeticao", "codigo"]

for index, workload in enumerate(workloads):
    samples = matches[index::3]

    accepted = sum(a for a, _ in samples)
    generated = sum(g for _, g in samples)

    rate = 100 * accepted / generated if generated else 0

    print(
        f"{workload}: {accepted}/{generated} "
        f"= {rate:.2f}%"
    )