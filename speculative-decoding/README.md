# Speculative Decoding — laboratório em um dia (RTX 3050 / Minikube)

**Status:** implementação preparada; execução e resultados ainda não validados em GPU. **Não substitui o laboratório Ollama existente.**

## 1. Contexto observado no repositório (31/10/2026)

- Host Ubuntu bare-metal, Intel i5 com quatro CPUs, 16 GiB DDR3, RTX 3050 6 GiB (6144 MiB).
- Com GNOME e navegador ativos: ~440 MiB de VRAM ocupados, ~5360 MiB disponíveis; RAM disponível ~7,4 GiB; disco livre ~318 GB.
- Minikube `gpu-lab`, namespace `ml-lab`, Kubernetes v1.37, GPU anunciada como `nvidia.com/gpu: 1`.
- `deployment/ollama` 1/1 Ready; PVC `ollama-models` 10 GiB Bound. **Não apagar esse PVC.**
- Erros antigos `UnexpectedAdmissionError` e `FailedScheduling` ocorreram na retomada do cluster e o pod novo foi agendado/ficou saudável; só investigar se reincidirem.
- O script raiz `scripts/benchmark.sh` usa `configs/benchmark.env` fixo com Ollama (`11435`, `guidellm-ollama:latest`). Não o alteramos.
- Reutilizamos `scripts/summarize.py`, `scripts/monitoring/metrics_collector.py`, `scripts/monitoring/gpu_monitor.sh`.
- Benchmark anterior (`results/runs/2026-10-16_235338`), concorrência 1: ITL média 16,84 ms, E2E média 2,07 s, VRAM pico 2962 MiB. **Não comparar esse número diretamente com resultados llama.cpp:** runtime/modelo/quantização podem mudar.

## 2. Técnicas e hipótese

- **Baseline:** Qwen2.5-3B-Instruct Q4_K_M no llama.cpp, sem especulação.
- **N-gramas:** `ngram-simple` e opcionalmente `ngram-mod`; propõem tokens a partir de padrões anteriores, sem segunda rede neural. Workloads sintéticos sem repetição podem não gerar propostas úteis.
- **Draft model:** Qwen2.5-0.5B-Instruct Q4_K_M propõe até K tokens; o 3B verifica um bloco e aceita/rejeita as propostas. Começar com K=4, depois K=2/6 se houver tempo.
- **Mecanismo:** a execução do target por token é parcialmente substituída por verificação conjunta. Em modo de amostragem compatível, o algoritmo pode preservar a distribuição da saída; neste piloto usamos `temperature=0` nos probes de validação.
- **Meta de pesquisa, não promessa:** descobrir se há ganho de geração sob concorrência 1. Um speedup de 0,8–1,3x é uma faixa hipotética para planejar; não é uma medição da RTX 3050.

## 3. Cronograma de um dia (8 horas)

| Janela | Etapa | Critério de passagem |
|---|---|---|
| 00h00–00h30 | Confirmar saúde do cluster, espaço e GPU; aplicar pacote em branch | Git limpo; Ollama Ready; GPU anunciada |
| 00h30–02h00 | Baixar target + draft; verificar SHA-256; criar PVC e copiar GGUF | Modelos verificados no PVC dedicado |
| 02h00–03h00 | Escalar Ollama para zero; implantar llama.cpp baseline; `port-forward` | `/health` e `/v1/models` respondendo |
| 03h00–04h00 | Warmup + GuideLLM baseline e probes | CSV e JSON no diretório novo |
| 04h00–05h00 | Ativar `ngram-simple` e opcional `ngram-mod`; medir | Registros de modo e comparativo |
| 05h00–06h30 | Ativar Draft Model K=4; medir; tratar eventual OOM | Experimento draft executado ou limitação documentada |
| 06h30–07h30 | Coletar logs de aceitação, comparar ITL/E2E/throughput/VRAM | `comparison.csv` e relatórios do GuideLLM |
| 07h30–08h00 | Restaurar Ollama; documentar resultados e hipóteses | Ollama 1/1 Ready; PVC original intacto |

> Se transferência de 2,6 GB + cópia ao PVC ou download da imagem demorar, reduzir o número de repetições e manter a prioridade **baseline → ngram → draft**. Não sacrificar rollback.

## 4. Pré-requisitos

No clone raiz (`~/Documentos/ml-lab-benchmarks`):

```bash
minikube status -p gpu-lab
minikube -p gpu-lab kubectl -- get deployment,pvc -n ml-lab
nvidia-smi
curl --version
python3 --version
# Imagem GuideLLM normal usada pelo novo experimento; evita patch Ollama.
docker build -t guidellm-lab:latest -f Dockerfile .
```

Executar scripts a partir da raiz do repositório ou por caminho absoluto.

## 5. Preparação e transferência de modelos

Os arquivos serão armazenados fora do Git em `~/.cache/ml-lab-benchmarks/speculative-decoding` e copiados a um **PVC novo `llama-spec-models` (8 GiB)** por um Pod `llama-spec-model-loader` sem GPU. Nenhum modelo será adicionado ao Git.

```bash
./speculative-decoding/scripts/models.sh download target
./speculative-decoding/scripts/models.sh download draft
./speculative-decoding/scripts/models.sh stage target
./speculative-decoding/scripts/models.sh stage draft
./speculative-decoding/scripts/models.sh remove-loader target
```

- O download usa Hugging Face Qwen oficial e **valida hashes SHA-256 publicados**. Se um arquivo completo falhar na validação, remova o arquivo local corrompido e repita o download.
- O Pod de transferência precisa de acesso ao PVC; `kubectl cp` utiliza `tar` (BusyBox inclui essa ferramenta).
- A transferência pode consumir tempo de CPU, memória e I/O, mas não aloca `nvidia.com/gpu`.
- Se o Minikube estiver sem acesso ao registry de imagens, verificar rede antes de prosseguir.

## 6. Trocar de runtime, implantar e testar

**A única GPU já está alocada ao Deployment Ollama.** O script `deploy.sh` interrompe com erro enquanto as réplicas do Ollama forem > 0; ele **não** modifica o Ollama automaticamente.

```bash
# Libera a GPU, mantendo deployment, service e PVC do Ollama.
minikube -p gpu-lab kubectl -- scale deployment/ollama -n ml-lab --replicas=0

# Carrega baseline; a primeira imagem CUDA pode demorar a baixar.
./speculative-decoding/scripts/deploy.sh baseline

# Em OUTRO terminal (deixar ativo durante os testes):
minikube -p gpu-lab kubectl -- port-forward -n ml-lab service/llama-spec 11436:8080

# No terminal principal:
curl -fsS http://127.0.0.1:11436/health
curl -fsS http://127.0.0.1:11436/v1/models
```

**Etapa baseline:**

```bash
./speculative-decoding/scripts/benchmark.sh baseline
python3 speculative-decoding/scripts/probe.py --mode baseline
```

**N-gramas:**

```bash
./speculative-decoding/scripts/deploy.sh ngram 4
./speculative-decoding/scripts/benchmark.sh ngram
python3 speculative-decoding/scripts/probe.py --mode ngram
# Opcional, se houver tempo:
./speculative-decoding/scripts/deploy.sh ngram-mod 4
./speculative-decoding/scripts/benchmark.sh ngram-mod
```

**Draft Model:**

```bash
./speculative-decoding/scripts/deploy.sh draft 4
./speculative-decoding/scripts/benchmark.sh draft
python3 speculative-decoding/scripts/probe.py --mode draft
```

`deploy.sh` reaplica o template **com replicas=0**, modifica os argumentos do container com patch estratégico e só então inicia uma réplica, evitando duplicar a reserva de GPU. `Recreate` evita sobreposição durante troca de modos. Em caso de startup falho:

```bash
minikube -p gpu-lab kubectl -- get pods -n ml-lab -o wide
minikube -p gpu-lab kubectl -- describe pod -n ml-lab -l app=llama-spec
minikube -p gpu-lab kubectl -- logs -n ml-lab deployment/llama-spec --tail=150
nvidia-smi
```

Se houver **CUDA OOM**, testar primeiro K=2; se persistir, diminuir `--ctx-size` para 1024 na lista `ARGS` de `deploy.sh` e no `deployment.yaml`, preservando o mesmo contexto do baseline para comparação. A memória solicitada pelo pod (7Gi host) não representa VRAM; esta é gerenciada pelo processo CUDA.

## 7. Benchmark e interpretação

A configuração está em `speculative-decoding/configs/benchmark.env`, **separada** do arquivo original. O teste rápido usa 15 requisições sintéticas, 128 tokens de entrada e 128 de saída e concorrência 1. Se houver tempo, elevar `REQUESTS=30` e `CONCURRENCIES="1 2"`, repetindo todos os modos com as **mesmas configurações**.

Cada execução salva em `results/speculative-decoding/<modo>/<timestamp>`:

- `deployment.json`, `nvidia-smi.txt`, `config.env` e `server.log`;
- `concurrency-1.json/csv`, `gpu-concurrency-1.csv`, `summary.csv`, `selected-metrics.csv`;
- `server-metrics.prom`, quando o endpoint estiver disponível.

Executar comparação (seleciona a última execução completa de cada modo):

```bash
python3 speculative-decoding/scripts/compare.py
# Output: results/speculative-decoding/comparison.csv
# Estatísticas de aceitação do servidor (se presentes):
minikube -p gpu-lab kubectl -- logs -n ml-lab deployment/llama-spec | grep -E 'acceptance|statistics (draft|ngram)' || true
```

**Cuidado na interpretação:** o `throughput_tokens_s` gerado pelo `scripts/summarize.py` reproduz a métrica de `output_tokens_per_second` do GuideLLM, não necessariamente throughput agregado do servidor sob concorrência >1. O `standalone_collector.py` mede intervalos entre *chunks* SSE, não necessariamente entre tokens individuais. Não anunciar ITL como tokens/ms sem verificar tamanho dos chunks. Priorizar E2E P95, TTFT, throughput com definição explícita e taxa de aceitação do runtime. Os probes medem `output_tokens/E2E`, não throughput contínuo de decodificação.

**Qualidade:** comparar prompts determinísticos `temperature=0`, medir divergências e investigar diferenças numéricas. N-gramas terão maiores oportunidades em tarefas com cópia/repetição; textos sintéticos uniformes podem ter poucas correspondências. A taxa de aceitação não é, por si só, o speedup.

## 8. Limites do hardware e mitigação

| Limitação | Efeito | O que testar |
|---|---|---|
| **RTX 3050 6 GiB** (na medição 5360 MiB livres) | Pesos de ambos os modelos + 2 KV caches + buffers CUDA; possível OOM | Q4_K_M, contexto 2048 ou 1024, batch 128, ubatch 128, draft ubatch 32, K=2–4, concorrência 1 |
| **4 CPUs i5 antigo** | Draft em CPU/offload pode custar mais do que a economia da verificação | Preferir draft inteiro na GPU; medir CPU se houver offload |
| **~7,4 GiB RAM disponível** | Pressão de cache e de containers, transferência do modelo | Evitar download/conversão durante benchmark, fechar aplicações não essenciais |
| **GPU compartilhada com desktop** | Variabilidade por GNOME/navegador | Registrar `nvidia-smi` e manter condições constantes |
| **1 GPU Kubernetes** | Deployment Ollama e llama.cpp com 1 GPU não podem coexistir | Escalar Ollama para 0 e recuperar depois |
| **Inferência Qwen 3B já leve** | Ganho pode ser menor que em LLM 14B/70B | Relatar speedup <1 como resultado válido, investigar taxa de aceitação |
| **NVIDIA SMI 1s** | Picos curtos não detectados | Usar logs do runtime e maior frequência numa máquina dedicada |

Estimativa: pesos GGUF target ~2,1 GB e draft ~491 MB. O restante não é VRAM automaticamente livre: KV, overhead, buffers de computação e display também ocupam memória.

## 9. Escalabilidade para outras máquinas

**GPU 24–48 GiB:** medir target 7B/14B com draft 0,5B/1,5B, contexts 4k–16k, K=2/4/8 e concorrência 1/2/4/8. Introduzir captura de TTFT, TPOT, tokens/s agregados, percentis P50–P99, peak KV, GPU power e acceptance rate. Controlar batch e comparações intra-runtime.

**DGX Spark (128 GB unificados):** modelos 14B/32B (conforme quantização) e contextos mais longos podem caber, mas não concluir desempenho só pela capacidade. Medir tráfego/largura de banda de memória compartilhada, contenção CPU–GPU, prefill/decode e concorrência. Speculative Decoding **não aumenta largura de banda física**, pode melhorar trabalho útil por ciclo de leitura. A eficiência do draft também depende do interconnect/compartilhamento de cache.

**vLLM / SGLang / TensorRT-LLM:** avaliar suportes concretos de versão para EAGLE, MTP, n-gramas e draft-model e realizar benchmark intra-runtime; não comparar runtimes diferentes como evidência isolada de ganho por especulação.

## 10. Rollback (obrigatório no fim do dia)

```bash
# Desativa runtime novo e libera GPU.
minikube -p gpu-lab kubectl -- scale deployment/llama-spec -n ml-lab --replicas=0
# Recupera o Ollama, preservado.
minikube -p gpu-lab kubectl -- scale deployment/ollama -n ml-lab --replicas=1
minikube -p gpu-lab kubectl -- rollout status deployment/ollama -n ml-lab --timeout=300s
minikube -p gpu-lab kubectl -- get deployments,pvc -n ml-lab
```

Não executar `kubectl delete pvc ollama-models`. O PVC `llama-spec-models` pode ser preservado para próximas rodadas. Não usar `kubectl delete --all`.

## 11. Rastreabilidade e fontes técnicas

- `llama.cpp` — [Speculative Decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- `llama.cpp` — [Docker/CUDA](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- `llama.cpp` — [Server OpenAI API e métricas](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Qwen2.5-3B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
- [Qwen2.5-0.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
- `results/runs/2026-10-16_235338` do repositório (histórico Ollama)

**Reprodutibilidade:** a tag `server-cuda` ainda é mutável; depois do primeiro pull funcional, registrar o digest da imagem, versões de runtime e SHA-256 dos modelos e fixar a imagem pelo digest. Mesmo com manifest verificado, ainda será necessário validar flags e alocação CUDA na RTX 3050.
