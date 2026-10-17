# MLOps · Laboratório de benchmark de inferência LLM

> Documentação de reprodutibilidade baseada nos scripts, nas configurações e nos resultados registrados neste laboratório. **Este README descreve o estado atual, não um benchmark padronizado entre runtimes.**

## 1. Objetivo e métodos

Avaliar latência de inferência e telemetria de GPU para um modelo local, inicialmente `qwen2.5:3b` servido pelo Ollama em Minikube. Há **dois fluxos independentes**:

| Fluxo | Execução | Processamento | Saída |
|---|---|---|---|
| GuideLLM | `scripts/benchmark.sh` | `scripts/summarize.py` e `scripts/monitoring/metrics_collector.py` | `results/runs/<data>/summary.csv` e `selected-metrics.csv` |
| Python independente | `scripts/monitoring/standalone_collector.py` | O próprio script | `results/standalone/<teste>/metrics.json`, `metrics.csv`, `gpu_samples.csv` |

**Não execute `metrics_collector.py` sobre a pasta `results/standalone/`:** ele exige arquivos `concurrency-N.json` no formato GuideLLM. O standalone já exporta suas estatísticas.

## 2. Ambiente registrado

- Host: Ubuntu Linux bare metal; Intel Core i5 de 3ª geração; 16 GiB DDR3; NVIDIA GeForce RTX 3050 com 6 GiB de VRAM.
- Driver NVIDIA observado no laboratório: `595.99.02`; `nvidia-smi` disponível no host.
- Orquestração: Minikube, perfil `gpu-lab`, namespace `ml-lab`, serviço Kubernetes `ollama`.
- Modelo: `qwen2.5:3b`; tokenizador GuideLLM: `Qwen/Qwen2.5-3B-Instruct`.
- API: compatível com OpenAI, acessível pelo port-forward em `http://127.0.0.1:11435`.
- GuideLLM: versão 0.8.1 utilizada durante o desenvolvimento, com imagem local corrigida para a compatibilidade com Ollama. **O Dockerfile fornecido não fixa essa versão**, portanto a reconstrução futura pode produzir uma versão diferente.

Registrar em cada reprodução: `uname -a`, `python3 --version`, `docker version`, `minikube version`, `nvidia-smi`, versão/digest da imagem, digest/quantização do modelo, configurações do Ollama, parâmetros da carga, data/hora, ambiente gráfico e processos concorrentes. As versões e a configuração completa do cluster **ainda não estão integralmente fixadas no repositório**.

## 3. Organização do repositório

```text
ml-lab-benchmarks/
├── Dockerfile                  # Imagem base GuideLLM
├── Dockerfile.ollama           # Patch de compatibilidade da API Ollama
├── configs/benchmark.env      # Parâmetros do GuideLLM
├── scripts/
│   ├── benchmark.sh           # Orquestra GuideLLM, warmup e telemetria
│   ├── summarize.py           # Resumo próprio do GuideLLM
│   └── monitoring/
│       ├── gpu_monitor.sh     # Amostragem nvidia-smi
│       ├── metrics_collector.py      # Consolida GuideLLM + GPU
│       └── standalone_collector.py   # Benchmark HTTP independente
└── results/
    ├── runs/<timestamp>/      # GuideLLM
    └── standalone/<teste>/    # Standalone
```

O diretório `Laboratorio_Kubernetes_GPU/`, quando presente no projeto, reúne manifesto e relatório do ambiente; consulte esses arquivos para detalhes do deployment. Evite distribuir metadados `.git` de repositórios internos desnecessariamente.

## 4. Dependências e preparação

**Host:** Bash, Python 3, Docker, Minikube/kubectl, `curl`, `nvidia-smi` operacional, driver NVIDIA compatível e acesso à API de inferência. Para o standalone, a dependência Python de terceiros é **`requests`**. `metrics_collector.py` e `summarize.py` utilizam biblioteca padrão do Python. `pandas` é opcional para análise posterior.

```bash
cd ~/Documentos/ml-lab-benchmarks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install requests
chmod +x scripts/benchmark.sh scripts/monitoring/gpu_monitor.sh
nvidia-smi
```

**Reprodutibilidade:** o projeto enviado não inclui `requirements.txt` com versões fixadas. Para congelar o ambiente efetivamente testado: `python -m pip freeze > requirements-lock.txt`. Não assuma que qualquer versão futura de `requests` ou GuideLLM terá comportamento idêntico.

### 4.1. Disponibilizar a API Ollama

Com Minikube e o serviço `ollama` já configurados, execute em **terminal separado**:

```bash
minikube kubectl -p gpu-lab -- \
  port-forward -n ml-lab service/ollama 11435:11434
```

Verifique:

```bash
curl -f http://127.0.0.1:11435/v1/models
```

O port-forward deve permanecer ativo durante todos os testes. Se o serviço, namespace ou perfil for diferente, ajuste o comando e a URL.

## 5. Método A — GuideLLM

### 5.1. Construção da imagem

O `Dockerfile` usa `python:3.13-slim`, instala `guidellm[recommended]` e define `ENTRYPOINT ["guidellm"]`. O `Dockerfile.ollama` usa a imagem base `guidellm-lab:latest` e modifica `request_handlers.py` para trocar `max_completion_tokens` por `max_tokens` em **dois pontos**. O patch contém `assert` e é **dependente da versão/estrutura interna do pacote**.

```bash
docker build -t guidellm-lab:latest -f Dockerfile .
docker build -t guidellm-ollama:latest -f Dockerfile.ollama .
```

> A correção é específica da compatibilidade observada com o Ollama e não deve ser assumida necessária para outros runtimes. Para reconstrução rigorosa, fixe a versão exata de GuideLLM e a imagem base e valide novamente o patch.

### 5.2. Configuração usada no experimento

Arquivo `configs/benchmark.env` (e cópia em `results/runs/2026-10-16_235338/config.env`):

```bash
MODEL="qwen2.5:3b"
TOKENIZER="Qwen/Qwen2.5-3B-Instruct"
API_URL="http://127.0.0.1:11435"
DOCKER_IMAGE="guidellm-ollama:latest"
INPUT_TOKENS=128
OUTPUT_TOKENS=128
REQUESTS=30
CONCURRENCIES="1 2 4 8"
WARMUP_TOKENS=32
```

O `benchmark.sh` carrega o `.env` com `source`, verifica `/v1/models`, faz uma requisição de **warmup** sem streaming, cria uma pasta datada, salva uma cópia do `.env` e, para cada concorrência, inicia `gpu_monitor.sh`, executa o GuideLLM e encerra o monitoramento. Ao final executa `summarize.py`. O monitoramento é iniciado **2 segundos antes** do processo Docker.

Parâmetros efetivos do GuideLLM:

```text
--backend kind=openai_http,target=http://127.0.0.1:11435,model=qwen2.5:3b,request_format=/v1/chat/completions,validate_backend=false
--tokenizer kind=huggingface_auto,model=Qwen/Qwen2.5-3B-Instruct
--profile kind=concurrent,streams=<1|2|4|8>
--constraint kind=max_requests,count=30
--data kind=synthetic_text,prompt_tokens=128,output_tokens=128
```

`--network host` permite ao contêiner acessar a porta encaminhada no host Linux. O tokenizador Hugging Face pode exigir acesso ou cache prévio dos arquivos do modelo.

### 5.3. Executar

```bash
./scripts/benchmark.sh
ls -lt results/runs/
```

Cada pasta gerada contém, conforme as concorrências executadas:

```text
config.env
concurrency-1.json / concurrency-1.csv
concurrency-2.json / concurrency-2.csv
concurrency-4.json / concurrency-4.csv
concurrency-8.json / concurrency-8.csv
gpu-concurrency-1.csv ... gpu-concurrency-8.csv
summary.csv
```

### 5.4. `gpu_monitor.sh`

Executa `exec nvidia-smi` com amostragem de **1 segundo** no `benchmark.sh` e colunas `timestamp`, `name`, `utilization.gpu`, `memory.used`, `memory.total`, `temperature.gpu`, `power.draw`. Usa `--format=csv,nounits`: o CSV mantém **cabeçalhos com unidades**, mas os valores são numéricos. O `exec` evita deixar um processo filho `nvidia-smi` vivo quando o PID monitorado é encerrado.

### 5.5. `metrics_collector.py`: consolidar GuideLLM e GPU

```bash
python3 scripts/monitoring/metrics_collector.py \
  results/runs/2026-10-16_235338
```

O programa procura `concurrency-*.json`, extrai de `benchmarks[0].metrics` as seções `inter_token_latency_ms.successful` e `request_latency.successful`, lê `gpu-concurrency-N.csv` e grava `selected-metrics.csv`. Métricas:

- `itl_ms_mean`, `itl_ms_p50`, `itl_ms_p90`, `itl_ms_p95`, `itl_ms_p99`;
- `e2e_s_mean`, `e2e_s_p50`, `e2e_s_p90`, `e2e_s_p95`, `e2e_s_p99`;
- `gpu_utilization_pct_mean/max`, `gpu_memory_mib_mean/max`, `gpu_power_w_mean/max`;
- `kv_cache_usage = N/A` (não coletado).

**Unidades diferentes:** ITL em **milissegundos**, E2E em **segundos**. Se o arquivo de GPU estiver ausente, o script preenche suas colunas com valores vazios; sem JSON do GuideLLM, encerra com `ValueError: Nenhum JSON do GuideLLM encontrado.`

O `benchmark.sh` **não chama automaticamente `metrics_collector.py`** na versão documentada. Para consolidar após a execução, execute o comando acima; opcionalmente acrescente ao final do `benchmark.sh`:

```bash
python3 "$SCRIPT_DIR/monitoring/metrics_collector.py" "$RUN_DIR"
```

## 6. Método B — `standalone_collector.py` (sem GuideLLM)

O script usa `requests` e `ThreadPoolExecutor` para enviar requisições POST a `/v1/chat/completions` com `stream=true`, `model`, `messages` e `max_tokens`. Marca os tempos com `time.perf_counter()`; calcula **E2E** do início do envio ao encerramento do streaming e **intervalos entre fragmentos de texto recebidos**. Em uma thread paralela, executa `nvidia-smi` periodicamente, coleta utilização, memória e potência da GPU `--id=0`, e finaliza a coleta ao terminar as requisições.

Exemplo reproduzível:

```bash
python3 scripts/monitoring/standalone_collector.py \
  --api-url http://127.0.0.1:11435 \
  --model qwen2.5:3b \
  --requests 30 \
  --concurrency 1 \
  --max-tokens 128 \
  --interval 1 \
  --output results/standalone/concurrency-1
```

Para testar concorrência 2, 4 e 8, repita alterando `--concurrency` e `--output` (use um diretório diferente por execução). O prompt padrão pede uma explicação dos componentes de Kubernetes; use `--prompt "..."` para personalizá-lo. `--timeout` configura o timeout de leitura HTTP, padrão **300 s**.

Arquivos:

- `metrics.json`: estatísticas de ITL, E2E, utilização, memória, potência e `kv_cache_usage: null`;
- `metrics.csv`: uma linha de estatísticas consolidadas;
- `gpu_samples.csv`: amostras individuais da GPU com timestamps UTC (quando disponíveis).

**Definição de ITL neste script:** intervalo entre **chunks de conteúdo** no streaming, não necessariamente entre tokens individuais. O campo `itl_ms.count` representa o número de intervalos entre fragmentos, e não o número de requisições. O cálculo de E2E é feito apenas para requisições marcadas como concluídas; erros são contados separadamente. Os percentis são calculados por interpolação linear sobre as amostras disponíveis.

**Não há warmup explícito** no standalone documentado. Uma primeira requisição lenta pode distorcer médias e percentis. O script também não exporta, nesta versão, um CSV individual de latências por requisição nem timestamps de cada chunk.

## 7. Resultados efetivamente observados

### 7.1. GuideLLM — execução `2026-10-16_235338`

Os valores abaixo foram extraídos da tabela `selected-metrics.csv` apresentada na conversa após executar o `metrics_collector.py`:

| Concorrência | ITL média (ms) | ITL P95 (ms) | E2E média (s) | E2E P95 (s) | GPU média (%) | VRAM pico (MiB) | Potência média (W) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16,835706 | 18,189535 | 2,065522 | 2,428783 | 66,280488 | 2962 | 57,798902 |
| 2 | 16,914227 | 17,385062 | 3,850535 | 4,451235 | 63,259259 | 2959 | 55,732469 |
| 4 | 17,039034 | 18,540270 | 7,482270 | 8,974937 | 63,209877 | 2939 | 55,814444 |
| 8 | 16,831898 | 17,332730 | 13,870149 | 16,680753 | 62,469880 | 2943 | 54,618193 |

**Leitura:** de concorrência 1 para 8, a latência E2E média cresceu cerca de **6,7 vezes**, enquanto a média de ITL variou pouco e o uso médio da GPU não aumentou. O comportamento é **compatível com enfileiramento ou limites de paralelismo**, mas **não prova** execução estritamente serial nem identifica sozinho o gargalo. Faltam, nesta tabela, TTFT, throughput agregado, dados individuais de fila e controle rigoroso das janelas de amostragem.

### 7.2. Standalone — duas observações distintas

**Saída compartilhada anteriormente no terminal:** 30/30 requisições concluídas; concorrência 1; ITL média **16,8259 ms**, P95 **25,4350 ms**; E2E média **2408,9614 ms**, P95 **2314,9784 ms**, P99 **7056,0628 ms**; GPU média **78,7571%**; VRAM média **2793,29 MiB**; potência média **62,644 W**.

**Resultado salvo em `results/standalone/concurrency-1/metrics.json`:** 30/30 requisições concluídas; ITL média **19,7616 ms**, P95 **25,5353 ms**; E2E média **5163,5857 ms**, P95 **2219,9845 ms**, P99 **66114,2643 ms**, máximo **92211,1902 ms**; GPU média **36,5364%**; VRAM média **2822,23 MiB**; potência média **42,4354 W**. Esse arquivo contém **151 amostras** de GPU, contra **70** no resultado anteriormente compartilhado.

**Importante:** os dois conjuntos standalone **não são idênticos**. A execução registrada no arquivo de resultados contém uma latência extrema (~92 s), que afeta fortemente a média e o P99 e pode reduzir as médias de utilização e potência ao incluir espera prolongada. Não atribua a diferença a uma causa específica sem logs e timestamps individuais. Para auditoria e reprodução, priorize os resultados persistidos em arquivos com identificação da execução; a saída do terminal constitui uma observação histórica adicional.

## 8. Interpretação e limitações metodológicas

| Indicador | O que representa | Cuidado |
|---|---|---|
| ITL | GuideLLM: métrica interna `inter_token_latency_ms`; standalone: tempo entre chunks de texto | Os métodos **não são automaticamente equivalentes** a token por token |
| E2E | Duração completa observada pelo cliente | Inclui transporte, espera e processamento; unidade GuideLLM = s, standalone = ms |
| P50/P90/P95/P99 | Percentis de amostras | P99 de 30 requisições é instável e muito sensível a extremos |
| GPU Utilization | Ocupação reportada pelo driver | Não é uso exclusivo do modelo; GPU também atende outros processos |
| Memory Usage | Memória GPU alocada | Inclui pesos, cache, runtime e outras aplicações; não isola KV cache |
| Power Consumption | Potência instantânea reportada pela GPU | Não é consumo da tomada nem energia total; não equivale a joules |
| KV Cache Usage | **Não disponível neste experimento** | Exige métricas específicas do runtime; `null`/`N/A` não é zero |

A GPU é monitorada **por amostragem periódica**, não continuamente; picos breves podem passar despercebidos. No fluxo GuideLLM, o monitoramento começa antes do cliente e termina depois, o que pode incluir períodos ociosos. No standalone, o monitor inicia antes do envio das requisições, mas a primeira amostra e o início efetivo da carga não são sincronizados formalmente. A média de potência é média de amostras; não é uma medição direta de energia consumida.

**Comparação GuideLLM × standalone:** as cargas **não são equivalentes**: GuideLLM utiliza entrada sintética de 128 tokens e saída-alvo de 128 tokens; standalone usa um prompt textual fixo e `max_tokens=128` (limite máximo, não garantia de 128 tokens gerados). Os mecanismos de concorrência, warmup e contagem de intervalos de streaming também diferem. **Não atribuir diferenças apenas ao coletor ou ao runtime.**

## 9. Protocolo recomendado para novas execuções

1. Registrar versões, digest do modelo, quantização, perfil de GPU e variáveis de paralelismo do Ollama (inclusive `OLLAMA_NUM_PARALLEL`, quando configurada).
2. Confirmar que `/v1/models` responde e que a GPU está acessível via `nvidia-smi`.
3. Definir cargas equivalentes (mesmo prompt/tokenização, comprimento real de saída, temperatura e limites) antes de comparar métodos.
4. Realizar warmup explícito e registrar se as amostras de warmup são descartadas.
5. Executar pelo menos três repetições independentes por concorrência; não misturar execuções.
6. Preservar **JSON/CSV brutos**, cópia do `config.env`, logs, início/fim da janela útil e anotações de falhas.
7. Reportar contagem de requisições bem-sucedidas, intervalos de confiança quando apropriado e percentis com a quantidade de amostras.
8. Para diagnosticar filas, medir **TTFT** e **throughput agregado real** (tokens de saída efetivamente gerados / tempo do experimento), além de observar paralelismo do runtime.
9. Para medir KV cache, integrar telemetria específica de Ollama/vLLM/SGLang/TensorRT-LLM conforme disponível; não estimar KV cache a partir da VRAM total.

## 10. Diagnóstico rápido

- **`Diretório não encontrado`**: confira o caminho relativo a partir da raiz do projeto (`pwd`, `ls results/runs/`).
- **`Nenhum JSON do GuideLLM encontrado`**: informe a pasta de uma execução GuideLLM, não a pasta do standalone.
- **`Colunas ausentes` no CSV GPU**: `metrics_collector.py` exige cabeçalho; `gpu_monitor.sh` deve usar `--format=csv,nounits`, **sem** `noheader`.
- **API indisponível**: verifique o port-forward e `curl http://127.0.0.1:11435/v1/models`.
- **Falha na construção da imagem corrigida**: o patch depende do texto exato do arquivo interno do GuideLLM; valide a versão da imagem base.
- **Médias inesperadas / P99 muito alto**: procure outliers, cold start, espera, interferência externa e diferença de janela de monitoramento; não elimine amostras sem documentar o critério.

## 11. Referências operacionais

- Código-fonte do projeto: `Dockerfile`, `Dockerfile.ollama`, `configs/benchmark.env`, `scripts/benchmark.sh`, `scripts/summarize.py`, `scripts/monitoring/{gpu_monitor.sh,metrics_collector.py,standalone_collector.py}`.
- Dados preservados: `results/runs/2026-10-16_235338/` e `results/standalone/concurrency-1/` presentes no projeto.
- Documentação das ferramentas: [GuideLLM](https://github.com/vllm-project/guidellm), [Ollama API](https://docs.ollama.com/api), [NVIDIA System Management Interface](https://developer.nvidia.com/system-management-interface), [Minikube](https://minikube.sigs.k8s.io/docs/).

---

**Estado do laboratório:** fluxos GuideLLM e standalone executados; exportação de resultados validada; métricas de KV cache não coletadas; investigação de TTFT, throughput agregado e sincronização de telemetria pendente.
