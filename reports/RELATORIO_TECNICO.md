# Relatório técnico — aquisição, EDA e preparação para M4 / 24 GB

## 1. Resultado e rastreabilidade

**Recomendação:** preparar BERTimbau Base para a classificação binária em português e um decoder Llama 3.1 8B Instruct quantizado em 4 bits, com adaptadores LoRA no MLX, para geração condicionada em evidências. Tratar o 8B como candidato a validar com contexto curto; manter um 3B como alternativa com maior margem de memória.

Nesta entrega foram executadas aquisição e EDA de todos os arquivos abaixo. Não foram implementados treinamento, RAG, extração multimodal nem busca ao vivo. A execução ocorreu em Windows/Python 3.13.1; as recomendações para Apple Silicon são planejamento de engenharia, não medições no M4.

| Dataset | Fonte adquirida | Revisão / identificação |
|---|---|---|
| Fake.br | Checkout local do repositório original | `780f5516c4ae070761632d98ac3368f3ded09d35` |
| AVeriTeC | `data/train.json` e `data/dev.json` do GitHub | `7c62d1ec8df3fb560d6efe2b85fa191135636f81` |
| FEVEROUS | JSONL de treino, desenvolvimento e teste cego da página oficial | URLs, SHA-256, bytes e cabeçalhos HTTP em `data/raw/manifest.json` |

Os cinco downloads JSON/JSONL totalizam **208.918.368 bytes (~199,2 MiB)**, além do Fake.br já presente. O relatório [EDA.md](EDA.md) e o arquivo [eda.json](eda.json) contêm a data efetiva, estatísticas e hash do manifesto usado. `reports/samples/` guarda amostras originais, e `data/processed/fakebr/` contém a ingestão padronizada das duas variantes de texto.

A decisão de usar biblioteca padrão elimina dependências de ML nesta etapa. O AVeriTeC é lido um split de cada vez; o FEVEROUS, linha a linha. As contagens e tipos são calculados sobre **todos** os registros, não extrapolados de amostras.

## 2. Estruturas observadas

### Fake.br — português brasileiro

- `full_texts/fake/*.txt` e `full_texts/true/*.txt`: uma notícia completa por arquivo UTF-8.
- `full_texts/fake-meta-information/` e `true-meta-information/`: arquivos de 25 linhas com autor, URL, categoria, data e medidas linguísticas.
- `size_normalized_texts/{fake,true}/`: versões truncadas por par. É a mesma coleção de 7.200 notícias, e não 7.200 novos exemplos.
- `preprocessed/pre-processed.csv`: 7.200 linhas, colunas `index`, `label`, `preprocessed_news`. O parser CSV produz strings; conversões devem ser explícitas.
- Ingestão produz `id: str`, `pair_id: str`, `label: str`, `label_id: int`, `text: str`, `language: str`, `source_path: str`. Mapeamento explícito: **true = 0; fake = 1**.
- As duas variantes têm **3.600 pares completos**, sem textos vazios. O nome numérico do arquivo identifica o par.

O pareamento e o significado das variantes também estão descritos no [repositório original](https://github.com/roneysco/Fake.br-Corpus). O CSV remove informações como acentuação e stopwords; para BERTimbau cased, a recomendação é conservar texto e acentos. Não concatenar as variantes na futura divisão de treino/teste.

### AVeriTeC — alegações e explicações em inglês

Armazenamento: **um array JSON por split**. Foram adquiridas **3.068 alegações de treino + 500 de desenvolvimento**. O repositório selecionado não inclui o teste; não contamos as 1.000 alegações restantes da coleção original como adquiridas. A coleção de referência tem 4.568 alegações. [Página oficial do AVeriTeC](https://fever.ai/dataset/averitec.html).

Tipos verificados em todo o treino:

| Caminho | Tipo Python | Presença / observação |
|---|---|---|
| `claim` | `str` | 3.068 presentes; 1 vazia |
| `label` | `str` | 3.068 presentes e não vazios |
| `justification` | `str` | 3.068 presentes e não vazias |
| `questions` | `list[dict]` | 3.068 listas não vazias |
| `questions[].question` | `str` | 7.985 presentes; 1 vazia |
| `questions[].answers` | `list[dict]` | 7.985 listas não vazias |
| `questions[].answers[].answer` | `str` | 8.479 respostas não vazias |
| `questions[].answers[].answer_type` | `str` | `Extractive`, `Abstractive`, `Boolean`, `Unanswerable` |
| `questions[].answers[].source_url` | `str` | 8.479 presentes; 387 vazias |
| `questions[].answers[].source_medium` | `str` | 8.479 presentes; 387 vazias |
| `questions[].answers[].cached_source_url` | `str`, quando presente | 1.968 ausentes; 1 string vazia |
| `questions[].answers[].boolean_explanation` | `str`, quando presente | 1.297 presentes; campo opcional |

**`answers` não é uma chave da raiz:** cada pergunta possui sua própria lista de respostas. `required_reannotation` é `bool`; datas e diversos metadados admitem `None`. `claim_date` permanece string e inclui datas com dia primeiro. Ausência de uma chave, `null`, string vazia e lista vazia são condições diferentes; a EDA as registra sem preencher artificialmente.

As URLs das respostas servem como procedência. `source_medium` registra também imagens, áudio e vídeo, mas as anotações baixadas **não fornecem os respectivos tensores ou arquivos de mídia**. Não converter a existência desse campo em uma alegação de que esta entrega é multimodal.

### FEVEROUS — texto e tabelas da Wikipédia em inglês

Armazenamento: **JSONL**, com um objeto por linha. Os arquivos originais começam com um objeto de campos vazios; no teste, ele é `{"claim": ""}`. Excluir esse cabeçalho evita adicionar uma alegação fictícia a cada split.

| Caminho | Tipo observado | Significado |
|---|---|---|
| `id` | `int` | Identificador em treino/dev |
| `claim` | `str` | Alegação |
| `label` | `str` | `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO` |
| `evidence` | `list[dict]` | 1 a 3 conjuntos alternativos de evidências observados |
| `evidence[].content` | `list[str]` | IDs de elementos que fundamentam o conjunto |
| `evidence[].context` | `dict[str, list[str]]` | Cada ID é associado a outros IDs de contexto |
| `challenge` | `str` | Tipo de dificuldade de verificação |
| `annotator_operations` | `list[dict]` | Operações dos anotadores; não são entrada natural do usuário |
| `annotator_operations[].operation/value/time` | `str` | Inclusive `time` chega como string nos arquivos adquiridos |

O teste adquirido contém somente `claim`. Não há rótulos, evidências ou `id` oficial; qualquer índice local serve apenas para rastreio. O rótulo ausente não significa `NOT ENOUGH INFO`.

Um ID como `Michael Folivi_cell_1_2_0` identifica uma célula, e não seu texto. Os tipos incluem `sentence`, `cell`, `header_cell`, `table_caption` e `item`. `context` referencia títulos, seções e cabeçalhos. A resolução textual depende do **snapshot da Wikipédia de dezembro de 2020**, distribuído separadamente em JSONL/SQLite. Esse dump não foi baixado e os textos das evidências não foram analisados. [Formato e arquivos oficiais do FEVEROUS](https://fever.ai/dataset/feverous.html).

## 3. Distribuição dos rótulos

As porcentagens são calculadas dentro de cada split rotulado, antes de limpeza ou reamostragem.

| Dataset / split | Rótulo | N | % |
|---|---|---:|---:|
| Fake.br / cada variante | fake | 3.600 | 50,00 |
| Fake.br / cada variante | true | 3.600 | 50,00 |
| AVeriTeC / train | Refuted | 1.742 | 56,78 |
| AVeriTeC / train | Supported | 849 | 27,67 |
| AVeriTeC / train | Not Enough Evidence | 282 | 9,19 |
| AVeriTeC / train | Conflicting Evidence/Cherrypicking | 195 | 6,36 |
| AVeriTeC / dev | Refuted | 305 | 61,00 |
| AVeriTeC / dev | Supported | 122 | 24,40 |
| AVeriTeC / dev | Not Enough Evidence | 35 | 7,00 |
| AVeriTeC / dev | Conflicting Evidence/Cherrypicking | 38 | 7,60 |
| FEVEROUS / train | SUPPORTS | 41.835 | 58,68 |
| FEVEROUS / train | REFUTES | 27.215 | 38,17 |
| FEVEROUS / train | NOT ENOUGH INFO | 2.241 | 3,14 |
| FEVEROUS / dev | SUPPORTS | 3.908 | 49,53 |
| FEVEROUS / dev | REFUTES | 3.481 | 44,12 |
| FEVEROUS / dev | NOT ENOUGH INFO | 501 | 6,35 |

FEVEROUS: **71.291 treino + 7.890 dev + 7.845 teste sem rótulos = 87.026 alegações**. O balanceamento do teste não é mensurável com o arquivo público adquirido.

Decisão para a etapa seguinte:

- **Fake.br:** não precisa de oversampling por frequência. Verificar novamente depois da limpeza e divisão por grupos.
- **AVeriTeC:** razão maior/menor classe de **8,93×** no treino. Primeiro auditar repetidos e conflitos; depois comparar amostragem moderada por classe com distribuição original. Repetir toda a classe minoritária até igualar a maioria pode intensificar memorização.
- **FEVEROUS:** razão de **18,67×** no treino. Avaliar pesos de classe em uma cabeça discriminativa ou amostragem balanceada no gerador. A proporção de NEI muda de 3,14% no treino para 6,35% em dev; isso afeta calibração.
- Fazer reamostragem **somente após a divisão**, somente no treino. Para geração, balancear exemplos por alegação, sem desmembrar suas perguntas entre splits. Não aplicar SMOTE diretamente sobre texto/token IDs.
- Medir macro-F1, precisão/recall/F1 por classe, matriz de confusão e calibração. Para geração, acrescentar sustentação da justificativa nas evidências, precisão das citações e revisão humana. Accuracy isolada encobre as classes pequenas.
- Preservar as três taxonomias. `fake` de uma notícia completa, `REFUTES` condicionado a evidências e `Not Enough Evidence` têm semânticas diferentes; a classe de evidência conflitante também não deve ser colapsada em uma classe binária.

## 4. Qualidade, vazamento e tamanho dos exemplos

- **Fake.br:** médias de **185,89 palavras nas falsas e 1.100,29 nas verdadeiras** em `full_texts`. Comprimento é um forte atalho potencial. Em `size_normalized_texts`, as médias são **182,76 e 184,31**, respectivamente. A contagem por espaços desta EDA não replica necessariamente o tokenizador usado pelos autores.
- Na variante normalizada, P95 = **371 palavras**, P99 = **584,04**, máximo = **2.095**. Esses números não são contagens WordPiece/BPE. Ainda será necessário medir tokens e taxa de truncamento com o tokenizer escolhido, incluindo especiais e template.
- Há uma duplicata normalizada em cada variante do Fake.br: `true/61` e `true/69`. Além de manter cada par junto, unir grupos conectados por duplicatas para evitar vazamento. As variantes não podem ocupar splits diferentes.
- **AVeriTeC:** 41 grupos de alegações repetidas no treino; 13 desses grupos têm rótulos diferentes. Há 6 grupos em dev e **4 textos normalizados comuns entre treino e dev**. Comparar datas, contexto e evidências antes de chamar esses casos de erros de anotação. O arquivo bruto foi preservado.
- O AVeriTeC tem uma alegação vazia e uma pergunta vazia no treino. Quarentenar esses casos na futura curadoria; não treinar silenciosamente com eles.
- Das 387 respostas de treino sem `source_url`, **386 são `Unanswerable` e 1 é `Boolean`**. Preservar o sinal de insuficiência e revisar separadamente a resposta booleana sem fonte, sem inventar URLs.
- A mediana de perguntas por alegação no AVeriTeC é **2**, P95 = **6**, máximo = **16**. A justificativa tem mediana de **18 palavras**, P95 = **51**. O dataset supervisiona justificativas curtas; explicações extensas exigiriam uma política de anotação adicional.
- Somando alegação, perguntas, respostas, explicações booleanas e justificativa no treino do AVeriTeC: mediana = **120 palavras**, P95 = **369**, máximo = **1.221**. Isso exclui URLs e templates; dimensionar a sequência em tokens somente após a serialização final.
- **FEVEROUS:** P95 = **14 IDs únicos de evidência por alegação** no treino; máximo = **254**. Isso indica cauda longa, mas **não permite inferir o tamanho em tokens das evidências resolvidas**. Não há texto normalizado comum entre os splits adquiridos; essa checagem não detecta paráfrases nem páginas compartilhadas.
- Os fingerprints usam NFKC + casefold + normalização de espaços + SHA-256. Não são uma análise semântica. Datas, tópicos, fontes e grupos de fatos ainda precisam ser considerados na avaliação futura.

## 5. Modelo para classificação linguística binária

**Base recomendada: `neuralmind/bert-base-portuguese-cased` (BERTimbau Base).** É um encoder em português brasileiro com 12 camadas e cerca de 110 milhões de parâmetros. Sua escala é adequada a uma coleção de 7.200 textos e permite fine-tuning completo com margem maior que modelos generativos de bilhões de parâmetros. [Model card do BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased).

Preparação recomendada, sem executar treinamento agora:

- Futuramente carregar como `AutoModelForSequenceClassification`, com dois logits, e o tokenizer cased correspondente. Preservar acentos, pontuação e maiúsculas.
- Executar PyTorch/Transformers sobre **MPS** no Mac; MLX-LM não é necessário para esta cabeça BERT. [Backend de treinamento no Mac, Apple](https://developer.apple.com/metal/pytorch/).
- Começar com 256 tokens e batch pequeno (por exemplo 8), avaliar truncamento e comparar com 512 tokens. Para notícias longas, definir janelas com agregação por notícia; janelas da mesma notícia devem permanecer no mesmo split.
- Fine-tuning completo é o primeiro candidato. Somente os pesos FP32 de 110M consomem aproximadamente **0,41 GiB**; pesos, gradientes e dois momentos Adam em FP32 dão **1,64 GiB**, antes de ativações, buffers e runtime. Não há necessidade inicial de QLoRA para esse porte.
- Usar pares e grupos de duplicatas na divisão; avaliar também por fonte/tempo. Adicionar depois uma baseline simples de TF-IDF + regressão logística para medir quanto o encoder melhora sobre sinais superficiais.
- O resultado é um classificador dos padrões da coleção. A probabilidade de `fake` não estabelece a verdade de uma nova alegação. A checagem factual futura dependerá de evidências e de avaliação fora do domínio do corpus.

## 6. Modelo gerador de justificativas e perguntas

**Candidato principal: `meta-llama/Llama-3.1-8B-Instruct`**, em formato MLX quantizado a 4 bits, com a base congelada e adaptadores LoRA. A escolha é uma hipótese de engenharia a validar por qualidade e memória. O modelo declara suporte a português e inglês, facilitando o futuro uso em português com anotações AVeriTeC em inglês. [Model card oficial](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

**Alternativas:** `Mistral-7B-Instruct-v0.3` para comparação em inglês; `Llama-3.2-3B-Instruct` para o primeiro ensaio quando a margem de memória/latência for prioritária. A aptidão em português deve ser medida nos dados do projeto. [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

Um decoder não consome dicionários Python diretamente. Na próxima fase, serializar entradas e alvos com limites claros:

| Tarefa futura | Entrada | Alvo |
|---|---|---|
| Perguntas investigativas | Alegação e metadados disponíveis no momento da consulta | Perguntas anotadas |
| Veredito e justificativa | Alegação + evidências/QA disponíveis + identificadores de fonte | Rótulo + justificativa + referências |

`label` e `justification` não entram no prompt que deverá predizê-los. `fact_checking_article` é proveniência da anotação; seu veredito não pode vazar para a entrada de uma avaliação de busca aberta. AVeriTeC oferece respostas anotadas e URLs, não o conjunto de resultados que uma Search API retornaria hoje. Desempenho com evidência anotada deve ser separado de desempenho ponta a ponta.

| Parâmetro de preparação | Ponto inicial sugerido |
|---|---|
| Base | 4 bits; congelada |
| Adaptação | LoRA sobre a base quantizada (QLoRA no MLX-LM) |
| Rank | 8; comparar 16 depois |
| Camadas adaptadas | Últimas 4–8, ampliando somente após medição |
| Microbatch | 1 exemplo |
| Acumulação | 8 passos como ponto de partida |
| Sequência total | 1.024–2.048 tokens, incluindo prompt e resposta |
| Ativações | Gradient checkpointing; FP16/BF16 conforme configuração suportada |
| Loss | Apenas tokens da resposta, com mascaramento do prompt |

O MLX-LM suporta famílias Llama/Mistral, LoRA sobre modelos quantizados, acumulação de gradientes, checkpointing e mascaramento de prompt. Sua quantização não deve ser presumida idêntica ao NF4/otimizador paginado de uma receita CUDA/bitsandbytes. A extensão de contexto e as camadas adaptadas devem ser calibradas com o pico real. [Guia oficial de LoRA/QLoRA do MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

Como planejamento, um 8B em FP16 requer **14,90 GiB somente em pesos**. Em 4 bits, o limite ideal é **3,73 GiB**; incluindo escala e bias FP16 por grupo de 64, a aproximação é **4,19 GiB**, antes de módulos não quantizados, adaptadores e runtime. Os números usam 8 bilhões arredondados e não são o tamanho exato de um checkpoint. Fine-tuning completo com um orçamento de 16 bytes por parâmetro chega a **119,21 GiB**, antes das ativações: não é viável neste Mac.

Planejar margem de **6–8 GB para macOS e outros processos**, verificando a pressão real de memória. Os 24 GB são compartilhados entre CPU e GPU; buffers de dados, cache de arquivos e tensores disputam o mesmo conjunto de memória. [Arquitetura de memória unificada no MLX](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

## 7. Gargalos de memória com FEVEROUS no MLX

**Primeiro, resolver os IDs.** O dump é armazenamento; não precisa caber inteiro em RAM. Na fase seguinte, manter SQLite em disco, buscar páginas necessárias, preservar títulos/cabeçalhos/unidades das células e serializar conjuntos completos dentro de um orçamento de tokens. Evitar converter a Wikipédia inteira para objetos Python ou tensores e evitar padding global ao maior exemplo. Esses procedimentos são recomendações de ingestão futura, não implementação de retrieval nesta entrega.

Os limites principais são:

1. **Ativações de treino:** crescem com batch, comprimento, largura e camadas que participam do backward. Congelar pesos reduz gradientes/estados do otimizador, mas não elimina as ativações necessárias para treinar adaptadores. Checkpointing troca computação por memória.
2. **Atenção:** uma matriz densa de scores ocupa `B × H × T² × bytes`. Kernels eficientes podem evitar materializá-la inteira; portanto, o tamanho abaixo não é uma previsão automática do pico do MLX. O custo computacional de atenção global ainda cresce quadraticamente.
3. **Logits:** a saída sobre todo o vocabulário ocupa `B × T × V × bytes`. Uma loss que materializa logits para todas as posições pode custar GiBs, mesmo com pesos em 4 bits. Casts FP32 ou temporários podem aumentar esse valor.
4. **KV cache de inferência:** ocupa `2 × L × B × T × Hkv × Dhead × bytes`. É diferente das ativações de treinamento; no SFT convencional o cache de geração é desativado. Quantizar pesos não quantiza automaticamente esse cache.
5. **Expansão de tabelas e buffers:** cabeçalhos repetidos, linhas inteiras, conversões entre formatos e batches com comprimentos desiguais podem ultrapassar o orçamento antes do que a contagem de IDs sugere. Manter só o contexto necessário para entender cada evidência, sem truncar a conclusão ou evidência decisiva.

Exemplo analítico para Llama 3.1 8B: `L=32`, `H=32`, `Hkv=8`, `Dhead=128`, `V=128256`, batch 1, tensores de 2 bytes. Os parâmetros vêm da [definição oficial da arquitetura](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py).

| Tokens totais | KV cache de inferência | Uma matriz densa de atenção, uma camada | Logits de todas as posições |
|---:|---:|---:|---:|
| 2.048 | 0,25 GiB | 0,25 GiB | 0,49 GiB |
| 4.096 | 0,50 GiB | 1,00 GiB | 0,98 GiB |
| 8.192 | 1,00 GiB | 4,00 GiB | 1,96 GiB |
| 32.768 | 4,00 GiB | 64,00 GiB | 7,83 GiB |
| 131.072 | 16,00 GiB | 1.024,00 GiB | 31,31 GiB |

Essas colunas não devem ser somadas como uma estimativa universal: cache de geração e ativações de treino pertencem a regimes diferentes, e kernels podem evitar algumas materializações. A janela nominal de 128k não significa que treinar com 128k caiba em 24 GB. `scripts/memory_budget.py` reproduz a tabela e permite variar batch e comprimento.

Na próxima etapa, medir `mlx.core.get_peak_memory()`, memória ativa/cache, RSS do processo e pressão de memória do macOS após avaliação efetiva das operações lazy. O allocator do MLX não representa sozinho o consumo total do sistema. Definir a expansão para 4k/8k somente depois de medir exemplos da cauda longa e verificar a preservação das evidências. [APIs de memória do MLX](https://ml-explore.github.io/mlx/build/html/python/memory_management.html).

## 8. Comandos e limites da validação

```bash
python3 scripts/acquire.py
python3 scripts/eda.py
python3 scripts/inspect_samples.py
python3 scripts/memory_budget.py --batch 1 --tokens 2048 4096 8192
python3 -m unittest discover -s tests -v
```

Os arquivos brutos não foram reamostrados, deduplicados, traduzidos ou divididos novamente. A decisão de oversampling é recomendação para a próxima etapa. Não foram medidas taxas reais de tokens, memória no M4, qualidade de classificadores ou fidelidade de geração. O comprimento das evidências textuais do FEVEROUS permanece dependente da aquisição e resolução do dump separado.
