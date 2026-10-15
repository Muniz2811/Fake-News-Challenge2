# Contraponto

Bot do Telegram para investigar notícias em português. Recebe texto, calcula um
score linguístico local, extrai alegações, busca evidências e apresenta uma
conclusão por alegação com justificativa e fontes.

**O score linguístico mede associação aos padrões das bases de treino; não
confirma que uma notícia é falsa.** Por padrão, usa TF-IDF + regressão logística;
BERTimbau + regressão logística ou MLP podem ser ativados localmente. A
verificação factual é separada e usa: apoiada, refutada, evidência insuficiente,
conflitante e não verificada.
Sem evidências ou comparação segura, o verificador se abstém.

## Demonstração web: grupo, conversas privadas e bot

Interface inspirada no WhatsApp Web, independente do WhatsApp e do Telegram.
A apresentação usa a identidade **Contra — a pergunta é o começo**, com os PNGs
originais em `assets/`: logomarca, símbolo e wallpaper dos chats. O motor e o
bot do Telegram continuam sendo os mesmos componentes do Contraponto.
A tela de entrada oferece **Pessoa 1 a Pessoa 4**, com um navegador ativo por
perfil. Cada perfil tem cinco conversas:

- **Grupo:** os quatro participantes conversam sem análise automática. O botão **Analisar mensagem** abaixo de cada texto solicita a análise linguística e a investigação factual; o resultado fica visível no grupo. Não há limite de score para solicitar a análise.
- **Contra:** chat privado com o bot para verificar notícias e alegações.
- **Outras três pessoas:** conversas privadas entre dois perfis, sem análise automática. O botão **Encaminhar ao Contra** envia somente o texto da mensagem escolhida ao chat do bot de quem clicou. Somente quem encaminhou vê o relatório; os demais trechos da conversa não são enviados ao motor.

O backend chama o mesmo `MotorIA`, com o modelo definido em
`LANGUAGE_RISK_BACKEND`, sem duplicar extração, busca ou verificação. A interface
usa HTML/CSS/JavaScript, sem instalação de Node. Atualiza mensagens a cada
aproximadamente dois segundos e mostra conclusões e fontes por alegação,
separadas da análise da linguagem.

O score linguístico não comprova falsidade e não dispara análises no grupo.
No chat do bot, mensagens enviadas ou encaminhadas são investigadas diretamente. Estas regras são da demonstração web; os comandos
e fluxos do Telegram descritos abaixo continuam disponíveis.

Com o ambiente virtual ativo, na raiz do projeto:

```bash
python -m pip install -r requirements/web.txt
python src/contraponto/web.py
```

No Windows, sem ativar o ambiente, use `.\.venv\Scripts\python.exe` no lugar de
`python`. Abra **http://localhost:8080**. Para simular as quatro pessoas no mesmo
computador, use navegadores ou perfis de navegador diferentes; abas normais do
mesmo perfil compartilham o login. Um perfil é liberado ao sair ou pode ser
retomado após 60 segundos sem atividade do navegador anterior.

No `.env`, configure `WEB_ROOM_NAME` e, antes de compartilhar um deploy,
`WEB_ACCESS_CODE` com um código entregue aos participantes. A web não exige
`TELEGRAM_BOT_TOKEN`; usa as mesmas configurações de Ollama, Tavily e Google
descritas abaixo. As chaves ficam no servidor. Para BERTimbau nativo, instale
também `requirements/ml.txt` e mantenha seus artefatos locais.

Com Docker, o perfil web usa TF-IDF e monta o mesmo diretório de modelos:

```bash
docker compose -f compose.web.yaml up -d --build
docker compose -f compose.web.yaml logs -f web
docker compose -f compose.web.yaml down
```

A interface inteira usa **Crimson Text Regular (400)**, fornecida pela designer
em `assets/CrimsonText-Regular.ttf`. A fonte é carregada pelo próprio servidor,
incluída na imagem web e aplicada aos textos, títulos, botões e campos, sem
negrito ou itálico sintetizados e sem depender do Adobe Fonts. Sua licença
acompanha o arquivo em `assets/CrimsonText-OFL.txt`. Os logos permanecem em PNG.

Para deploy, use o target **`web`** do `Dockerfile`, porta **8080** (ou `PORT`
no ambiente) e health check **`/healthz`**. Rode **uma única instância/processo**,
com HTTPS fornecido pela hospedagem ou proxy, variáveis configuradas no servidor
e modelo confiável montado em `/app/models`. O servidor do deploy precisa
alcançar o endpoint do Ollama; se ele estiver no desktop via Tailscale, essa
conexão precisa existir também no servidor. O navegador não acessa o Ollama
diretamente. `docker compose up` continua iniciando apenas o Telegram.

Esta versão é uma demonstração com perfis compartilhados, sem contas pessoais
ou criptografia de ponta a ponta. Quem retomar um perfil verá seu histórico.
Cada conversa mantém até 80 mensagens em memória; reiniciar apaga mensagens e
sessões. As verificações são enfileiradas, uma por vez, e falhas preservam as
mensagens. Não há banco, anexos ou integração com o WhatsApp real.

## Usar Qwen local com Ollama

Não é necessária uma chave OpenAI. Com o aplicativo Ollama aberto, execute:

```bash
ollama --version
ollama pull qwen3:8b
ollama list
```

Se `ollama serve` responder `address already in use` na porta `11434`, isso
normalmente significa que o aplicativo já iniciou o servidor; não abra uma
segunda instância. Confirme com:

```bash
curl http://localhost:11434/api/tags
```

Para rodar o bot diretamente no macOS, configure no `.env`:

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:8b
LLM_TIMEOUT_SECONDS=90
LLM_MAX_TOKENS=1024
MAX_EVIDENCES_PER_CLAIM=2
MAX_EVIDENCE_SNIPPET_CHARS=600
```

Para rodar o bot dentro do Docker Desktop, use no lugar:

```dotenv
LLM_BASE_URL=http://host.docker.internal:11434/v1
```

O valor de `LLM_API_KEY` é exigido pelo formato do cliente, mas o Ollama local
não valida essa chave. O Qwen executa três tarefas: extração estruturada de
alegações, comparação NLI de cada alegação com seus próprios trechos e geração
das perguntas do diálogo socrático. No NLI, o modelo devolve somente IDs e
relações; justificativas, citações e categorias finais são montadas e validadas
em Python. As evidências são ranqueadas e limitadas antes da inferência. Saída
inválida ou servidor fora do ar ativa extração por regras ou abstenção, sem
derrubar o bot e sem esconder as fontes consultadas.

Para avaliar o checkpoint sem Telegram nem APIs de busca:

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen3:8b \
python scripts/evaluate_qwen.py
```

O script imprime um JSON com o checkpoint, as saídas e o resultado dos casos
de extração, NLI e diálogo. Ele consulta o Ollama configurado.

Para usar o Ollama de outro computador via Tailscale Serve, configure
`LLM_BASE_URL=https://` seguido do domínio `.ts.net` do desktop e `/v1`.
O cliente envia `Host: localhost` nesses endpoints para evitar o bloqueio 403
do Ollama quando ele está escutando em loopback. A conexão continua usando
HTTPS e validando o certificado do desktop; outros endpoints mantêm seu Host.
Após atualizar esse código, reinicie tanto a web quanto o bot do Telegram.

## Rodar no macOS com Docker Desktop

Não é necessário instalar Python para executar o bot com Docker.

1. Instale o [Docker Desktop para Mac](https://docs.docker.com/desktop/setup/install/mac-install/):
   escolha **Apple Silicon** para chips M1/M2/M3/M4… ou **Intel** conforme
   ** → Sobre Este Mac**. Abra o `.dmg`, arraste Docker para Aplicativos, abra
   o aplicativo e aguarde a inicialização.
2. No Terminal, confira:

   ```bash
   docker --version
   docker compose version
   docker info
   ```

3. Clone o repositório e entre no projeto:

   ```bash
   git clone https://github.com/Muniz2811/Fake-News-Challenge2.git
   cd Fake-News-Challenge2
   ```

   Se já clonou, use `git switch main` e `git pull --ff-only` dentro do
   repositório.
4. Crie a configuração **somente se ainda não existir**:

   ```bash
   test -f .env || cp .env.example .env
   ```

   Preencha o token do Telegram no seu editor. Preserve um `.env` já configurado.
5. Para habilitar o score, coloque o modelo confiável treinado pelo projeto em:

   ```text
   models/language_risk/baseline_tfidf_logreg.joblib
   ```

   A pasta é `language_risk`, com underscore. O modelo não vem no clone; cada
   computador precisa ter sua cópia, compatível com `requirements.txt`.
6. Encerre outras instâncias usando o mesmo token, construa e inicie:

   ```bash
   docker compose build
   docker compose up -d
   docker compose logs -f bot
   ```

   Procure `Contraponto iniciado — aguardando mensagens...` e, com modelo válido,
   `Modelo linguístico carregado: /app/models/language_risk/baseline_tfidf_logreg.joblib`.
7. Abra o chat privado do bot, envie `/start` e depois um texto, por exemplo:

   > A universidade publicou um estudo sobre a qualidade da água.

   Confira a confirmação inicial e o relatório. A frase é apenas um teste do
   fluxo; não garante uma conclusão factual. `/ajuda` apresenta as orientações.

### Configuração

| Variável no `.env` | Função |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Obrigatória; obtenha com `/newbot` no `@BotFather`. |
| `LLM_BASE_URL` | Opcional; endpoint `/v1` do Ollama. Vazia mantém fallback/abstenção. |
| `LLM_API_KEY` | Para Ollama, use `ollama`; não é uma credencial real. |
| `LLM_MODEL` | Checkpoint; use `qwen3:8b` com Ollama. |
| `LLM_TIMEOUT_SECONDS` | Timeout por chamada; padrão `90`. |
| `LLM_MAX_TOKENS` | Teto de geração por resposta; padrão `1024`. |
| `MAX_EVIDENCES_PER_CLAIM` | Máximo de evidências enviadas ao NLI; padrão `2`. |
| `MAX_EVIDENCE_SNIPPET_CHARS` | Máximo de caracteres por trecho no contexto; padrão `600`. |
| `OPENAI_API_KEY` | Compatibilidade opcional; não é necessária para usar Ollama. |
| `GOOGLE_FACT_CHECK_KEY` | Opcional; habilita busca de checagens existentes. |
| `TAVILY_API_KEY` | Opcional; habilita busca de evidências web. |
| `MAX_CLAIMS_PER_MESSAGE` | Máximo de alegações por texto; padrão `5`. |
| `AUTHORIZED_GROUPS` | IDs separados por vírgula; vazia não autoriza nenhum grupo. |
| `GROUP_ANALYSIS_MODE` | `command` (padrão): responde a `/analisar`; `automatic`: também analisa textos comuns e envia alerta + deep link. |
| `WEB_ROOM_NAME` | Nome do grupo na demonstração web. |
| `WEB_ACCESS_CODE` | Código compartilhado para entrar nos quatro perfis da web. |
| `LANGUAGE_RISK_BACKEND` | `tfidf` (padrão), `bertimbau_logreg` ou `bertimbau_mlp` para execução Python local. |
| `BERTIMBAU_LOGREG_METRICS_PATH` | Relatório local com o classificador BERTimbau + regressão e hashes dos artefatos. |
| `BERTIMBAU_MLP_METRICS_PATH` | Relatório local que identifica o checkpoint MLP escolhido pela validação. |
| `BERTIMBAU_MODEL_DIR` | Diretório local do checkpoint do encoder BERTimbau. |
| `BERTIMBAU_DEVICE` | `auto`, `cpu`, `mps` ou `cuda` para o encoder; padrão `auto`. |

Apague os valores ilustrativos das chaves opcionais que não for utilizar.
Sem uma LLM configurada, `claims.py` divide o texto em frases, evita perguntas, frases com
menos de quatro palavras e alguns marcadores de opinião, respeitando o limite.
Esse fallback é uma heurística, não um extrator treinado. O verificador pode
usar fact-checks comparáveis com rótulos reconhecidos; fora disso, se abstém.
Sem chaves de busca, não há evidências externas. Modelo ausente ou incompatível
deixa o score indisponível, mas preserva a investigação factual.

A imagem inclui código e dependências de execução/inferência. O Compose entrega
as variáveis ao processo e monta `models/` em `/app/models` **somente para
leitura**. Não incorpora `.env`, modelos, bases ou relatórios à imagem. O caminho
do modelo no Compose prevalece sobre `LANGUAGE_RISK_MODEL_PATH` no `.env`.

### Parar e atualizar

```bash
docker compose down                         # Encerra o bot com SIGTERM.
git pull --ff-only                          # Atualiza a branch atual.
docker compose up -d --build --force-recreate bot
```

Se substituir apenas o `.joblib`, use `docker compose restart bot` para
recarregá-lo. `Ctrl+C` sai dos logs sem parar o serviço iniciado com `-d`.
Investigações temporárias ficam em memória e se perdem ao reiniciar.

No Windows, instale Docker Desktop com contêineres Linux; no Linux, use Docker
Engine com Compose. Os comandos de execução são os mesmos. No PowerShell,
crie o exemplo com `if (-not (Test-Path .env)) { Copy-Item .env.example .env }`.

## Estrutura e função dos arquivos

```text
Fake-News-Challenge2/
├── src/contraponto/          # Aplicação do bot.
├── scripts/                 # Aquisição, EDA e scripts de treino em ml/.
├── tests/                   # Testes Python e integração Docker em docker/.
├── requirements/            # Perfis opcionais dev.txt, ml.txt e web.txt.
├── .github/workflows/ci.yaml # GitHub Actions: lint, testes e build.
├── Dockerfile
├── compose.yaml
├── compose.web.yaml
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── docs/                    # Documentos apenas locais; ignorados pelo Git.
├── reports/                 # Saídas locais regeneráveis; ignoradas pelo Git.
├── data/                    # Dados locais; ignorados pelo Git.
├── Fake.br-Corpus/           # Base local; ignorada pelo Git.
└── models/language_risk/     # Modelo local; ignorado pelo Git.
```

`reports/` contém resultados gerados pelos scripts de EDA, como JSON, gráficos e
amostras. Essa pasta não é versionada e pode ser recriada localmente. `data/`,
`Fake.br-Corpus/`, `models/`, `docs/` e `entrega_regressao_logistica/` também são
pastas locais e não acompanham o clone. `docs/` guarda os documentos de trabalho
do autor; as instruções compartilhadas de execução e treino ficam neste README.

| Arquivo em `src/contraponto/` | Responsabilidade |
|---|---|
| `bot.py` | Comandos e mensagens do Telegram, confirmação inicial, relatório dividido em mensagens e armazenamento temporário de investigações/deep links. |
| `web.py` / `web_static/` | API da demonstração e interface do chat; quatro perfis, grupo, conversas privadas e fila de verificação usando o mesmo motor. |
| `config.py` | Carrega variáveis, limites e grupos; valida o token obrigatório. |
| `llm.py` | Expõe um contrato único de JSON estruturado para Ollama ou outro endpoint compatível. |
| `motor_ia.py` | Coordena inferência, extração, busca e verificação; devolve `ResultadoAnalise` com score e vereditos individuais. |
| `language_risk.py` | Carrega o backend escolhido e aplica TF-IDF + regressão, BERTimbau + regressão ou BERTimbau + MLP ao texto original. |
| `model_catalog.py` | Reutiliza os classificadores carregados e compartilha o encoder BERTimbau entre regressão e MLP. |
| `claims.py` | Produz alegações numeradas com texto, trecho e posição; usa a LLM local quando configurada ou regras como fallback. |
| `search.py` | Adaptadores Google Fact Check/Tavily e evidências padronizadas: título, URL, publicador, trecho, data e metadados de checagem. |
| `verifier.py` | Compara cada alegação com suas evidências, valida referências da LLM e mantém abstenção quando não há sustentação segura. |
| `socratic.py` | Mantém sessões por usuário/chat, expiração, até três perguntas e síntese sem tratar respostas como fatos. |
| `diagnostics.py` | Correlaciona investigações com UUID técnico e registra duração e categoria de falha das etapas, sem conteúdo pessoal. |

| Arquivo de suporte | Responsabilidade |
|---|---|
| `Dockerfile` | Constrói a imagem Python 3.13 e inicia o bot com usuário sem privilégios. |
| `compose.yaml` | Configura build, ambiente, modelo montado, reinício e encerramento. |
| `compose.web.yaml` | Execução opcional da demonstração web na porta 8080. |
| `.dockerignore` / `.gitignore` | Excluem arquivos do contexto de build e do versionamento, respectivamente. |
| `.env.example` | Exemplo sem credenciais reais; `.env` guarda a configuração local. |
| `requirements.txt` | Bot, APIs e inferência TF-IDF com as versões usadas pelo modelo; base comum das duas imagens Docker. |
| `requirements/dev.txt` | Inclui bot, web e Ruff para testes/lint e CI. |
| `requirements/ml.txt` | Inclui o bot e acrescenta gráficos, PyTorch, Transformers e Hugging Face para os três treinos e inferência BERTimbau. |
| `requirements/web.txt` | Inclui o motor do bot e acrescenta FastAPI e Uvicorn para a interface web; usado somente pelo target web do Docker. |
| `scripts/deploy_demo.sh` | Atalho para iniciar, parar, reiniciar e acompanhar logs da demonstração com Docker Compose. |
| `scripts/acquire.py` | Obtém Fake.br, FakeRecogna, AVeriTeC e anotações FEVEROUS, registrando um manifesto. |
| `scripts/eda.py` / `scripts/eda_visuals.py` | Analisam formatos, distribuição e características textuais; geram artefatos em `reports/`. |
| `scripts/ml/prepare_language_data.py` | Padroniza Fake.br e FakeRecogna em texto/rótulo e prepara os dados linguísticos. |
| `scripts/ml/split_language_data.py` | Separa treino, validação e teste, controlando duplicatas/vazamento. |
| `scripts/ml/train_baseline.py` | Treina TF-IDF de palavras/caracteres + regressão logística e salva modelo, métricas e gráfico de validação. |

O fluxo atual aceita texto no Telegram e oferece o diálogo socrático pelo botão
do relatório privado. WhatsApp, áudio/imagem e treinamento com Docker são etapas
futuras. Em grupos autorizados, `/analisar` publica o relatório completo quando
usado como resposta a uma mensagem de texto. O modo automático opcional mantém
o alerta curto com deep link ao privado quando `requer_revisao=True`, em casos
inconclusivos ou com falhas externas. O score linguístico não decide o veredito
factual nem o alerta.

### Analisar uma mensagem no grupo

1. Adicione seu bot ao grupo, com permissão para enviar mensagens. Consulte o
   **@usuário exato** no perfil dele. Envie `/chatid@` seguido desse usuário,
   sem repetir o `@`. O comando
   mostra apenas o identificador do grupo atual, sem analisar suas mensagens.
2. Cadastre esse ID em `AUTHORIZED_GROUPS` no `.env`. Para vários grupos, use
   vírgulas, por exemplo `AUTHORIZED_GROUPS=-1001234567890,-1009876543210`.
   Configuração vazia não autoriza nenhum grupo. Mantenha `GROUP_ANALYSIS_MODE=command`
   (também é o padrão quando a variável não existe).
3. Reinicie o processo nativo ou, com Docker, reconstrua após atualizar o código:
   `docker compose up -d --build bot`. Não execute duas instâncias com o mesmo token.
4. Envie uma notícia no grupo. Use **Responder** nessa mensagem e mande
   `/analisar@` seguido do usuário exato do bot. O Telegram costuma completar
   o comando ao digitá-lo. Exemplo fictício: para um bot com usuário
   `@contraponto_test_bot`, use `/analisar@contraponto_test_bot`.
5. O bot confirma o recebimento e publica o relatório no grupo, dividido quando
   necessário. Usa o modelo definido em `LANGUAGE_RISK_BACKEND`; `/modelo` vale
   para o privado. O diálogo socrático permanece no chat privado.

O aviso **“não tem acesso às mensagens”** é esperado com o **Privacy Mode** ativo.
Comandos explicitamente dirigidos ao bot são recebidos mesmo nesse modo; não é
necessário torná-lo administrador. O relatório usa o texto original recebido em
`reply_to_message`; se o Telegram não o disponibilizar, o bot orienta a responder
a uma mensagem textual. Veja a [FAQ oficial](https://core.telegram.org/bots/faq#what-messages-will-my-bot-get).

Para manter o fluxo anterior de análise de textos comuns, configure
`GROUP_ANALYSIS_MODE=automatic` e reinicie. Nesse modo, o bot precisa receber
mensagens comuns: administradores as recebem; para um bot não administrador,
use `/setprivacy` no `@BotFather` para desativar a privacidade com a concordância
do grupo. Pode ser necessário remover e adicionar o bot novamente. O alerta
automático continua curto e leva ao relatório privado; o comando `/analisar`
continua disponível para pedir o relatório no grupo. Consulte a
[documentação do Privacy Mode](https://core.telegram.org/bots/features#privacy-mode).

Áudios, imagens, vídeos, documentos e figurinhas de grupos não entram na análise.
O processamento não grava mensagens em arquivos ou banco. Resultados de casos
que exigem revisão ficam temporariamente em memória para o deep link, com TTL
de 24 horas. Os handlers do bot registram apenas tipos de falha, sem incluir o
texto recebido ou IDs de grupos. A revisão dos logs dos demais componentes é
parte da MVP-14. O teste em grupo real continua necessário para validar a configuração.

## Desenvolvimento, dados e treinamento local

Para rodar sem Docker no macOS/Linux, use Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/contraponto/bot.py
```

No Windows, crie com `py -3.13 -m venv .venv` e ative com
`.\.venv\Scripts\Activate.ps1`. Não execute o bot nativo junto com o contêiner.

Escolha o perfil conforme a tarefa; os opcionais já incluem `requirements.txt`:

| Tarefa | Instalação |
|---|---|
| Rodar o bot com TF-IDF | `python -m pip install -r requirements.txt` |
| Executar testes e lint | `python -m pip install -r requirements/dev.txt` |
| Executar a demonstração web | `python -m pip install -r requirements/web.txt` |
| Treinar modelos ou usar BERTimbau no bot | `python -m pip install -r requirements/ml.txt` |

Não é preciso instalar todos para executar o bot. As versões de ML foram
mantidas para preservar a compatibilidade dos artefatos treinados.

Aquisição e treino são opcionais para quem já possui o modelo. No ambiente
virtual, para reconstruí-lo:

```bash
python -m pip install -r requirements/ml.txt
python scripts/acquire.py
python scripts/ml/prepare_language_data.py
python scripts/ml/split_language_data.py
python scripts/ml/train_baseline.py --figure-output "$(pwd)/data/processed/language_risk/baseline/validation.png"
```

O treino linguístico usa Fake.br e FakeRecogna, rótulos `0=true` / `1=fake` e
semente 42. Salva o `.joblib` em `models/language_risk/` e métricas em
`data/processed/language_risk/baseline/`; AVeriTeC e FEVEROUS não entram nessa
baseline. Para reproduzir resultados, mantenha dados, splits e versões iguais.

### Baseline BERTimbau Base + regressão logística

Esta experiência usa os mesmos splits de treino e validação da TF-IDF. O
BERTimbau fica congelado e transforma cada notícia em um vetor de 768 números;
uma regressão logística é treinada nesses vetores. Notícias longas são divididas
em blocos de até 256 tokens, sem descartar o restante do texto. O pooling faz a
média de todos os tokens de conteúdo, excluindo `[CLS]`, `[SEP]` e padding.
O split de teste permanece reservado. Os pesos, vetores e classificador são
artefatos locais ignorados pelo Git; o bot usa TF-IDF por padrão e esta baseline
somente quando `LANGUAGE_RISK_BACKEND=bertimbau_logreg` for selecionado.

No ambiente virtual, com os splits já preparados:

```bash
python -m pip install -r requirements/ml.txt
hf download neuralmind/bert-base-portuguese-cased \
  --revision 94d69c95f98f7d5b2a8700c420230ae10def0baa \
  --local-dir models/language_risk/bertimbau_base \
  --include config.json pytorch_model.bin tokenizer_config.json \
    special_tokens_map.json vocab.txt added_tokens.json
python scripts/ml/extract_bertimbau_embeddings.py
python scripts/ml/train_bertimbau_logreg.py
```

A extração pode levar bastante tempo sem GPU. Ela grava progresso a cada lote:
reexecute o mesmo comando para continuar após uma interrupção. Em uma máquina
com GPU NVIDIA compatível, use a opção `--device cuda`. A opção
`--splits validation` permite testar a velocidade antes de processar todo o
treino. Use `--restart` somente se quiser recalcular os splits selecionados
desde o início.
O treino produz `models/language_risk/bertimbau_logreg.joblib` e
`data/processed/language_risk/bertimbau/metrics.json`, que inclui a comparação
com a TF-IDF após conferir os hashes dos splits.
As métricas gerais e por recorte ficam nesse JSON local.

### BERTimbau Base + rede neural MLP

Para testar um classificador não linear sem repetir a vetorização, use os mesmos
embeddings, splits e ambiente da experiência anterior:

```bash
python scripts/ml/train_bertimbau_mlp.py
```

A MLP tem arquitetura `768 → 128 (ReLU + dropout 0,2) → 1`. O BERTimbau
permanece congelado; apenas a MLP é treinada. O script ajusta a normalização
somente no treino e executa as sementes 42, 43 e 44, com parada antecipada pela
validação. Grava um checkpoint por semente em `models/language_risk/` e o
relatório `data/processed/language_risk/bertimbau/metrics_mlp.json`. Esse
relatório compara média e desvio das três execuções à regressão sobre BERTimbau
e à baseline TF-IDF, após conferir os hashes dos dados. O melhor checkpoint é
apontado para inspeção, mas seus recortes usam a mesma validação empregada na
seleção e não substituem uma avaliação independente. Depois de fixar esse
checkpoint pela validação, o teste pode ser avaliado uma única vez:

```bash
python scripts/ml/extract_bertimbau_embeddings.py --splits test
python scripts/ml/evaluate_bertimbau_mlp.py
```

O avaliador confere os hashes dos vetores e do checkpoint, não treina nem
ajusta parâmetros com o teste e recusa sobrescrever um relatório de teste já
existente. O resultado fica em
`data/processed/language_risk/bertimbau/metrics_mlp_test.json`.

### Testar o bot com BERTimbau + regressão ou MLP

No chat privado, envie **`/modelo`** e escolha um dos três botões: TF-IDF +
regressão logística, BERTimbau + regressão logística ou BERTimbau + MLP.
O bot confirma a seleção após carregar o modelo. Envie uma notícia para testar
e confira o nome do modelo na seção de análise textual do relatório.

A escolha vale para o usuário naquele chat e é temporária: reiniciar o bot
restaura `LANGUAGE_RISK_BACKEND` do ambiente (TF-IDF por padrão). Até 1.000
preferências recentes ficam em memória. Grupos continuam usando o modelo
configurado no ambiente. Relatórios antigos e deep links preservam o modelo
da investigação original. Trocar o classificador não altera as buscas, o Qwen,
os vereditos factuais nem um diálogo socrático em andamento.

O primeiro carregamento do BERT pode demorar. O encoder é compartilhado pelos
dois classificadores, com inferências serializadas para controlar memória.
Modelo indisponível não substitui a seleção anterior; o bot informa o motivo.
Para usar as três opções neste notebook Windows, encerre a instância antiga
com **Ctrl+C** e, na raiz do projeto, execute:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/ml.txt
.\.venv\Scripts\python.exe -u src\contraponto\bot.py
```

Mantenha o Ollama no desktop e o Tailscale conectados. O BERT roda no computador
do bot; o endereço `LLM_BASE_URL` continua direcionando as tarefas do Qwen para
o desktop. Os artefatos locais necessários são:

```text
models/language_risk/baseline_tfidf_logreg.joblib
models/language_risk/bertimbau_base/
models/language_risk/bertimbau_logreg.joblib
models/language_risk/bertimbau_mlp_seed43.pt
data/processed/language_risk/bertimbau/metrics.json
data/processed/language_risk/bertimbau/metrics_mlp.json
```

O checkpoint MLP utilizado é o apontado pela seleção na validação do relatório;
nos artefatos atuais, é a semente 43. Não é necessário retreinar ou recalcular
os vetores para usar `/modelo`.

O classificador e o encoder são locais; o bot confere os hashes registrados
em `metrics.json` ou `metrics_mlp.json` antes de usá-los. Para iniciar o bot
**fora do Docker**, com Ollama aberto e `.env` já configurado:

```bash
source .venv/bin/activate
python -m pip install -r requirements/ml.txt
LANGUAGE_RISK_BACKEND=bertimbau_logreg python src/contraponto/bot.py
```

O comando ativa a regressão só neste processo; não precisa editar `.env`.
Confira no terminal `Modelo linguístico BERTimbau + regressão carregado.`.
O relatório do Telegram deve identificar `BERTimbau Base + regressão logística`
na seção linguística. Para comparar com a MLP, substitua `bertimbau_logreg` por
`bertimbau_mlp`. Se os artefatos ou dependências faltarem, o score ficará
indisponível e a investigação
factual continuará. Encerre qualquer outra instância do bot com o mesmo token
antes de iniciar, para evitar conflito no polling do Telegram. Para voltar ao
padrão, inicie com `LANGUAGE_RISK_BACKEND=tfidf`.

A imagem Docker atual instala apenas as dependências da TF-IDF, não PyTorch e
Transformers, e não monta os relatórios locais do BERTimbau. Portanto, esse
comando é para o ambiente Python do computador; não basta mudar o `.env` do Compose.
Os scores continuam experimentais e **não são vereditos factuais**.

Para EDA, execute `python scripts/eda.py` e `python scripts/eda_visuals.py`;
eles podem recriar `reports/EDA.md` e os demais artefatos.

## Testes e CI

No ambiente virtual, execute:

```bash
python -m pip install -r requirements/dev.txt
python -m unittest discover -s tests -v
python -m ruff check --select E9,F821,F822,F823 src/contraponto tests
```

Os testes simulam APIs e não precisam de chaves reais. O lint detecta erros
críticos de sintaxe e nomes indefinidos; não impõe toda a formatação do projeto.
Os testes específicos de PyTorch são ignorados quando ele não está instalado;
adicione o perfil `requirements/ml.txt` para executá-los também. O CI principal
usa somente o perfil `dev`, sem baixar modelos nem instalar BERTimbau.

| Arquivo em `tests/` | O que verifica |
|---|---|
| `test_config.py` | Token obrigatório, LLM opcional e extração por regras. |
| `test_local_llm.py` | Contratos estruturados, validação da origem, fallback e sessões socráticas sem executar modelo no CI. |
| `test_group_authorization.py` | Autorização, `/chatid`, `/analisar` em resposta, relatório no grupo, falhas, limites e preservação do alerta → deep link → relatório privado. |
| `test_bot_report.py` | Relatório, deep link, separação linguagem/fatos e limite de mensagens com emojis. |
| `test_search.py` | Evidências, tipos de fonte, Tavily, ausência de chave e falhas de conexão. |
| `test_verifier.py` | Abstenção, conflitos, negação, referências inventadas, respostas inválidas, estabilidade e fontes separadas por alegação. |
| `test_language_risk.py` | Carregamento/inferência, equivalência com o pipeline, incompatibilidade e continuidade sem score. |
| `test_model_selection.py` | Seleção privada isolada, cache, encoder compartilhado, falhas, concorrência e preservação do modelo em relatórios anteriores. |
| `test_web.py` | Quatro perfis exclusivos, autorização, isolamento de conversas privadas, bot privado, grupo, fila, falhas e preservação das fontes por alegação. |
| `test_bertimbau_baseline.py` | Retomada da extração, alinhamento dos splits e comparação das baselines sem baixar o checkpoint no CI; treino sintético da MLP quando PyTorch está instalado. |
| `test_motor_resilience.py` | Score preservado, falhas reais simuladas dos adaptadores, continuidade por alegação e marca de revisão sem perder resultados anteriores. |
| `test_logging_security.py` | Ocultação de credenciais fictícias nos logs e tracebacks. |
| `test_diagnostics_logging.py` | Duração, categorias de falha, isolamento de investigações simultâneas, propagação para threads e preservação dos resultados. |
| `docker/check_image.py` | Testa imagem limpa, inferência com montagem somente leitura, polling e encerramento por SIGTERM. |
| `docker/model_probe.py` | Gera um modelo sintético e confere seu score e proteção contra escrita na montagem. |
| `docker/polling_probe.py` | Simula o transporte Telegram para exercitar `/start` e uma mensagem pelo fluxo real sem OpenAI. |
| `docker/web_probe.py` | Testa HTTP real na imagem web, quatro perfis, grupo, privacidade, fallback e encerramento por SIGTERM sem rede externa. |

Para testar a imagem, com Docker disponível e Python no host:

```bash
docker compose build
python tests/docker/check_image.py
```

Esse teste usa um artefato sintético, não altera seu modelo e roda os contêineres
com rede desabilitada. O build precisa de internet para baixar dependências.
Ele não substitui o teste manual com seu bot e os provedores reais.

Os logs técnicos usam `correlation_id` (UUIDv4, sem vínculo com o usuário),
`stage` e `event`. `INFO` indica início e conclusão com duração; `WARNING`
indica indisponibilidade tratada por um adaptador; `ERROR` indica uma etapa
que lançou exceção. As categorias são `CONFIGURATION_FAILURE`,
`EXTERNAL_SERVICE_UNAVAILABLE` e `INTERNAL_ERROR`. Uma investigação pode
concluir com resultados parciais após uma falha; os avisos continuam no
relatório. Texto da notícia, IDs pessoais e mensagens das exceções não entram
nesses registros. Todos os testes, inclusive os de diagnóstico, usam `unittest`.

O workflow [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) roda em PRs
para `main`, pushes na `main` e execução manual. Primeiro executa lint e testes;
depois valida Compose, constrói e testa a imagem. Para ativar, permita GitHub
Actions em **Settings → Actions → General** e abra o PR. Acompanhe **Checks**;
não cadastre tokens reais para esse CI. Para bloquear merges com falhas,
configure ruleset de `main` exigindo `Python - lint e testes` e
`Docker - build e smoke`. Não há deploy, treinamento nem publicação automática
da imagem.
