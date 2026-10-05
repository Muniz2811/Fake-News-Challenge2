# Contraponto

Bot do Telegram para investigar notícias em português. Recebe texto, calcula um
score linguístico local, extrai alegações, busca evidências e apresenta uma
conclusão por alegação com justificativa e fontes.

**O score TF-IDF + regressão logística mede associação aos padrões das bases de
treino; não confirma que uma notícia é falsa.** A verificação factual é separada
e usa: apoiada, refutada, evidência insuficiente, conflitante e não verificada.
Sem evidências ou comparação segura, o verificador se abstém.

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
   computador precisa ter sua cópia, compatível com `requirements-inference.txt`.
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
| `OPENAI_API_KEY` | Opcional; vazia habilita extração por regras e verificação conservadora. |
| `GOOGLE_FACT_CHECK_KEY` | Opcional; habilita busca de checagens existentes. |
| `TAVILY_API_KEY` | Opcional; habilita busca de evidências web. |
| `MAX_CLAIMS_PER_MESSAGE` | Máximo de alegações por texto; padrão `5`. |
| `AUTHORIZED_GROUPS` | IDs separados por vírgula; vazia permite os grupos em que o bot participar. |

Apague os valores ilustrativos das chaves opcionais que não for utilizar.
Sem OpenAI, `claims.py` divide o texto em frases, evita perguntas, frases com
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
├── reports/                 # Saída local regenerável de EDA; ignorada pelo Git.
├── .github/workflows/ci.yaml # GitHub Actions: lint, testes e build.
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-inference.txt
├── requirements-ci.txt
├── README.md
├── data/                    # Dados locais; ignorados pelo Git.
├── Fake.br-Corpus/           # Base local; ignorada pelo Git.
└── models/language_risk/     # Modelo local; ignorado pelo Git.
```

`reports/` contém resultados gerados pelos scripts de EDA, como JSON, gráficos e
amostras. Essa pasta não é versionada e pode ser recriada localmente. `data/`,
`Fake.br-Corpus/` e `models/` também são pastas locais e não acompanham o clone.

| Arquivo em `src/contraponto/` | Responsabilidade |
|---|---|
| `bot.py` | Comandos e mensagens do Telegram, confirmação inicial, relatório dividido em mensagens e armazenamento temporário de investigações/deep links. |
| `config.py` | Carrega variáveis, limites e grupos; valida o token obrigatório. |
| `motor_ia.py` | Coordena inferência, extração, busca e verificação; devolve `ResultadoAnalise` com score e vereditos individuais. |
| `language_risk.py` | Carrega uma vez o `.joblib` e aplica TF-IDF + regressão logística ao texto original. |
| `claims.py` | Produz alegações numeradas com texto, trecho e posição; usa OpenAI quando configurada ou regras como fallback. |
| `search.py` | Adaptadores Google Fact Check/Tavily e evidências padronizadas: título, URL, publicador, trecho, data e metadados de checagem. |
| `verifier.py` | Compara cada alegação com suas evidências, valida referências da LLM e mantém abstenção quando não há sustentação segura. |
| `socratic.py` | Mantém a estrutura interna das sessões socráticas; a integração completa com os handlers do Telegram ainda é futura. |

| Arquivo de suporte | Responsabilidade |
|---|---|
| `Dockerfile` | Constrói a imagem Python 3.13 e inicia o bot com usuário sem privilégios. |
| `compose.yaml` | Configura build, ambiente, modelo montado, reinício e encerramento. |
| `.dockerignore` / `.gitignore` | Excluem arquivos do contexto de build e do versionamento, respectivamente. |
| `.env.example` | Exemplo sem credenciais reais; `.env` guarda a configuração local. |
| `requirements.txt` | Dependências do bot, APIs e configuração. |
| `requirements-inference.txt` | Versões de joblib, NumPy, scikit-learn e SciPy para carregar o modelo. |
| `requirements-ci.txt` | Ruff para lint; não é instalado na imagem do bot. |
| `scripts/acquire.py` | Obtém Fake.br, FakeRecogna, AVeriTeC e anotações FEVEROUS, registrando um manifesto. |
| `scripts/eda.py` / `scripts/eda_visuals.py` | Analisam formatos, distribuição e características textuais; geram artefatos em `reports/`. |
| `scripts/ml/prepare_language_data.py` | Padroniza Fake.br e FakeRecogna em texto/rótulo e prepara os dados linguísticos. |
| `scripts/ml/split_language_data.py` | Separa treino, validação e teste, controlando duplicatas/vazamento. |
| `scripts/ml/train_baseline.py` | Treina TF-IDF de palavras/caracteres + regressão logística e salva modelo, métricas e gráfico de validação. |

O fluxo atual aceita texto no Telegram. Diálogo socrático adaptativo, WhatsApp,
áudio/imagem e treinamento com Docker são etapas futuras. Existem handlers de
alerta/deep link, mas o motor atual retorna `requer_revisao=False`, portanto
não aciona alertas automaticamente.

## Desenvolvimento, dados e treinamento local

Para rodar sem Docker ou executar testes no macOS/Linux, use Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-inference.txt -r requirements-ci.txt
python src/contraponto/bot.py
```

No Windows, crie com `py -3.13 -m venv .venv` e ative com
`.\.venv\Scripts\Activate.ps1`. Não execute o bot nativo junto com o contêiner.

Aquisição e treino são opcionais para quem já possui o modelo. No ambiente
virtual, para reconstruí-lo:

```bash
python scripts/acquire.py
python -m pip install matplotlib
python scripts/ml/prepare_language_data.py
python scripts/ml/split_language_data.py
python scripts/ml/train_baseline.py --figure-output "$(pwd)/data/processed/language_risk/baseline/validation.png"
```

O treino linguístico usa Fake.br e FakeRecogna, rótulos `0=true` / `1=fake` e
semente 42. Salva o `.joblib` em `models/language_risk/` e métricas em
`data/processed/language_risk/baseline/`; AVeriTeC e FEVEROUS não entram nessa
baseline. Para reproduzir resultados, mantenha dados, splits e versões iguais.
Para EDA, execute `python scripts/eda.py` e `python scripts/eda_visuals.py`;
eles podem recriar `reports/EDA.md` e os demais artefatos.

## Testes e CI

No ambiente virtual, execute:

```bash
python -m unittest discover -s tests -v
python -m ruff check --select E9,F821,F822,F823 src/contraponto tests
```

Os testes simulam APIs e não precisam de chaves reais. O lint detecta erros
críticos de sintaxe e nomes indefinidos; não impõe toda a formatação do projeto.

| Arquivo em `tests/` | O que verifica |
|---|---|
| `test_config.py` | Token obrigatório, OpenAI opcional e extração por regras. |
| `test_bot_report.py` | Relatório, deep link, separação linguagem/fatos e limite de mensagens com emojis. |
| `test_search.py` | Evidências, tipos de fonte, Tavily, ausência de chave e falhas de conexão. |
| `test_verifier.py` | Abstenção, conflitos, negação, referências inventadas, respostas inválidas, estabilidade e fontes separadas por alegação. |
| `test_language_risk.py` | Carregamento/inferência, equivalência com o pipeline, incompatibilidade e continuidade sem score. |
| `test_logging_security.py` | Ocultação de credenciais fictícias nos logs e tracebacks. |
| `docker/check_image.py` | Testa imagem limpa, inferência com montagem somente leitura, polling e encerramento por SIGTERM. |
| `docker/model_probe.py` | Gera um modelo sintético e confere seu score e proteção contra escrita na montagem. |
| `docker/polling_probe.py` | Simula o transporte Telegram para exercitar `/start` e uma mensagem pelo fluxo real sem OpenAI. |

Para testar a imagem, com Docker disponível e Python no host:

```bash
docker compose build
python tests/docker/check_image.py
```

Esse teste usa um artefato sintético, não altera seu modelo e roda os contêineres
com rede desabilitada. O build precisa de internet para baixar dependências.
Ele não substitui o teste manual com seu bot e os provedores reais.

O workflow [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) roda em PRs
para `main`, pushes na `main` e execução manual. Primeiro executa lint e testes;
depois valida Compose, constrói e testa a imagem. Para ativar, permita GitHub
Actions em **Settings → Actions → General** e abra o PR. Acompanhe **Checks**;
não cadastre tokens reais para esse CI. Para bloquear merges com falhas,
configure ruleset de `main` exigindo `Python - lint e testes` e
`Docker - build e smoke`. Não há deploy, treinamento nem publicação automática
da imagem.
