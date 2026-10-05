# Contraponto — verificação investigativa de notícias

O Contraponto é um assistente de investigação de notícias que combina análise linguística, extração de alegações, busca de evidências e diálogo socrático. O objetivo não é substituir o julgamento do usuário, mas ajudá-lo a examinar criticamente uma informação e consultar as fontes utilizadas.

O MVP atual será textual e integrado ao Telegram. O módulo do WhatsApp, áudio e demais formatos multimodais ficam para versões posteriores.

Este repositório reúne:

- scripts reproduzíveis para adquirir as bases de dados;
- a Análise Exploratória de Dados (EDA) das quatro bases selecionadas;
- os requisitos e o backlog do MVP;
- a estrutura inicial do bot do Telegram e do motor central, que será incrementada pelas issues.

## Estrutura do projeto

```text
Fake-News-Challenge2/
├── src/
│   └── contraponto/
│       ├── bot.py                 # Estrutura inicial da interface Telegram
│       └── motor_ia.py            # Estrutura inicial do Divã Socrático
├── scripts/
│   ├── acquire.py                 # Aquisição reproduzível das bases
│   ├── eda.py                     # EDA estrutural
│   └── eda_visuals.py             # Métricas textuais e gráficos
├── data/                          # Criada localmente e ignorada pelo Git
│   ├── raw/                       # Downloads originais e manifesto
│   └── processed/                 # Dados gerados pelo processamento
├── Fake.br-Corpus/                # Clone local; ignorado pelo Git
├── reports/
│   ├── RELATORIO_TECNICO.md
│   ├── REQUISITOS_E_PRIORIZACAO.md
│   ├── ISSUES_MVP_TELEGRAM.md
│   ├── eda.json
│   ├── eda_nlp.json
│   ├── figures/
│   └── samples/
├── .gitignore
└── README.md
```

## Bases de dados utilizadas

| Base | Papel no projeto | Conteúdo local |
|---|---|---|
| [Fake.br](https://github.com/roneysco/Fake.br-Corpus) | Análise linguística de notícias reais e falsas em português brasileiro. | 7.200 notícias, balanceadas entre 3.600 reais e 3.600 falsas. |
| [FakeRecogna 2.0](https://huggingface.co/datasets/recogna-nlp/fakerecogna2-extrativa) | Complemento da análise linguística em português. | 52.800 resumos extrativos, balanceados entre as duas classes. |
| [AVeriTeC](https://github.com/MichSchli/AVeriTeC) | Estrutura investigativa, perguntas, respostas, justificativas e evidências. | 3.068 alegações de treino e 500 de desenvolvimento. |
| [FEVEROUS](https://fever.ai/dataset/feverous.html) | Verificação de alegações com evidências textuais e tabulares. | 87.026 alegações distribuídas entre treino, desenvolvimento e teste. |

O snapshot completo da Wikipédia usado pelo FEVEROUS **não é baixado**. O projeto mantém apenas os arquivos de anotações necessários à EDA; o MVP usará fontes externas atuais para recuperar evidências.

## Por que as bases não estão no GitHub?

As pastas `data/` e `Fake.br-Corpus/` estão no `.gitignore`. Elas não são enviadas ao GitHub porque contêm arquivos grandes, cópias de fontes externas e dados que podem ser obtidos novamente.

Cada integrante deve clonar este repositório e executar `scripts/acquire.py` para reconstruir localmente a mesma organização de dados. O script utiliza revisões fixadas quando a fonte permite, valida os formatos e registra origem, tamanho e checksum dos arquivos em `data/raw/manifest.json`.

Portanto:

- o GitHub armazena o código, os relatórios e o planejamento;
- cada computador mantém sua própria cópia das bases;
- ninguém deve usar `git add -f` para enviar `data/` ou `Fake.br-Corpus/`.

## Configuração inicial — clone do projeto e aquisição das bases

### 1. Pré-requisitos

- Git;
- Python 3.10 ou superior;
- conexão com a internet durante a aquisição;
- espaço local para as bases e os arquivos processados.

Os scripts de aquisição e EDA usam somente a biblioteca padrão do Python. Não é necessário instalar pacotes para executar esta etapa.

### 2. Clonar este repositório

**macOS, Linux ou Windows:**

```bash
git clone https://github.com/Muniz2811/Fake-News-Challenge2.git
cd Fake-News-Challenge2
```

### 3. Baixar todas as bases automaticamente

Esta é a forma recomendada. Não é necessário clonar cada base manualmente.

**macOS ou Linux:**

```bash
python3 scripts/acquire.py
```

**Windows / PowerShell:**

```powershell
python scripts/acquire.py
```

O processo pode demorar conforme a conexão. O script exibe `Download` para arquivos novos e `Verified/reused` quando encontra uma cópia local válida.

### 4. Estrutura criada pelo script

Depois da aquisição, os arquivos locais ficam assim:

```text
Fake-News-Challenge2/
├── Fake.br-Corpus/
│   ├── full_texts/
│   ├── size_normalized_texts/
│   └── preprocessed/
└── data/
    └── raw/
        ├── averitec/
        │   ├── train.json
        │   └── dev.json
        ├── fakerecogna2/
        │   └── fakerecogna_extrativo.csv
        ├── feverous/
        │   ├── train.jsonl
        │   ├── dev.jsonl
        │   └── test.jsonl
        └── manifest.json
```

O comportamento por base é:

- **Fake.br:** se `Fake.br-Corpus/` não existir, o script clona o repositório e posiciona a revisão definida no código. Se a pasta já existir, ela é inventariada sem executar `git pull`.
- **FakeRecogna 2.0:** o CSV é baixado do Hugging Face para `data/raw/fakerecogna2/`.
- **AVeriTeC:** os arquivos `train.json` e `dev.json` são baixados para `data/raw/averitec/`.
- **FEVEROUS:** as anotações de treino, desenvolvimento e teste são baixadas para `data/raw/feverous/`.

### 5. Reexecutar ou recuperar um download interrompido

O comando de aquisição pode ser executado novamente:

```bash
python3 scripts/acquire.py
```

Arquivos válidos são reutilizados. Downloads parciais usam arquivos temporários e podem ser retomados quando o servidor permite. Se um arquivo existente não corresponder ao manifesto, o script interrompe a execução em vez de sobrescrevê-lo silenciosamente.

Também é possível usar diretórios alternativos:

```bash
python3 scripts/acquire.py \
  --data-dir /caminho/para/data/raw \
  --fakebr-dir /caminho/para/Fake.br-Corpus
```

Ao usar caminhos alternativos, informe os mesmos parâmetros ao executar a EDA.

## Executar a EDA

Execute os comandos na ordem abaixo, depois que a aquisição terminar.

**macOS ou Linux:**

```bash
python3 scripts/eda.py
python3 scripts/eda_visuals.py
```

**Windows / PowerShell:**

```powershell
python scripts/eda.py
python scripts/eda_visuals.py
```

Os scripts produzem:

- `data/processed/`: dados processados localmente;
- `reports/EDA.md`: tabelas estruturais geradas pela análise;
- `reports/eda.json`: estatísticas em formato JSON;
- `reports/eda_nlp.json`: métricas textuais e diagnósticos adicionais;
- `reports/figures/`: gráficos SVG;
- `reports/samples/`: exemplos pequenos de registros.

Quando a aquisição usar diretórios alternativos, execute:

```bash
python3 scripts/eda.py \
  --data-dir /caminho/para/data/raw \
  --fakebr-dir /caminho/para/Fake.br-Corpus

python3 scripts/eda_visuals.py \
  --data-dir /caminho/para/data/raw \
  --fakebr-dir /caminho/para/Fake.br-Corpus
```

## Estrutura inicial do MVP

O código da aplicação ainda é propositalmente mínimo:

- [`src/contraponto/bot.py`](src/contraponto/bot.py) define o ponto inicial da interface com o Telegram;
- [`src/contraponto/motor_ia.py`](src/contraponto/motor_ia.py) define o resultado da análise e o contrato inicial do motor central.

O método principal do motor ainda não implementa a análise completa. Cada componente será acrescentado pelas issues do backlog, evitando concentrar toda a solução em um único arquivo antes de o grupo dividir o trabalho.

## Executar o bot com Docker (Windows, macOS e Linux)

Esta execução não exige Python nem ambiente virtual no computador. Instale
[Docker Desktop](https://docs.docker.com/desktop/) no Windows ou macOS; no Linux,
use Docker Desktop ou [Docker Engine](https://docs.docker.com/engine/install/)
com o [plugin Compose](https://docs.docker.com/compose/install/linux/).
No Windows, use contêineres Linux. Abra o Docker Desktop e aguarde o mecanismo
iniciar antes de executar os comandos.

### 1. Preparar o projeto e a configuração local

Clone o repositório e entre na pasta `Fake-News-Challenge2`. Use uma revisão que
contenha `Dockerfile` e `compose.yaml`; depois que este PR for integrado, eles
estarão disponíveis na `main`. Não é necessário baixar bases para executar o bot.

Confira a instalação:

```sh
docker --version
docker compose version
```

Se ainda não houver `.env`, copie o exemplo. **Preserve um `.env` já existente.**

Windows / PowerShell:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

macOS / Linux:

```sh
test -f .env || cp .env.example .env
```

Edite esse arquivo local no seu editor:

| Variável | Configuração |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Obrigatória. Token do bot criado com `/newbot` no `@BotFather`. |
| `OPENAI_API_KEY` | Opcional. Deixe vazia para extração por regras e verificação conservadora. |
| `GOOGLE_FACT_CHECK_KEY` | Opcional. Habilita Google Fact Check quando preenchida. |
| `TAVILY_API_KEY` | Opcional. Habilita busca web quando preenchida. |
| `MAX_CLAIMS_PER_MESSAGE` | Limite de alegações, por padrão `5`. |
| `AUTHORIZED_GROUPS` | IDs separados por vírgula; vazia permite os grupos em que o bot participar. |

Sem as duas chaves de busca, a análise continua sem evidências externas e se
abstém de concluir. Sem OpenAI, não são feitas chamadas à OpenAI. Configure as
chaves apenas no `.env` local, nunca no Dockerfile ou no workflow do GitHub.

O Compose entrega as variáveis ao processo por `env_file`; não copia nem monta
o arquivo `.env` dentro do contêiner. A imagem instala `requirements.txt` e
`requirements-inference.txt` e contém o código de `src/contraponto`, sem bases,
modelos, relatórios ou ferramentas de treino. O Compose monta a pasta local
`models/` em `/app/models`, somente para leitura, para habilitar a inferência.
Docker para treinamento fica para uma issue separada.

Para calcular o score TF-IDF + regressão logística, coloque seu artefato confiável
exatamente neste caminho, relativo à raiz do projeto:

```text
models/language_risk/baseline_tfidf_logreg.joblib
```

A pasta é `language_risk`, com underscore, e não `languagerisk`. O modelo continua
fora da imagem e do Git; cada computador precisa ter sua cópia. A configuração
do Compose fixa `LANGUAGE_RISK_MODEL_PATH` em
`/app/models/language_risk/baseline_tfidf_logreg.joblib`, prevalecendo sobre um
eventual caminho do Windows/macOS no `.env`. Não é necessário alterar o `.env`.
O artefato deve ser compatível com as versões de `requirements-inference.txt`;
um modelo ausente ou incompatível deixa o score indisponível e preserva a
investigação factual.

Depois de atualizar esta branch, reconstrua e recrie o serviço para instalar as
dependências e ativar a montagem:

```sh
git pull --ff-only
docker compose up -d --build --force-recreate bot
docker compose logs -f bot
```

Nos logs, confira `Modelo linguístico carregado: /app/models/language_risk/baseline_tfidf_logreg.joblib`.
Quando substituir somente o `.joblib`, use `docker compose restart bot` para
recarregar o modelo; não é necessário reconstruir a imagem nesse caso.

### 2. Construir e iniciar

Encerre qualquer outra instância usando o mesmo token: dois processos de polling
do mesmo bot podem causar conflito. Os comandos abaixo são iguais nos três
sistemas operacionais, sempre na raiz do repositório:

```sh
docker compose build
docker compose up
```

Aguarde a mensagem `Contraponto iniciado — aguardando mensagens...`.
Uma configuração obrigatória ausente gera erro explicativo nos logs.

Para executar em segundo plano:

```sh
docker compose up -d
docker compose logs -f bot
```

O contêiner não precisa publicar portas: o Telegram é consultado por polling.
A busca e o Telegram precisam de acesso à internet.

### 3. Testar no Telegram

Abra o chat privado do bot cujo token você configurou. Envie `/start` e `/ajuda`.
Depois envie, por exemplo:

> A universidade publicou um estudo sobre a qualidade da água.

Confira a confirmação inicial e o relatório por alegação. Com OpenAI vazia,
a extração usa o fallback de `claims.py`. Com os provedores vazios, espere
abstenção por ausência de evidências; a frase acima não garante uma conclusão
factual. Este teste manual verifica a conexão real com o Telegram.

### 4. Encerrar ou atualizar

```sh
docker compose down
```

O Compose envia `SIGTERM`; o polling encerra antes da remoção do contêiner,
com até 30 segundos de tolerância. As investigações temporárias estão em memória
e se perdem ao reiniciar. Para sair apenas do acompanhamento dos logs, use
`Ctrl+C`; com `up -d`, isso mantém o bot rodando.

Depois de atualizar o código ou modificar o `.env`, recrie o serviço:

```sh
docker compose up -d --build --force-recreate
```

## CI mínimo com GitHub Actions

O arquivo [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) executa em
PRs direcionados à `main`, pushes na `main` e acionamento manual. Há dois jobs:

1. **Python - lint e testes:** instala Python 3.13 e dependências de testes;
   impede `.env` rastreado; roda Ruff para erros críticos (sintaxe e nomes
   indefinidos) e toda a suíte `unittest`. O lint inicial não impõe uma
   reformatação geral ao projeto.
2. **Docker - build e smoke:** se o primeiro passar, valida Compose com valores
   fictícios, executa `docker compose build` e testa a imagem. Confere imports,
   usuário sem privilégios e ausência de arquivos indevidos; gera uma fixture
   sintética e verifica inferência equivalente com o modelo montado somente para
   leitura, sem modificar o artefato; exercita o polling
   real com transporte Telegram simulado, `/start`, mensagem textual sem OpenAI
   e encerramento com `SIGTERM`, exigindo código de saída zero.

O teste da imagem roda com `--network none` e não recebe o `.env` local.
Não exige tokens reais nem faz chamadas pagas. A instalação de dependências
durante o build exige rede. Isso é **CI**, sem implantação automática (CD),
publicação de imagem ou início de um bot real no GitHub.

### Configurar o CI passo a passo

1. Revise as mudanças na branch `chore/docker-bot-ci`. Confirme que o `.env`
   local existe e que `git ls-files .env` não retorna nada. A remoção desse
   arquivo do índice deve entrar no commit, preservando sua cópia local.
   Essa remoção não apaga segredos de commits antigos: credenciais expostas
   anteriormente precisam ser revogadas no respectivo provedor.
2. Faça commit dos arquivos desta implementação e envie a branch ao GitHub.
   Não inclua `.env`, bases ou modelos. Exemplo:

   ```sh
   git add Dockerfile .dockerignore compose.yaml README.md requirements-ci.txt .github/workflows/ci.yaml tests/docker/check_image.py tests/docker/model_probe.py tests/docker/polling_probe.py src/contraponto/bot.py tests/test_logging_security.py
   git diff --cached --stat
   git commit -m "chore: dockerizar bot e adicionar CI de testes e build"
   git push -u origin chore/docker-bot-ci
   ```

   Se `.env` ainda estiver rastreado em outro checkout, execute
   `git rm --cached -- .env` antes do commit; isso preserva o arquivo local.
3. No GitHub, abra **Settings → Actions → General** e permita GitHub Actions
   e as actions oficiais `actions/checkout` e `actions/setup-python`. O
   workflow solicita somente `contents: read`. **Não cadastre chaves reais
   em Actions Secrets para este CI.** Se Actions já estiver habilitado,
   nenhuma configuração adicional é necessária.
4. Abra um PR dessa branch para `main`. Na aba **Checks**, acompanhe os dois
   jobs. Também é possível vê-los em **Actions → CI do bot**. Corrija qualquer
   falha antes do merge; novos commits no PR repetem os checks automaticamente.
   O botão **Run workflow** fica disponível quando o workflow estiver na
   branch padrão; então permite executar manualmente e selecionar a branch.
5. Para bloquear merges com CI quebrado, após a primeira execução vá a
   **Settings → Rules → Rulesets**, crie uma regra para `main` e habilite
   **Require a pull request before merging** e **Require status checks to
   pass**. Selecione `Python - lint e testes` e `Docker - build e smoke`,
   conforme aparecerem na execução. Ative e salve a regra. Essa opção depende
   das permissões e da disponibilidade de rulesets para o repositório.
6. Depois que os checks passarem, revise e faça o merge do PR. A partir daí,
   os integrantes podem usar os comandos Docker na `main`. O CI não faz
   merge nem deploy por conta própria.

Para reproduzir as verificações Python localmente, use o Python do seu ambiente
virtual (no Windows, `.\.venv\Scripts\python.exe`; no macOS/Linux,
`./.venv/bin/python`):

```sh
python -m pip install -r requirements.txt -r requirements-inference.txt -r requirements-ci.txt
python -m ruff check --select E9,F821,F822,F823 src/contraponto tests
python -m unittest discover -s tests -v
```

Com Docker instalado e a imagem construída, execute também:

```sh
python tests/docker/check_image.py
```

Esse último comando é uma ferramenta de desenvolvimento/CI e exige Python no
host. **A execução normal do bot com Compose não exige Python no host.**
O teste automatizado simula Telegram; o teste manual do chat continua necessário
para validar seu token, conectividade e respostas dos provedores reais.

Na preparação desta implementação, o lint e os 66 testes Python passaram, e o
transporte simulado confirmou `/start` e o fallback local. Docker não estava
disponível nesse computador: o build da imagem e o encerramento por `SIGTERM`
no contêiner ainda precisam ser confirmados pelo job Docker ou por execução local.
O teste de montagem somente leitura e inferência na imagem também exige Docker.

## Testar o classificador linguístico no Telegram

O motor usa o pipeline completo salvo por `scripts/ml/train_baseline.py`: os dois
vetorizadores TF-IDF e a regressão logística treinada. Cada mensagem é transformada
com o vocabulário aprendido, sem novo treinamento. O score da classe `1=fake`
aparece em **Análise da Linguagem**, separado das conclusões factuais. É um score
experimental, não uma probabilidade calibrada de a notícia ser falsa. Nesta versão
ele não aciona alertas em grupos.

`OPENAI_API_KEY` é opcional. Deixe-a vazia ou remova-a do ambiente para executar
sem OpenAI: o classificador linguístico continua local, a extração usa regras e
as buscas continuam com as chaves dos provedores configurados. Sem a LLM, o
verificador se abstém quando não há comparação segura. Regras ainda podem
concluir apoiada, refutada ou conflitante com fact-checks comparáveis da mesma
alegação e rótulos reconhecidos. O token do Telegram continua
obrigatório. Remova também uma eventual chave OpenAI definida no terminal antes
de reiniciar, pois variáveis já exportadas têm precedência sobre o `.env`.

Para reconstruir o artefato depois de adquirir Fake.br e FakeRecogna conforme as
instruções acima, use os scripts existentes e as dependências do treino:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe scripts/ml/prepare_language_data.py
.\.venv\Scripts\python.exe scripts/ml/split_language_data.py
$figurePath = Join-Path (Get-Location).Path 'data/processed/language_risk/baseline/validation.png'
.\.venv\Scripts\python.exe scripts/ml/train_baseline.py --figure-output $figurePath
```

O treino usa semente 42, avalia somente a validação e salva as métricas em
`data/processed/language_risk/baseline/metrics.json`. Para comparar com outro PC,
confira as versões e os hashes dos arquivos de treino e validação registrados
nas métricas. Com os mesmos dados, divisões e ambiente, espera-se reprodução das
métricas; pequenas diferenças numéricas entre plataformas ainda são possíveis.

Na raiz do repositório, com o ambiente virtual já criado:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
Test-Path models/language_risk/baseline_tfidf_logreg.joblib
.\.venv\Scripts\python.exe src/contraponto/bot.py
```

O segundo comando precisa retornar `True`. Se o modelo estiver em outro lugar,
configure `LANGUAGE_RISK_MODEL_PATH` com o caminho do `.joblib` antes de iniciar o
bot. Caminhos relativos são resolvidos a partir da raiz do repositório, mesmo
quando o bot é iniciado em outra pasta. Use somente o artefato confiável gerado
pelo treino e o mesmo ambiente de versões. Não é necessário instalar matplotlib
para inferência.

Encerre o processo antigo do bot antes de iniciar a nova versão. No chat privado,
envie uma notícia em português e confira o score no relatório. Uma palavra ou
frase muito curta não é um teste representativo do classificador treinado com
notícias. A extração de alegações, Tavily, Google Fact Check e a verificação
continuam no fluxo. Se faltar o artefato, as dependências ou houver erro de
inferência, o relatório informa que a análise linguística está indisponível e
continua a investigação factual, sem inventar um score.

Testes locais, sem chamadas externas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Documentação principal

- [Guia com exemplos de TF-IDF, regressão logística e fallback de extração](reports/GUIA_REGRESSAO_LOGISTICA_E_EXTRACAO.md);
- `reports/EDA.md`: tabelas estruturais geradas novamente ao executar `scripts/eda.py`;
- [Relatório técnico](reports/RELATORIO_TECNICO.md): recomendações de modelagem;
- [Requisitos e priorização](reports/REQUISITOS_E_PRIORIZACAO.md): escopo reorganizado para o MVP no Telegram;
- [Backlog de issues](reports/ISSUES_MVP_TELEGRAM.md): labels, cronograma, issues, dependências e configuração do GitHub Project.

## Fluxo planejado do MVP

```text
Notícia em texto
├── Análise da linguagem
└── Extração de alegações
    ├── Busca de fact-checks existentes
    └── Busca de evidências
         ↓
Verificação de cada alegação
         ↓
Relatório com análise linguística e resultado factual separados
         ↓
Diálogo socrático no Telegram
```
