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

## Bot do Telegram: grupos autorizados

O bot encaminha ao `MotorIA` apenas mensagens textuais enviadas em grupos cujo ID esteja configurado em `AUTHORIZED_GROUPS`. Mensagens de grupos não cadastrados e mensagens que não sejam texto são ignoradas. O grupo autorizado fica restrito a IDs explícitos; se a variável estiver vazia, nenhum grupo é processado.

1. Copie `.env.example` para `.env` e preencha `TELEGRAM_BOT_TOKEN`.
2. Obtenha o `chat_id` por um método administrativo/seguro já aprovado pela equipe, consultando os metadados do grupo ou um registro administrativo que exponha somente o identificador. Não adicione bots de terceiros ao grupo para esse fim nem compartilhe mensagens ou atualizações do grupo com serviços externos. O ID de grupo/supergrupo costuma ser negativo.
3. Cadastre um ou mais IDs separados por vírgula em `.env`, por exemplo:

   ```dotenv
   AUTHORIZED_GROUPS=-1001234567890,-1009876543210
   ```

4. Inicie o bot com `python -m src.contraponto.bot` (ou o comando de execução usado no seu ambiente).

O texto é entregue diretamente a `MotorIA.analisar_texto`; este MVP não o grava em arquivo nem banco. O bot não registra o conteúdo das mensagens ou tokens nos logs. Neste estágio o motor é apenas um stub e não realiza análise de fake news. Fotos, vídeos, áudios, figurinhas e documentos não entram no fluxo de grupos.

### Privacy Mode

O Privacy Mode do Telegram fica ativado por padrão para bots em grupos. Nesse modo o bot normalmente recebe comandos direcionados a ele, respostas a mensagens do bot e alguns outros eventos, mas não recebe todas as mensagens textuais comuns. Para receber mensagens comuns do grupo, o bot precisa ser administrador do grupo ou ter o Privacy Mode desativado em `@BotFather` com `/setprivacy`; alterações de privacidade podem exigir remover e adicionar o bot novamente ao grupo. O projeto não altera essa configuração automaticamente. Consulte a [documentação de grupos do Telegram](https://core.telegram.org/bots/features#privacy-mode) antes de alterar a configuração do bot.

O método principal do motor ainda não implementa a análise completa. Cada componente será acrescentado pelas issues do backlog, evitando concentrar toda a solução em um único arquivo antes de o grupo dividir o trabalho.

## Documentação principal

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
