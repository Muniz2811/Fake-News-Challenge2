# Análise de dados para verificação de notícias

Etapa inicial de um projeto de verificação de notícias com IA, dedicada à aquisição e análise exploratória (EDA) das bases Fake.br, AVeriTeC e FEVEROUS.

## Estrutura do projeto

```text
Challenge2/
├── scripts/
│   ├── acquire.py
│   └── eda.py
├── Fake.br-Corpus/           # Base original do Fake.br
├── data/
│   ├── raw/                  # Bases baixadas e manifesto de origem
│   └── processed/            # Dados padronizados do Fake.br
├── reports/
│   ├── EDA.md                # Relatório gerado pela análise
│   ├── eda.json              # Estatísticas em formato JSON
│   ├── RELATORIO_TECNICO.md  # Interpretação e recomendações de modelagem
│   └── samples/              # Exemplos de registros para consulta
├── .gitignore
└── README.md
```

As pastas `data/` e `Fake.br-Corpus/` ficam no ambiente local e são ignoradas pelo Git. Os scripts permitem obter os dados novamente.

## Scripts

- **[acquire.py](scripts/acquire.py):** clona o Fake.br quando necessário, baixa as anotações do AVeriTeC e FEVEROUS e registra a origem e os checksums dos arquivos.
- **[eda.py](scripts/eda.py):** analisa classes, tipos dos campos, dados ausentes, duplicatas e comprimentos. Gera os relatórios, as amostras e os JSONL padronizados do Fake.br.

O `eda.py` utiliza funções do `acquire.py`; os dois devem ser mantidos juntos.

## Como executar

É necessário ter **Python 3.10+**, **Git** e acesso à internet para baixar as bases. Os scripts usam apenas a biblioteca padrão do Python, sem instalação de pacotes adicionais.

Após clonar o repositório, abra o terminal na pasta do projeto, onde está este README, e execute os comandos na ordem abaixo.

**Windows / PowerShell:**

```powershell
# 1. Baixar as bases e registrar sua origem
python scripts/acquire.py

# 2. Executar a análise e gerar os relatórios
python scripts/eda.py
```

**macOS / Linux:**

```bash
python3 scripts/acquire.py
python3 scripts/eda.py
```

As bases completas não vêm no clone: são obtidas pelo `acquire.py`, que reaproveita os arquivos já presentes. Depois da aquisição, a análise pode ser executada offline. Os resultados são salvos em `reports/`, e os dados padronizados do Fake.br, em `data/processed/`.

## Bases e formatos

| Base | Resumo | Estrutura |
|---|---|---|
| [Fake.br](https://github.com/roneysco/Fake.br-Corpus) | 7.200 notícias em português: metade falsa, metade verdadeira. | TXT nas pastas `fake` e `true`, com metadados separados. Possui versões completas, truncadas e um CSV pré-processado. |
| [AVeriTeC](https://github.com/MichSchli/AVeriTeC) | 3.068 alegações de treino e 500 de validação, em inglês, com quatro classes. | JSON com `claim`, `label`, `justification` e `questions`. Cada pergunta contém sua lista de `answers`. |
| [FEVEROUS](https://github.com/Raldir/FEVEROUS) | 87.026 alegações em inglês, divididas entre treino, validação e teste. | JSONL com `claim`, `label` e `evidence`. O teste contém somente alegações, sem rótulos ou evidências. |

No Fake.br, arquivos com o mesmo número em `fake` e `true` formam um par. As versões completas e truncadas representam as mesmas notícias. No FEVEROUS, `evidence` contém IDs de frases e células de tabelas; seu texto depende de um dump separado da Wikipédia, não incluído.

Os resultados estão em [EDA.md](reports/EDA.md), e sua interpretação está no [relatório técnico](reports/RELATORIO_TECNICO.md).
