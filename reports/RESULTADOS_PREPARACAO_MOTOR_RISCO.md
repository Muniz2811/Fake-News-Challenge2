# Preparação dos Dados do Motor de Risco Linguístico

## 1. Objetivo

Esta etapa preparou os dados que serão usados no primeiro modelo de análise linguística. Foram integradas somente as bases **Fake.br** e **FakeRecogna 2.0**. AVeriTeC e FEVEROUS pertencem ao fluxo de investigação factual e não entram no treinamento deste classificador.

O resultado desta etapa ainda não é um modelo treinado. Foram implementados o contrato dos dados, a política de duplicatas e a divisão reproduzível entre treino, validação e teste.

## 2. Contrato dos dados

Cada registro possui texto, rótulo, identificação de origem e informações destinadas à auditoria. O contrato impede que metadados conhecidos por introduzir atalhos sejam usados acidentalmente pelo modelo.

| Papel | Campos | Uso |
|---|---|---|
| Entrada permitida | `text` | Único conteúdo fornecido ao classificador |
| Alvo | `label_id` | `1 = fake` e `0 = true` |
| Identificação | `id`, `dataset`, `source_record_id`, `split` | Rastreabilidade e reprodução |
| Auditoria | `audit.*` | Análise de qualidade e vazamento; nunca entra como feature |

Os metadados de auditoria incluem domínio, categoria, data, comprimento, hash normalizado do texto, hash da URL e grupo usado no split. A URL original não é copiada para o corpus; somente seu SHA-256 é preservado para agrupar registros oriundos da mesma página.

A saída do futuro modelo deverá ser interpretada como **risco linguístico**, e não como prova de que a notícia é verdadeira ou falsa.

## 3. Política de duplicatas

O corpus canônico possuía **60.000 registros** e **82 grupos de textos repetidos**.

- Quando o texto e o rótulo eram iguais, foi mantido o registro com metadados mais completos. Em caso de empate, foi escolhido o menor ID.
- Quando o mesmo texto possuía rótulos diferentes, todos os registros do grupo foram excluídos dos três splits. Os IDs e o motivo continuam registrados no manifesto para auditoria.
- Os dados brutos e o corpus canônico não foram modificados.

### Resultado da deduplicação

| Resultado | Registros |
|---|---:|
| Corpus de entrada | 60.000 |
| Cópias de mesmo rótulo removidas | 78 |
| Registros conflitantes excluídos | 8 |
| Corpus utilizável | **59.914** |

As 78 cópias removidas estavam distribuídas da seguinte maneira:

| Base e classe | Removidos |
|---|---:|
| Fake.br — `true` | 1 |
| FakeRecogna — `fake` | 10 |
| FakeRecogna — `true` | 67 |

Foram encontrados quatro grupos conflitantes, todos na FakeRecogna:

| Grupo | Registro `true` | Registro `fake` |
|---:|---|---|
| 1 | `fakerecogna2:11830` | `fakerecogna2:48693` |
| 2 | `fakerecogna2:19110` | `fakerecogna2:48550` |
| 3 | `fakerecogna2:2892` | `fakerecogna2:48638` |
| 4 | `fakerecogna2:8967` | `fakerecogna2:48899` |

Nenhum desses oito registros foi usado em treino, validação ou teste.

## 4. Estratégia de split

Foi adotada a proporção **70% / 15% / 15%**, com semente fixa `42`.

A estratificação considera simultaneamente a base e o rótulo (`dataset + label`). Antes da divisão, os registros são unidos em componentes conectados pelas seguintes chaves:

1. hash normalizado do texto;
2. `pair_id` dos pares falso/verdadeiro da Fake.br;
3. hash da URL de origem.

Todos os registros de um componente são colocados no mesmo split. Isso impede que versões relacionadas da mesma notícia apareçam no treino e na avaliação.

Foram formados **53.898 componentes**, dos quais **4.250** possuem mais de um registro. O maior componente possui 56 registros provenientes da mesma página. Após a limpeza, 12.475 registros não possuem URL; eles continuam protegidos pelas demais chaves quando aplicável.

## 5. Resultado dos splits

### Contagem total

| Split | Meta aproximada | Resultado | Percentual |
|---|---:|---:|---:|
| Treino | 41.940 | **41.939** | 69,9987% |
| Validação | 8.987 | **8.988** | 15,0015% |
| Teste | 8.987 | **8.987** | 14,9998% |
| **Total** | **59.914** | **59.914** | **100%** |

A diferença de um registro entre treino e validação é consequência do arredondamento dos quatro estratos e não representa desbalanceamento relevante.

### Distribuição por base e classe

| Base e classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Fake.br — `fake` | 2.520 | 540 | 540 | 3.600 |
| Fake.br — `true` | 2.519 | 540 | 540 | 3.599 |
| FakeRecogna — `fake` | 18.470 | 3.958 | 3.958 | 26.386 |
| FakeRecogna — `true` | 18.430 | 3.950 | 3.949 | 26.329 |
| **Total** | **41.939** | **8.988** | **8.987** | **59.914** |

As classes permaneceram próximas do equilíbrio:

| Split | `fake` | `true` |
|---|---:|---:|
| Treino | 20.990 | 20.949 |
| Validação | 4.498 | 4.490 |
| Teste | 4.498 | 4.489 |

Não é necessário aplicar oversampling por frequência nesta etapa.

## 6. Verificações de vazamento

| Verificação | Grupos presentes em mais de um split |
|---|---:|
| Mesmo texto normalizado | **0** |
| Mesmo par da Fake.br | **0** |
| Mesma URL de origem | **0** |

Essas verificações cobrem duplicatas exatas e vínculos disponíveis nas bases. Elas não garantem a detecção de paráfrases, notícias diferentes sobre o mesmo evento ou URLs distintas com o mesmo conteúdo. Esses casos deverão ser considerados na avaliação por fonte, período e comprimento.

## 7. Arquivos e reprodução

Código versionado:

- `scripts/ml/prepare_language_data.py`: cria e valida o corpus canônico;
- `scripts/ml/split_language_data.py`: aplica a deduplicação, o agrupamento e o split.

Artefatos locais, ignorados pelo Git:

- `data/processed/language_risk/corpus.jsonl`;
- `data/processed/language_risk/validation_report.json`;
- `data/processed/language_risk/splits/train.jsonl`;
- `data/processed/language_risk/splits/validation.jsonl`;
- `data/processed/language_risk/splits/test.jsonl`;
- `data/processed/language_risk/splits/split_manifest.json`.

Para reproduzir a etapa completa a partir das bases já adquiridas:

```bash
python3 scripts/ml/prepare_language_data.py
python3 scripts/ml/split_language_data.py
```

O manifesto registra a semente, a política aplicada, os registros removidos, os conflitos, as contagens e o SHA-256 de cada arquivo gerado.

## 8. Próxima etapa recomendada

O próximo passo é treinar uma baseline de **TF-IDF + regressão logística** usando somente o campo `text`. A avaliação deverá incluir F1 macro, precisão, revocação, matriz de confusão e resultados separados por base, classe e faixa de comprimento. O conjunto de teste deve permanecer fechado até a escolha final do modelo e do limiar de risco.
