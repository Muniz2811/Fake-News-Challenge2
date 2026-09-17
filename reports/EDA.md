# EDA executada — anotações textuais

Execução UTC: 2026-09-17T18:04:15.325464+00:00

Contagens calculadas sobre todos os registros adquiridos. Amostras em `samples/`.

Palavras são unidades separadas por espaços; **não são tokens de um modelo**.

## Rótulos por variante/split

| Dataset | Split | N | Rótulo | Contagem | % dos rotulados |
|---|---|---:|---|---:|---:|
| fakebr | full_texts | 7200 | fake | 3600 | 50.0 |
| fakebr | full_texts | 7200 | true | 3600 | 50.0 |
| fakebr | size_normalized_texts | 7200 | fake | 3600 | 50.0 |
| fakebr | size_normalized_texts | 7200 | true | 3600 | 50.0 |
| averitec | train | 3068 | Supported | 849 | 27.673 |
| averitec | train | 3068 | Refuted | 1742 | 56.78 |
| averitec | train | 3068 | Not Enough Evidence | 282 | 9.192 |
| averitec | train | 3068 | Conflicting Evidence/Cherrypicking | 195 | 6.356 |
| averitec | dev | 500 | Supported | 122 | 24.4 |
| averitec | dev | 500 | Refuted | 305 | 61.0 |
| averitec | dev | 500 | Not Enough Evidence | 35 | 7.0 |
| averitec | dev | 500 | Conflicting Evidence/Cherrypicking | 38 | 7.6 |
| feverous | train | 71291 | SUPPORTS | 41835 | 58.682 |
| feverous | train | 71291 | REFUTES | 27215 | 38.175 |
| feverous | train | 71291 | NOT ENOUGH INFO | 2241 | 3.143 |
| feverous | dev | 7890 | SUPPORTS | 3908 | 49.531 |
| feverous | dev | 7890 | REFUTES | 3481 | 44.119 |
| feverous | dev | 7890 | NOT ENOUGH INFO | 501 | 6.35 |
| feverous | test | 7845 | SUPPORTS | 0 | n/a |
| feverous | test | 7845 | REFUTES | 0 | n/a |
| feverous | test | 7845 | NOT ENOUGH INFO | 0 | n/a |
| feverous | test | 7845 | SEM RÓTULO | 7845 | n/a |

## fakebr / full_texts

Classes balanceadas: não aplicar oversampling por frequência.

Razão majoritária/minoritária: 1.0. Grupos de textos duplicados normalizados: 1; com rótulos conflitantes: 0.

Ocorrências de qualidade: `{}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_characters | 7200 | 46 | 2185.5 | 12026.15 | 16742.63 | 46084 | 3898.9 |
| text_words_whitespace | 7200 | 9 | 362.0 | 1985.05 | 2803.07 | 7517 | 643.09 |
| words_fake | 3600 | 9 | 157.0 | 378.05 | 644.01 | 2288 | 185.89 |
| words_true | 3600 | 20 | 918.0 | 2284.15 | 3209.13 | 7517 | 1100.29 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 7200 | n/a | 0 |
| `$.id` | str: 7200 | 0 | 0 |
| `$.label` | str: 7200 | 0 | 0 |
| `$.label_id` | int: 7200 | 0 | 0 |
| `$.language` | str: 7200 | 0 | 0 |
| `$.pair_id` | str: 7200 | 0 | 0 |
| `$.source_path` | str: 7200 | 0 | 0 |
| `$.text` | str: 7200 | 0 | 0 |

## fakebr / size_normalized_texts

Classes balanceadas: não aplicar oversampling por frequência.

Razão majoritária/minoritária: 1.0. Grupos de textos duplicados normalizados: 1; com rótulos conflitantes: 0.

Ocorrências de qualidade: `{}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_characters | 7200 | 48 | 957.0 | 2276.05 | 3584.53 | 12488 | 1114.67 |
| text_words_whitespace | 7200 | 9 | 157.0 | 371.0 | 584.04 | 2095 | 183.54 |
| words_fake | 3600 | 9 | 157.0 | 369.05 | 573.11 | 2088 | 182.76 |
| words_true | 3600 | 10 | 158.0 | 372.0 | 588.02 | 2095 | 184.31 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 7200 | n/a | 0 |
| `$.id` | str: 7200 | 0 | 0 |
| `$.label` | str: 7200 | 0 | 0 |
| `$.label_id` | int: 7200 | 0 | 0 |
| `$.language` | str: 7200 | 0 | 0 |
| `$.pair_id` | str: 7200 | 0 | 0 |
| `$.source_path` | str: 7200 | 0 | 0 |
| `$.text` | str: 7200 | 0 | 0 |

## averitec / train

Há desbalanceamento: avaliar pesos/amostragem apenas no treino; não fazer oversampling automaticamente.

Razão majoritária/minoritária: 8.933. Grupos de textos duplicados normalizados: 41; com rótulos conflitantes: 13.

Ocorrências de qualidade: `{"answer_without_source_url": 387, "empty_text": 1}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| answers_per_claim | 3068 | 1 | 2.0 | 6.0 | 9.0 | 16 | 2.76 |
| answers_per_question | 7985 | 1 | 1.0 | 2.0 | 2.0 | 3 | 1.06 |
| claim_qa_justification_words | 3068 | 15 | 120.0 | 369.0 | 624.33 | 1221 | 153.61 |
| justification_words | 3068 | 1 | 18.0 | 51.0 | 87.66 | 193 | 21.5 |
| questions_per_claim | 3068 | 1 | 2.0 | 6.0 | 8.0 | 16 | 2.6 |
| text_characters | 3068 | 0 | 93.0 | 203.0 | 286.32 | 392 | 103.37 |
| text_words_whitespace | 3068 | 0 | 15.0 | 33.0 | 46.33 | 73 | 17.03 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 3068 | n/a | 0 |
| `$.cached_original_claim_url` | NoneType: 1343, str: 1188 | 537 | 1386 |
| `$.claim` | str: 3068 | 0 | 1 |
| `$.claim_date` | str: 2963, NoneType: 105 | 0 | 105 |
| `$.claim_types` | list: 3068 | 0 | 0 |
| `$.claim_types[]` | str: 3693 | n/a | 0 |
| `$.fact_checking_article` | str: 3068 | 0 | 0 |
| `$.fact_checking_strategies` | list: 3068 | 0 | 0 |
| `$.fact_checking_strategies[]` | str: 4581 | n/a | 0 |
| `$.justification` | str: 3068 | 0 | 0 |
| `$.label` | str: 3068 | 0 | 0 |
| `$.location_ISO_code` | str: 2732, NoneType: 336 | 0 | 336 |
| `$.original_claim_url` | NoneType: 1565, str: 1503 | 0 | 1611 |
| `$.questions` | list: 3068 | 0 | 0 |
| `$.questions[]` | dict: 7985 | n/a | 0 |
| `$.questions[].answers` | list: 7985 | 0 | 0 |
| `$.questions[].answers[]` | dict: 8479 | n/a | 0 |
| `$.questions[].answers[].answer` | str: 8479 | 0 | 0 |
| `$.questions[].answers[].answer_type` | str: 8479 | 0 | 0 |
| `$.questions[].answers[].boolean_explanation` | str: 1297 | 7182 | 0 |
| `$.questions[].answers[].cached_source_url` | str: 6511 | 1968 | 1 |
| `$.questions[].answers[].source_medium` | str: 8479 | 0 | 387 |
| `$.questions[].answers[].source_url` | str: 8479 | 0 | 387 |
| `$.questions[].question` | str: 7985 | 0 | 1 |
| `$.reporting_source` | str: 2468, NoneType: 600 | 0 | 615 |
| `$.required_reannotation` | bool: 3068 | 0 | 0 |
| `$.speaker` | str: 2402, NoneType: 666 | 0 | 695 |

## averitec / dev

Há desbalanceamento: avaliar pesos/amostragem apenas no treino; não fazer oversampling automaticamente.

Razão majoritária/minoritária: 8.714. Grupos de textos duplicados normalizados: 6; com rótulos conflitantes: 0.

Ocorrências de qualidade: `{"answer_without_source_url": 39}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| answers_per_claim | 500 | 1 | 2.0 | 6.0 | 9.01 | 16 | 2.8 |
| answers_per_question | 1287 | 1 | 1.0 | 2.0 | 3.0 | 4 | 1.09 |
| claim_qa_justification_words | 500 | 23 | 129.5 | 405.2 | 665.52 | 2227 | 168.38 |
| justification_words | 500 | 1 | 17.5 | 55.0 | 80.0 | 164 | 21.63 |
| questions_per_claim | 500 | 1 | 2.0 | 5.05 | 7.01 | 12 | 2.57 |
| text_characters | 500 | 19 | 96.0 | 201.0 | 238.21 | 390 | 105.73 |
| text_words_whitespace | 500 | 3 | 16.0 | 31.0 | 42.01 | 66 | 17.39 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 500 | n/a | 0 |
| `$.cached_original_claim_url` | NoneType: 228, str: 268 | 4 | 238 |
| `$.claim` | str: 500 | 0 | 0 |
| `$.claim_date` | str: 500 | 0 | 0 |
| `$.claim_types` | list: 500 | 0 | 0 |
| `$.claim_types[]` | str: 578 | n/a | 0 |
| `$.fact_checking_article` | str: 500 | 0 | 0 |
| `$.fact_checking_strategies` | list: 500 | 0 | 0 |
| `$.fact_checking_strategies[]` | str: 722 | n/a | 0 |
| `$.justification` | str: 500 | 0 | 0 |
| `$.label` | str: 500 | 0 | 0 |
| `$.location_ISO_code` | NoneType: 50, str: 450 | 0 | 50 |
| `$.original_claim_url` | NoneType: 228, str: 272 | 0 | 238 |
| `$.questions` | list: 500 | 0 | 0 |
| `$.questions[]` | dict: 1287 | n/a | 0 |
| `$.questions[].answers` | list: 1287 | 0 | 0 |
| `$.questions[].answers[]` | dict: 1399 | n/a | 0 |
| `$.questions[].answers[].answer` | str: 1399 | 0 | 0 |
| `$.questions[].answers[].answer_type` | str: 1399 | 0 | 0 |
| `$.questions[].answers[].boolean_explanation` | str: 292 | 1107 | 0 |
| `$.questions[].answers[].cached_source_url` | str: 1326 | 73 | 0 |
| `$.questions[].answers[].source_medium` | str: 1399 | 0 | 39 |
| `$.questions[].answers[].source_url` | str: 1399 | 0 | 39 |
| `$.questions[].question` | str: 1287 | 0 | 0 |
| `$.reporting_source` | str: 412, NoneType: 88 | 0 | 90 |
| `$.required_reannotation` | bool: 500 | 0 | 0 |
| `$.speaker` | NoneType: 110, str: 390 | 0 | 118 |

## feverous / train

Há desbalanceamento: avaliar pesos/amostragem apenas no treino; não fazer oversampling automaticamente.

Razão majoritária/minoritária: 18.668. Grupos de textos duplicados normalizados: 1; com rótulos conflitantes: 0.

Ocorrências de qualidade: `{"header_records": 1}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| elements_per_evidence_set | 77492 | 1 | 3.0 | 13.0 | 32.09 | 254 | 4.51 |
| evidence_sets_per_claim | 71291 | 1 | 1.0 | 2.0 | 2.0 | 3 | 1.09 |
| text_characters | 71291 | 16 | 142.0 | 278.0 | 359.0 | 807 | 153.23 |
| text_words_whitespace | 71291 | 3 | 23.0 | 46.0 | 59.0 | 129 | 25.28 |
| unique_context_ids_per_claim | 71291 | 1 | 3.0 | 8.0 | 14.0 | 90 | 3.94 |
| unique_evidence_ids_per_claim | 71291 | 1 | 3.0 | 14.0 | 34.0 | 254 | 4.85 |
| unique_pages_per_claim | 71291 | 1 | 1.0 | 2.0 | 3.0 | 12 | 1.2 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 71291 | n/a | 0 |
| `$.annotator_operations` | list: 71291 | 0 | 0 |
| `$.annotator_operations[]` | dict: 1134590 | n/a | 0 |
| `$.annotator_operations[].operation` | str: 1134590 | 0 | 0 |
| `$.annotator_operations[].time` | str: 1134590 | 0 | 0 |
| `$.annotator_operations[].value` | str: 1134590 | 0 | 57 |
| `$.challenge` | str: 71291 | 0 | 0 |
| `$.claim` | str: 71291 | 0 | 0 |
| `$.evidence` | list: 71291 | 0 | 0 |
| `$.evidence[]` | dict: 77492 | n/a | 0 |
| `$.evidence[].content` | list: 77492 | 0 | 0 |
| `$.evidence[].content[]` | str: 349556 | n/a | 0 |
| `$.evidence[].context` | dict: 77492 | 0 | 0 |
| `$.evidence[].context.*` | list: 349556 | n/a | 0 |
| `$.evidence[].context.*[]` | str: 983532 | n/a | 0 |
| `$.id` | int: 71291 | 0 | 0 |
| `$.label` | str: 71291 | 0 | 0 |

## feverous / dev

Há desbalanceamento: avaliar pesos/amostragem apenas no treino; não fazer oversampling automaticamente.

Razão majoritária/minoritária: 7.8. Grupos de textos duplicados normalizados: 1; com rótulos conflitantes: 0.

Ocorrências de qualidade: `{"header_records": 1}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| elements_per_evidence_set | 8681 | 1 | 3.0 | 11.0 | 20.0 | 34 | 3.87 |
| evidence_sets_per_claim | 7890 | 1 | 1.0 | 2.0 | 3.0 | 3 | 1.1 |
| text_characters | 7890 | 23 | 140.0 | 268.0 | 364.55 | 656 | 151.23 |
| text_words_whitespace | 7890 | 4 | 23.0 | 44.0 | 60.0 | 107 | 24.96 |
| unique_context_ids_per_claim | 7890 | 1 | 3.0 | 8.0 | 14.0 | 34 | 3.87 |
| unique_evidence_ids_per_claim | 7890 | 1 | 3.0 | 12.0 | 21.0 | 36 | 4.2 |
| unique_pages_per_claim | 7890 | 1 | 1.0 | 2.0 | 3.0 | 7 | 1.2 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 7890 | n/a | 0 |
| `$.annotator_operations` | list: 7890 | 0 | 0 |
| `$.annotator_operations[]` | dict: 119460 | n/a | 0 |
| `$.annotator_operations[].operation` | str: 119460 | 0 | 0 |
| `$.annotator_operations[].time` | str: 119460 | 0 | 0 |
| `$.annotator_operations[].value` | str: 119460 | 0 | 8 |
| `$.challenge` | str: 7890 | 0 | 0 |
| `$.claim` | str: 7890 | 0 | 0 |
| `$.evidence` | list: 7890 | 0 | 0 |
| `$.evidence[]` | dict: 8681 | n/a | 0 |
| `$.evidence[].content` | list: 8681 | 0 | 0 |
| `$.evidence[].content[]` | str: 33581 | n/a | 0 |
| `$.evidence[].context` | dict: 8681 | 0 | 0 |
| `$.evidence[].context.*` | list: 33581 | n/a | 0 |
| `$.evidence[].context.*[]` | str: 92065 | n/a | 0 |
| `$.id` | int: 7890 | 0 | 0 |
| `$.label` | str: 7890 | 0 | 0 |

## feverous / test

Sem rótulos: balanceamento não pode ser medido.

Razão majoritária/minoritária: None. Grupos de textos duplicados normalizados: 0; com rótulos conflitantes: 0.

Ocorrências de qualidade: `{"unlabeled": 7845, "evidence_missing_or_not_list": 7845, "header_records": 1}`.

### Comprimentos e cardinalidades

| Métrica | N | Min | P50 | P95 | P99 | Max | Média |
|---|---:|---:|---:|---:|---:|---:|---:|
| text_characters | 7845 | 24 | 141.0 | 279.8 | 361.68 | 511 | 153.4 |
| text_words_whitespace | 7845 | 3 | 23.0 | 46.0 | 59.0 | 79 | 25.29 |

### Esquema observado

Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.

| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |
|---|---|---:|---:|
| `$` | dict: 7845 | n/a | 0 |
| `$.claim` | str: 7845 | 0 | 0 |

## Sobreposição entre splits

- averitec: train / dev: 4 textos normalizados comuns.
- feverous: train / dev: 0 textos normalizados comuns.
- feverous: train / test: 0 textos normalizados comuns.
- feverous: dev / test: 0 textos normalizados comuns.

Método de duplicação: SHA-256 de NFKC + casefold + espaços normalizados. Isso não detecta paráfrases nem notícias sobre o mesmo evento.

Taxonomias originais foram preservadas. Nenhum split foi reamostrado ou alterado.
