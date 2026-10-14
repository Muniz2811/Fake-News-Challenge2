# BERTimbau Base + MLP pequena

Avaliação local realizada em 14/10/2026. O treino reutilizou exatamente os
vetores BERTimbau e os splits das duas baselines anteriores, sem vetorizar
notícias novamente. O split de teste foi vetorizado e avaliado apenas depois de
fixar o checkpoint pela validação.

## Protocolo

- Treino: 41.939 notícias; validação: 8.988 notícias, de Fake.br e
  FakeRecogna2. Rótulos `0=true`, `1=fake`.
- Encoder congelado: `neuralmind/bert-base-portuguese-cased`, revisão
  `94d69c95f98f7d5b2a8700c420230ae10def0baa`; vetor de 768 dimensões por
  média dos tokens de conteúdo, com blocos de até 256 tokens.
- Normalização: `StandardScaler` ajustado apenas no treino e aplicado à
  validação. As matrizes e os relatórios das baselines têm hashes dos splits
  conferidos pelo script antes de treinar.
- Classificador: `768 → 128 → 1`, ReLU, dropout `0,2`, saída logit. Perda
  `BCEWithLogitsLoss`; otimizador AdamW com taxa `0,001` e weight decay `0,01`;
  lotes de 256 e limiar `0,5`.
- Sementes: 42, 43 e 44. Até 30 épocas por semente; parada após cinco épocas
  sem melhora do F1 macro na validação (desempate pelo menor Brier score).
  Ambiente: Python 3.14.6, PyTorch 2.14.1, NumPy 2.5.3, CPU.

Execute `python scripts/ml/train_bertimbau_mlp.py` no ambiente com
`requirements-bertimbau.txt` instalado. Os três checkpoints `.pt` ficam em
`models/language_risk/`; as métricas completas, matrizes de confusão e hashes
dos artefatos ficam no relatório local ignorado pelo Git
`data/processed/language_risk/bertimbau/metrics_mlp.json`.

## Resultados na validação

| Métrica | TF-IDF + regressão | BERTimbau + regressão | BERTimbau + MLP (média ± desvio) |
|---|---:|---:|---:|
| F1 macro | 0,9488 | 0,9478 | 0,9720 ± 0,0003 |
| Acurácia | 0,9488 | 0,9478 | 0,9720 ± 0,0003 |
| ROC-AUC | 0,9878 | 0,9876 | 0,9953 ± 0,0002 |
| Brier score (menor é melhor) | 0,0418 | 0,0385 | 0,0226 ± 0,0004 |
| F1 macro — Fake.br | 0,8749 | 0,8861 | 0,9577 ± 0,0049 |
| F1 macro — FakeRecogna2 | 0,9589 | 0,9562 | 0,9740 ± 0,0004 |
| F1 macro — 1 a 50 palavras | 0,8669 | 0,8561 | 0,9218 ± 0,0070 |
| F1 macro — 51 a 150 palavras | 0,9458 | 0,9446 | 0,9682 ± 0,0012 |
| F1 macro — 151 palavras ou mais | 0,8934 | 0,8982 | 0,9536 ± 0,0009 |

F1 macro por semente: `0,971629` (42), `0,972296` (43) e `0,972074` (44).
As melhores épocas foram 9, 13 e 9, respectivamente. A média superou a
regressão nos mesmos embeddings em `0,0242` ponto de F1 macro nesta validação.

## Limites da conclusão

A validação orientou a parada antecipada e a escolha da melhor época; portanto,
essas métricas não são uma estimativa independente de generalização. Os recortes
acima apresentam a média das três sementes, não apenas o melhor checkpoint.

## Avaliação única no teste reservado

O checkpoint da semente **43** foi escolhido por ter o maior F1 macro na
validação (`0,972296`, com desempate previsto pelo Brier score). Seus pesos,
limiar `0,5` e normalização foram mantidos. O teste tem **8.987 notícias** e
foi vetorizado com o mesmo encoder, sem retreino nem escolha baseada nos
resultados abaixo. O avaliador confere os hashes dos splits, dos vetores e do
checkpoint e não sobrescreve um relatório de teste existente.

| Métrica no teste | Resultado |
|---|---:|
| F1 macro | 0,9692 |
| Acurácia | 0,9692 |
| ROC-AUC | 0,9951 |
| Brier score | 0,0240 |
| F1 macro — Fake.br (1.080 notícias) | 0,9574 |
| F1 macro — FakeRecogna2 (7.907 notícias) | 0,9708 |
| F1 macro — 1 a 50 palavras | 0,9074 |
| F1 macro — 51 a 150 palavras | 0,9641 |
| F1 macro — 151 palavras ou mais | 0,9492 |

A matriz de confusão foi `[[4354, 135], [142, 4356]]`, na ordem
`0=true, 1=fake`. O relatório completo permanece local em
`data/processed/language_risk/bertimbau/metrics_mlp_test.json`; os vetores do
teste têm SHA-256
`ef57db8df73eecc97bef1c49b502e9a5322a65fd73fe727f90dfb30629eb28ce`.
O F1 macro do teste ficou cerca de `0,0031` abaixo do checkpoint na validação.
Não foi feita uma comparação no teste com TF-IDF ou BERTimbau + regressão,
pois este passo avaliou apenas a MLP já escolhida.

Antes de substituir a baseline em produção, ainda vale analisar os erros e o
comportamento em notícias recentes de fontes externas. O bot usa TF-IDF por
padrão, mas pode usar esta MLP numa execução local opcional com
`LANGUAGE_RISK_BACKEND=bertimbau_mlp`. O score continua separado da verificação
factual.

Como nas baselines anteriores, o classificador aprende padrões dos rótulos dos
corpora. Seu score não verifica alegações nem determina a veracidade de uma
notícia específica.
