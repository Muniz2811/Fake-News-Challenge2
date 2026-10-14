# Baseline BERTimbau Base + regressão logística

Avaliação local realizada em 14/10/2026 com o mesmo treino e a mesma validação
da baseline TF-IDF. O conjunto de teste não foi usado.

## Protocolo

- Dados: Fake.br e FakeRecogna2, com rótulos `0=true` e `1=fake`.
- Treino: 41.939 notícias; validação: 8.988 notícias.
- Encoder congelado: `neuralmind/bert-base-portuguese-cased`, revisão
  `94d69c95f98f7d5b2a8700c420230ae10def0baa`.
- Pesos: SHA-256
  `cb1693767adef60abf23d9fde3996f0c1e6310afad103a2db94ad44854568955`.
- Vetorização: blocos sem sobreposição de até 256 tokens, com `[CLS]` e `[SEP]`;
  média de todos os tokens de conteúdo de cada notícia. Vetores `float32` de
  768 dimensões. Nenhuma parte de uma notícia longa é descartada.
- Classificador: `StandardScaler` e `LogisticRegression` com `C=1`, solver
  `liblinear`, `max_iter=1000`, limiar `0,5` e semente `42`.
- Ambiente desta execução: Python 3.14.6, PyTorch 2.14.1, Transformers 4.57.6,
  macOS com GPU Metal. O classificador convergiu em 11 iterações.

Os hashes dos splits foram conferidos antes da comparação. Os scripts, comandos
de instalação e caminhos dos artefatos estão no [README](../README.md).

## Resultados na validação

| Métrica | TF-IDF + regressão | BERTimbau + regressão |
|---|---:|---:|
| F1 macro | 0,9488 | 0,9478 |
| Acurácia | 0,9488 | 0,9478 |
| ROC-AUC | 0,9878 | 0,9876 |
| Brier score (menor é melhor) | 0,0418 | 0,0385 |
| F1 macro — Fake.br | 0,8749 | 0,8861 |
| F1 macro — FakeRecogna2 | 0,9589 | 0,9562 |
| F1 macro — textos de 1 a 50 palavras | 0,8669 | 0,8561 |
| F1 macro — textos de 51 a 150 palavras | 0,9458 | 0,9446 |
| F1 macro — textos com 151 palavras ou mais | 0,8934 | 0,8982 |

A diferença geral de F1 macro foi `−0,0010` para o BERTimbau. Isso não sustenta
uma conclusão de superioridade de nenhuma das duas baselines. O BERTimbau teve
melhor F1 no Fake.br e em textos longos; a TF-IDF teve melhor F1 no
FakeRecogna2 e em textos curtos. O Brier score menor sugere probabilidades
melhores nesta validação, mas isso requer avaliação adicional de calibração.

As matrizes de confusão e métricas completas estão no arquivo local ignorado
pelo Git `data/processed/language_risk/bertimbau/metrics.json`. O checkpoint,
os vetores e o classificador também permanecem apenas no computador. Os vetores
têm SHA-256 `e15d48dd03a369513479f97039743e3599a3d285fdafb3c7e79e495bb057d7f5`
(treino) e `4f86c57c3dd0ebfc457852873e04a1f9d84686f62ce32dec5da952c9130457ae`
(validação). O classificador salvo tem SHA-256
`691c85c50f7180bdd2e202d9d6ad1052919f0f716b07777682f9e53492f3092e`.

## Interpretação

Esses modelos estimam associação aos rótulos das bases de treino. Eles não
verificam alegações nem determinam se uma notícia específica é verdadeira. A
diferença entre bases e tamanhos de texto também recomenda cautela ao aplicar
os scores a notícias recentes ou de outra origem. O bot usa TF-IDF por padrão;
esta baseline BERTimbau pode ser testada localmente com
`LANGUAGE_RISK_BACKEND=bertimbau_logreg`, sem alterar a verificação factual.
