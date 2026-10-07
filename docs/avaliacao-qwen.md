# Avaliação local do Qwen

## Checkpoint

- Modelo: `qwen3:8b`
- Runtime avaliado: Ollama `0.32.5`, macOS arm64
- Temperatura: `0`
- Thinking: desativado
- Formato: JSON Schema pela API compatível com OpenAI

O modelo não é versionado nem baixado pelo CI ou pela imagem Docker.

## Como reproduzir

Com o Ollama aberto e `qwen3:8b` já baixado:

```bash
LLM_BASE_URL=http://localhost:11434/v1 \
LLM_API_KEY=ollama \
LLM_MODEL=qwen3:8b \
python scripts/evaluate_qwen.py
```

O script executa um conjunto pequeno de smoke tests end-to-end:

1. extrai uma alegação e descarta opinião, pergunta e pedido de compartilhamento;
2. compara apoio, refutação, irrelevância e conflito com trechos textuais;
3. percorre três perguntas e produz a síntese do diálogo.

O comando retorna código `0` somente quando os três blocos atendem ao esperado.
Esta avaliação pequena não mede acurácia geral e não substitui uma base anotada.

## Resultado registrado

Execução de 7 de outubro de 2026, no checkpoint `qwen3:8b` Q4_K_M:

| Bloco | Resultado |
|---|---|
| Extração | passou; uma alegação factual, sem opinião ou pedido de compartilhamento |
| NLI — apoio | `apoiada` |
| NLI — contradição | `refutada` |
| NLI — trecho sem relação | `evidencia_insuficiente` |
| NLI — fontes divergentes | `conflitante` |
| Diálogo | passou; três perguntas, segunda adaptada e síntese final |
| Fallback acionado | não |
| Duração total da avaliação inicial | 207,16 segundos para 8 chamadas estruturadas |

Essa avaliação inicial revelou que pedir trechos integrais e uma categoria
global tornava a saída desnecessariamente grande. O contrato foi reduzido para
`id + relação`; justificativa, fontes e categoria passaram a ser produzidas em
Python.

Na regressão com uma busca real sobre as Eleições Gerais de 2026, foram
encontradas 8 evidências; o fluxo foi então ajustado para enviar no máximo as 2
mais bem ranqueadas ao NLI, excluindo redes sociais, vídeo e Wikipédia. A
chamada produziu a categoria `apoiada`. O processamento do modelo registrado pelo Ollama levou
aproximadamente 16,4 segundos; o tempo HTTP observado foi maior porque uma
requisição antiga ainda ocupava a fila.

A latência observada é alta para conversa interativa. Antes de produção, vale
comparar um checkpoint menor ou reduzir chamadas, sem remover as validações e a
abstenção. Resultados podem variar quando checkpoint, quantizador, versão do
Ollama ou hardware mudarem.
