# MVP-10 — verificação individual

O motor mantém Google Fact Check e Tavily, a interface `WebSearchProvider` e a
injeção de provedores da main. Uma falha de provedor não interrompe os demais.
Cada `VereditoAlegacao` liga ID/texto da alegação, categoria, justificativa,
IDs das evidências efetivamente usadas e URLs. `resultado_factual` contém as
conclusões por alegação e é consumido pelo relatório existente do bot.

## Política de verificação

- Sem evidências: `evidencia_insuficiente`. Comparação indisponível ou saída
  inválida: `nao_verificada`, com justificativa de abstenção.
- Fact-checks preservam `checked_claim`, título, `textual_rating`, trecho, URL,
  publicador e data. Texto original ausente não é substituído pela consulta.
- A comparação determinística exige igualdade do texto original, normalizando
  somente Unicode, caixa, espaços e pontuação terminal. Preserva negações,
  números, datas e qualificadores. Só rótulos completos explicitamente conhecidos
  permitem apoio/refutação. Rótulos ambíguos não sustentam decisão afirmativa.
- Fact-checks de outra alegação ou sem texto original são excluídos da comparação.
  Essa política também é conservadora com páginas de checagem recuperadas pelo
  Tavily que não tragam metadados da alegação original.
- Para outros trechos, a LLM devolve relações por evidência. Python exige todos
  os IDs elegíveis, trechos integrais exatos, tipos e categorias válidos, conclusão
  compatível, justificativa não vazia e resposta completa. Campos adicionais,
  URLs inventadas, referências duplicadas e omissões causam abstenção.
- Apoio e refutação juntos produzem `conflitante`. Conflitos de fact-checks
  comparáveis são preservados antes de consultar a LLM.
- A justificativa publicada é construída em Python com os trechos, relações e
  URLs validados; a narrativa livre do modelo não é publicada.
- IDs por conteúdo, deduplicação, ordenação e fundamentação padronizada tornam
  a entrada independente da ordem. Um cache por instância repete o resultado
  para a mesma alegação e conjunto de evidências, inclusive após reordenação.

## Validação e limites

Execute `python -m unittest discover -s tests -v`. A suíte existente e as
regressões simulam dependências/chamadas externas, sem carregar `.env` ou usar
chaves reais. O teste do relatório importa o consumidor real com Telegram
simulado. Polling e serviços reais não são exercitados.

A validação de IDs/trechos não prova a interpretação semântica da LLM. Não há
garantia de decisões idênticas após reiniciar o processo ou mudar o modelo;
o cache é local à instância. Fact-checks para paráfrases são conservadoramente
descartados. Não há consulta adicional às URLs para provar seu conteúdo.

## Pendências para outra issue

- Integrar a política de `requer_revisao` ao alerta e ao deep link (MVP-04),
  considerando risco linguístico e categorias factuais. O valor atual permanece
  como na main; este PR não implementa a baseline MVP-07.
- Melhorar apresentação do Telegram: escape de Markdown, divisão de relatórios
  longos, exibição de títulos/datas e mensagens de ajuda atualizadas.
- Avaliar execução não bloqueante das chamadas de extração, busca e verificação
  no fluxo assíncrono, com limites de concorrência e política de tentativas.

Estas pendências são registradas aqui para abertura futura; nenhuma nova issue
foi criada. A API pública do GitHub retornou 404 para o PR e issues relacionadas;
a revisão usou as referências Git remotas e os critérios fornecidos pelo usuário.
