# Ponto de equilíbrio contra a API

A pergunta: a partir de que volume vale a pena servir o modelo local numa GPU dedicada, em vez de pagar o rerank à API?

## A conta

| Item | Valor | De onde |
|---|---|---|
| GPU dedicada na nuvem (uma T4, 24x7) | US$ 0,894/h × 730 h = **US$ 652,6 por mês** | API pública de preços do provedor de nuvem, consultada em 25/10/2026 (região Brasil, sob demanda) |
| Rerank pela API (gpt-4.1-mini, 2 chamadas por pergunta) | **US$ 2,80 por mil perguntas** | custo medido em produção em 25/10/2026, pela tabela de preços da API |
| **Ponto de equilíbrio** | 652,6 ÷ (2,80 / 1000) ≈ **233 mil perguntas por mês** | `python src/break_even.py` |

Para comparar, a medição única do teste cego da rodada 2 gastou 426 chamadas, 842.830 tokens de entrada e 23.858 de saída, ~US$ 0,374 a preço de tabela. Isso dá **~US$ 2,6 por mil perguntas**, perto dos 2,80.

Outras opções que o mesmo registro anotou, sem refazer a conta para elas:
- a mesma T4 em instância spot: US$ 203,6 por mês;
- a mesma T4 em outra região dos EUA: US$ 384,0 por mês.

## O que a conta não inclui

- **O modelo local também tem custo.** A conta só põe a máquina contra a API. Ficam de fora operação, monitoramento, redundância (uma GPU só é um ponto único de falha), energia, as horas de quem mantém, e o custo de treinar de novo e de montar um teste cego novo a cada rodada.
- **A latência e a vazão na T4 não foram medidas.** O tempo medido é o do notebook: 3,6 a 3,7 s por pergunta, com lote de 40 candidatas, numa GPU de notebook de 8 GB.
- **Só vale para esta tarefa,** reordenar 40 candidatas por pergunta. Não é uma comparação geral entre um modelo local e a API.
- **Os preços mudam.** Antes de qualquer decisão, refaça a conta com os preços do dia (`--usd-hora`, `--usd-por-mil`).
- Abaixo de ~233 mil perguntas por mês, a API sai mais barata do que deixar uma GPU ligada só para isso. A troca também pode se justificar por outro motivo: o dado não sair da empresa, não haver limite de pedidos, ou a resposta ser sempre a mesma. Mas isso é outra decisão, que não vem desta conta.
