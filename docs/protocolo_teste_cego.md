# Protocolo de teste

O objetivo é que o número final não dependa de escolhas feitas olhando o teste. Os dados do teste e do treino são de um cliente e não estão neste repositório: aqui ficam as regras, os scripts e o formato (ver `exemplos/`, que são **sintéticos**).

## A tarefa medida

- Para cada pergunta, a busca já devolve uma lista **fixa** de 40 telas candidatas.
- Cada sistema reordena as mesmas 40.
- A métrica principal é a **tela certa em 1.º lugar** (top-1). Também se medem top-3, top-5 e MRR.
- Todos os concorrentes reordenam exatamente as mesmas candidatas, o que permite uma comparação **pareada**, caso a caso.

## Separação dev / teste

- **Dev**: é onde se escolhe tudo: o checkpoint, o limiar do «freio» e o desempate entre checkpoints.
- **Teste**: é medido **uma vez**, no fim, com o modelo já escolhido. O `src/avaliar.py` recusa uma segunda medição quando já existe um resultado gravado para aquele teste. Também confere o sha256 do arquivo contra o `CONGELADO.json`.
- **Descontaminação**: o montador do dataset tira do treino as perguntas sintéticas parecidas com perguntas do teste ou do dev (Jaccard de termos ≥ 0,5). Só o treino perde exemplos; o teste não muda.

## Rodada 1: o teste de 168 perguntas

- Esse teste já existia antes do treino: foi gravado no controle de versão em 05/10/2026, e o treino foi em 25/10/2026.
- Ele já tinha sido usado antes para medir as variantes do rerank por API. Ou seja, era cego para o modelo local, mas não era virgem.
- O relatório da época registra que um dos grupos desse teste tinha sido usado antes para afinar a busca. Por isso esse grupo também foi reportado à parte.

## Rodada 2: um teste cego novo, congelado antes do treino

1. **Sorteio** (semente fixa) de casos de QA do cliente que nenhum gabarito antigo usava, com uma tela por caso e um reforço no grupo onde a rodada 1 mais errou.
2. **Escrita**: agentes de LLM escreveram as perguntas lendo só os casos, sem acesso ao treino, aos cartões ou ao modelo.
3. **Conferência**: agentes conferentes, com o catálogo de telas, recusaram perguntas com vazamento do nome da tela ou sem tela certa, e listaram as telas alternativas certas.
4. **Congelamento**: `src/congelar_teste.py` grava o teste, o sha256, a data e o sha256 dos rascunhos, e deixa o arquivo só de leitura. O script recusa congelar duas vezes.
5. **Treino depois**: o montador do dataset da rodada 2 só roda se o teste congelado existir e o sha256 bater.

Na rodada 2, o teste foi congelado em 25/10/2026 às 05:15:39 UTC. O dataset da rodada 2 foi gravado depois, às 05:16 UTC, e o treino terminou às 06:26 UTC.

Depois de medidos, **os dois testes ficam «gastos»** para estes modelos: já se viu onde o modelo erra. Uma rodada nova precisa de outro teste congelado antes do treino.

## Comparação estatística

- Usa-se o **McNemar exato** (binomial bilateral) sobre os casos em que só um dos dois sistemas acerta o top-1.
- «Empate estatístico» quer dizer só isto: **p ≥ 0,05, ou seja, sem diferença significativa** nesse tamanho de amostra. Isso não prova que os dois são equivalentes: não foi feito teste de equivalência (TOST), e com 168 ou 142 casos o intervalo de incerteza é largo.
- Na rodada 1 havia uma régua escrita para dizer «empata com o sistema em produção»: ficar a até 5 acertos e com p ≥ 0,05. O modelo local não passou nessa régua (ver o README).

## O «freio» (medido junto)

Se a melhor nota entre as 40 fica abaixo de um limiar, a resposta seria «não consegui confirmar».

- A tela ausente é simulada: tiram-se as telas certas das 40.
- O limiar é escolhido no dev: o que mais trava telas ausentes perdendo no máximo 7,2% das certas.
