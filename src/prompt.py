"""
O texto que o modelo vê, igual no montador do dataset, no treino e na avaliação.
Mudar aqui muda os três: depois de mexer, montar o dataset outra vez e treinar outra vez.

A tarefa é pontual: uma pergunta e UMA tela. O modelo responde «Sim» ou «Não» e a nota da tela
é P(Sim) no primeiro token da resposta. As 40 candidatas de uma pergunta são ordenadas por essa nota.

Nota da publicação: no experimento original o texto de sistema nomeava o sistema do cliente e os
módulos dele. Aqui ficou um texto neutro, configurável por variável de ambiente. Um adapter treinado
com o texto original deve ser avaliado com o mesmo texto com que foi treinado.
"""
import os

SISTEMA = os.environ.get(
    "RERANKER_SISTEMA",
    "Você conhece o sistema corporativo descrito nos cartões de tela. Responda só «Sim» ou «Não».")

PERGUNTA_FINAL = "Esta é a tela onde o usuário faz exatamente o que a pergunta pede?"

SIM, NAO = "Sim", "Não"

# limites do cartão: cabe em ~250 tokens e mostra o que distingue as telas «gêmeas»
N_BOTOES, N_ACOES, N_CAMPOS = 10, 8, 10


def cartao(rota: str, ficha: dict, descricao: str, menu: str | None, n_casos_qa: int) -> str:
    """O cartão de uma tela: rota, nome, menu, descrição, botões, ações e campos."""
    lst = lambda k, n: " · ".join(map(str, list(ficha.get(k) or [])[:n]))  # noqa: E731
    return (f"Tela: [{rota}] {ficha.get('nome') or ''}\n"
            f"Menu: {menu or 'fora do menu observado'}\n"
            f"Casos de teste do QA: {int(n_casos_qa or 0)}\n"
            f"O que faz: {descricao or ''}\n"
            f"Botões: {lst('botoes', N_BOTOES)}\n"
            f"Ações: {lst('acoes', N_ACOES)}\n"
            f"Campos: {lst('campos', N_CAMPOS)}")


def mensagens(pergunta: str, cartao_tela: str, resposta: str | None = None) -> list[dict]:
    """A conversa no formato de chat (system, user e, no treino, assistant)."""
    msgs = [{"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"Pergunta do usuário: «{pergunta}»\n\n{cartao_tela}\n\n{PERGUNTA_FINAL}"}]
    if resposta is not None:
        msgs.append({"role": "assistant", "content": resposta})
    return msgs
