"""
Monta o dataset do reranker local: exemplos (pergunta, cartão de UMA tela) -> «Sim» / «Não».

    python src/montar_dataset.py --catalogo catalogo.json --vetores-fichas fichas.npy \
        --vetores-perguntas perguntas.npy --excluir teste/teste.json --excluir dados/dev_real.json --saida dados

Entrada (formato em exemplos/catalogo_sintetico.json):
  - catalogo.json: {"telas": [{"rota", "ficha": {nome, botoes, acoes, campos, capacidades}, "descricao",
    "perguntas": [~20 perguntas sintéticas], "menu"?: caminho no menu, "n_casos_qa"?: int}]};
  - dois .npy com os vetores já calculados (um por tela, da ficha; um por pergunta, na ordem do catálogo).
    Com eles, as 40 candidatas de cada pergunta saem SEM chamar API nenhuma.

O que faz:
  - Para cada pergunta escolhida, refaz a fusão da busca (denso na ficha, denso nas perguntas sem a própria
    pergunta, BM25; RRF k=60) e fica com as 40 primeiras. As negativas difíceis vêm daí.
  - Separa: treino (K perguntas por tela) e dev sintético (1 outra pergunta por tela).
  - Descontaminação: tira do treino a pergunta sintética quase igual (Jaccard de termos >= 0,5) a uma pergunta
    dos arquivos --excluir (o teste e o dev). O teste NÃO é copiado nem usado para escolher nada.
  - Limpezas: pergunta genérica demais (a tela dela fica abaixo da 10.ª) sai; negativa «gêmea» (uma pergunta
    quase idêntica, cosseno >= 0,92) não vira negativa.

Adaptado do script original: o carregamento do catálogo do cliente e os caminhos do PC viraram parâmetros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import prompt  # noqa: E402

SEMENTE = 3407
K_TREINO = 4          # perguntas de treino por tela (das ~19 que sobram depois da de dev)
N_DUROS = 2           # negativas das 10 primeiras da fusão
N_MEDIOS = 1          # negativas das posições 11 a 40
K_LISTA = 40          # o tamanho da lista de candidatas
K_RRF = 60            # a constante do RRF
LIMIAR_GEMEA = 0.92   # tela negativa com uma pergunta quase igual à pergunta (cosseno) é «gêmea»: não vira negativa
MAX_POS_EXEMPLO = 10  # pergunta cuja tela fica abaixo da 10.ª é genérica demais: as negativas dela seriam ruído
LIMIAR_CONTAMINA = 0.5  # Jaccard de termos entre pergunta sintética e pergunta de teste/dev: acima disto sai
N_DEV_RANK = 400      # perguntas do dev sintético com a lista de 40 (checagem rápida de ordem)

_TOKEN_RX = re.compile(r"[a-zà-ú0-9]+")


def tokens(texto: str) -> list[str]:
    return [w for w in _TOKEN_RX.findall(str(texto or "").lower()) if len(w) > 2]


def texto_ficha(x: dict, descricao: str | None = None) -> str:
    """A ficha numa linha (o texto do BM25)."""
    s = (f"{x.get('nome') or ''} | {x.get('rota_h') or ''} | {x.get('titulo') or ''} | "
         f"botões: {' · '.join(x.get('botoes') or [])} | campos: {' · '.join(x.get('campos') or [])} | "
         f"{' · '.join(x.get('capacidades') or [])} | ações: {' · '.join(x.get('acoes') or [])}")
    return (f"{descricao} | " if descricao else "") + s


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class BM25:
    """BM25 (k1=1,2, b=0,75)."""

    def __init__(self, textos: list[str]):
        self.n = len(textos)
        docs = [Counter(tokens(t)) for t in textos]
        dl = np.array([sum(d.values()) for d in docs], dtype=np.float64)
        avg = float(dl.mean()) or 1.0
        df = Counter(w for d in docs for w in d)
        self.post = {}
        por_termo: dict[str, list] = {}
        for i, d in enumerate(docs):
            for w, f in d.items():
                por_termo.setdefault(w, []).append((i, f))
        for w, lst in por_termo.items():
            idf = math.log(1 + (self.n - df[w] + 0.5) / (df[w] + 0.5))
            ids = np.array([i for i, _ in lst], dtype=np.int32)
            f = np.array([x for _, x in lst], dtype=np.float64)
            self.post[w] = (ids, idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl[ids] / avg)))

    def top(self, texto: str, k: int = K_LISTA) -> list[int]:
        sc = np.zeros(self.n, dtype=np.float64)
        for w in set(tokens(texto)):
            if w in self.post:
                ids, wt = self.post[w]
                sc[ids] += wt
        com = np.flatnonzero(sc > 0)
        return [int(i) for i in com[np.argsort(-sc[com], kind="stable")][:k]]


def rrf(listas: list[list], k: int = K_RRF) -> list:
    s: dict = {}
    for lista in listas:
        for i, x in enumerate(lista):
            s[x] = s.get(x, 0.0) + 1.0 / (k + i + 1)
    return [x for x, _ in sorted(s.items(), key=lambda kv: -kv[1])]


def carregar(catalogo: Path, v_fichas: Path, v_perguntas: Path):
    doc = json.loads(catalogo.read_text(encoding="utf-8"))
    telas = doc["telas"]
    M_f = np.load(v_fichas).astype(np.float32)
    M_q = np.load(v_perguntas).astype(np.float32)
    n_perg = [len(t.get("perguntas") or []) for t in telas]
    if M_f.shape[0] != len(telas) or M_q.shape[0] != sum(n_perg):
        raise SystemExit("o número de vetores não bate com as telas/perguntas do catálogo")
    for m in (M_f, M_q):
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    return telas, M_f, M_q, n_perg


def perguntas_excluidas(arquivos: list[Path]) -> list[tuple[str, str, set]]:
    """(arquivo, id, termos) das perguntas de teste e dev: só para descontaminar."""
    out = []
    for p in arquivos:
        d = json.loads(p.read_text(encoding="utf-8"))
        for c in (d.get("casos") if isinstance(d, dict) else d) or []:
            if isinstance(c, dict) and c.get("pergunta"):
                out.append((p.name, str(c.get("id")), set(tokens(c["pergunta"]))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogo", type=Path, required=True)
    ap.add_argument("--vetores-fichas", type=Path, required=True)
    ap.add_argument("--vetores-perguntas", type=Path, required=True)
    ap.add_argument("--excluir", type=Path, action="append", default=[],
                    help="arquivo de casos (teste, dev) cujas perguntas não podem ter gêmea no treino; pode repetir")
    ap.add_argument("--saida", type=Path, default=AQUI.parent / "dados")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    rng = random.Random(SEMENTE)
    a.saida.mkdir(parents=True, exist_ok=True)

    telas, M_f, M_q, n_perg = carregar(a.catalogo, a.vetores_fichas, a.vetores_perguntas)
    rotas = [t["rota"] for t in telas]
    inicio = np.cumsum([0] + n_perg[:-1])
    dono = np.repeat(np.arange(len(telas)), n_perg)
    perguntas = [q for t in telas for q in (t.get("perguntas") or [])]
    cartoes = {t["rota"]: prompt.cartao(t["rota"], t["ficha"], t.get("descricao") or "", t.get("menu"),
                                        t.get("n_casos_qa") or 0) for t in telas}
    bm = BM25([texto_ficha(t["ficha"], t.get("descricao")) for t in telas])

    # ---- quem vai para onde: 1 pergunta de dev e K de treino por tela (perguntas diferentes)
    papel = {}
    for s in range(len(telas)):
        if n_perg[s] < 2:
            continue
        idx = list(range(inicio[s], inicio[s] + n_perg[s]))
        rng.shuffle(idx)
        papel[idx[0]] = "dev"
        for j in idx[1:1 + K_TREINO]:
            papel[j] = "treino"

    # ---- descontaminação contra o teste e o dev
    excl = perguntas_excluidas(a.excluir)
    contaminadas = 0
    for j in list(papel):
        q = set(tokens(perguntas[j]))
        if len(q) < 3:
            continue
        if any(len(q & d) / len(q | d) >= LIMIAR_CONTAMINA for _, _, d in excl):
            contaminadas += 1
            papel.pop(j)

    # ---- fusão local (denso ficha, denso perguntas sem a própria, BM25)
    sel = sorted(papel)
    com = np.flatnonzero(np.array(n_perg) > 0)
    ini_com = inicio[com]
    fusao, melhor_por = {}, {}
    for a0 in range(0, len(sel), 256):
        bloco = sel[a0:a0 + 256]
        Q = M_q[bloco]
        SF = Q @ M_f.T
        SQ = Q @ M_q.T
        for r, j in enumerate(bloco):
            SQ[r, j] = -2.0                # a própria pergunta não conta (no teste a pergunta não está no índice)
        mel = np.full((len(bloco), len(telas)), -1.0, dtype=np.float32)
        mel[:, com] = np.maximum.reduceat(SQ, ini_com, axis=1)
        for r, j in enumerate(bloco):
            lf = [int(i) for i in np.argsort(-SF[r], kind="stable")[:K_LISTA]]
            lq = [int(i) for i in np.argsort(-mel[r], kind="stable")[:K_LISTA]]
            fusao[j] = rrf([lf, lq, bm.top(perguntas[j])])[:K_LISTA]
            melhor_por[j] = mel[r]

    # ---- exemplos
    treino, dev_ex, dev_rank = [], [], []
    fora_40, genericas, gemeas = Counter(), Counter(), 0
    for j in sel:
        s = int(dono[j])
        cand = fusao[j]
        if s not in cand:
            fora_40[papel[j]] += 1
            continue
        p = cand.index(s) + 1
        q = perguntas[j]
        if papel[j] == "dev":
            dev_rank.append({"id": f"S{j}", "grupo": "dev_sintetico", "pergunta": q, "certas": [rotas[s]],
                             "candidatas": [rotas[i] for i in cand]})
        if p > MAX_POS_EXEMPLO:
            genericas[papel[j]] += 1
            continue
        mel = melhor_por[j]
        negs = [i for i in cand if i != s]
        limpas = [i for i in negs if mel[i] < LIMIAR_GEMEA]
        gemeas += len(negs) - len(limpas)
        duros = [i for i in limpas if cand.index(i) < 10]
        medios = [i for i in limpas if cand.index(i) >= 10]
        escolha = rng.sample(duros, min(N_DUROS, len(duros))) + rng.sample(medios, min(N_MEDIOS, len(medios)))
        linhas = [{"messages": prompt.mensagens(q, cartoes[rotas[s]], prompt.SIM),
                   "meta": {"tela": rotas[s], "rotulo": 1, "pos_fusao": p}}]
        for i in escolha:
            linhas.append({"messages": prompt.mensagens(q, cartoes[rotas[i]], prompt.NAO),
                           "meta": {"tela": rotas[i], "rotulo": 0, "pos_fusao": cand.index(i) + 1}})
        (treino if papel[j] == "treino" else dev_ex).extend(linhas)
    rng.shuffle(treino)
    rng.shuffle(dev_rank)
    dev_rank = dev_rank[:N_DEV_RANK]

    def jsonl(nome, linhas):
        with open(a.saida / nome, "w", encoding="utf-8", newline="\n") as f:
            for x in linhas:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    jsonl("treino.jsonl", treino)
    jsonl("dev_sintetico.jsonl", dev_ex)
    (a.saida / "dev_sintetico_rank.json").write_text(json.dumps({"casos": dev_rank}, ensure_ascii=False, indent=1),
                                                     encoding="utf-8", newline="\n")
    (a.saida / "telas.json").write_text(json.dumps(cartoes, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    manifesto = {
        "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parametros": {"semente": SEMENTE, "k_treino_por_tela": K_TREINO, "n_duros": N_DUROS, "n_medios": N_MEDIOS,
                       "limiar_gemea": LIMIAR_GEMEA, "limiar_contamina_jaccard": LIMIAR_CONTAMINA,
                       "max_pos_exemplo": MAX_POS_EXEMPLO},
        "fontes": {"catalogo": sha256(a.catalogo), "excluidos": {p.name: sha256(p) for p in a.excluir}},
        "contagens": {"telas": len(telas), "perguntas_sinteticas": len(perguntas), "treino_exemplos": len(treino),
                      "dev_sintetico_exemplos": len(dev_ex), "dev_sintetico_rank": len(dev_rank),
                      "fora_das_40": dict(fora_40), "genericas_descartadas": dict(genericas),
                      "negativas_gemeas_puladas": gemeas, "retiradas_por_contaminacao": contaminadas},
    }
    (a.saida / "MANIFESTO.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1), encoding="utf-8",
                                            newline="\n")
    print(json.dumps(manifesto["contagens"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
