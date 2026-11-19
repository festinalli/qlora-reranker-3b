"""
Gera os vetores (.npy) que o montar_dataset.py pede, SEM modelo de embeddings: só numpy.

    python src/gerar_vetores.py --catalogo exemplos/biblioteca/catalogo.json --saida vetores \
        [--conjunto exemplos/biblioteca/dev.json --conjunto exemplos/biblioteca/teste.json]

ATENÇÃO: no experimento, os vetores vinham de um modelo de embeddings (a mesma busca densa do sistema em
produção). Este script existe só para o exemplo sintético correr de ponta a ponta sem descarregar nada e sem
chamar API nenhuma. É um saco de palavras com hashing: não entende sinônimos e a busca que sai dele é pior que
uma busca densa de verdade. Com dados próprios, use os vetores do seu modelo de embeddings (um vetor por tela, da
ficha; um por pergunta), no mesmo formato.

O que faz, de forma determinística (o mesmo texto dá sempre o mesmo vetor):
  - texto sem acentos e em minúsculas; palavras com 3 letras ou mais, sem as palavras vazias mais comuns;
  - cada palavra e cada trigrama de letras da palavra («#emp», «emp», ...) vai para uma de D posições pelo
    md5 (o hash do Python muda a cada execução, o md5 não); o trigrama pesa menos que a palavra;
  - peso 1 + log(contagem) vezes o IDF calculado no catálogo (fichas + perguntas), e norma 1.

Saída em --saida:
  - fichas.npy: um vetor por tela, na ordem do catálogo (o texto da ficha, o mesmo do BM25 do montador);
  - perguntas.npy: um vetor por pergunta sintética, na ordem do catálogo (tela a tela);
  - <nome>.npy para cada --conjunto (as perguntas dos casos, na ordem do arquivo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from montar_dataset import texto_ficha  # noqa: E402

DIM = 4096
PESO_TRIGRAMA = 0.3
_PALAVRA = re.compile(r"[a-z0-9]+")
VAZIAS = set("""
a o as os um uma uns umas de da do das dos em na no nas nos por para pra pelo pela pelos pelas com sem
que se e ou mas como onde quando qual quais quem porque por que eu ele ela eles elas voce nos meu minha
seu sua isso esse essa este esta aqui ali ja nao sim mais menos muito todo toda todos todas tem ter ser
esta estao foi vai vou quero preciso posso pode faco fazer dar sistema tela
""".split())


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def termos(texto: str) -> list[str]:
    out = []
    for w in _PALAVRA.findall(normalizar(texto)):
        if len(w) < 3 or w in VAZIAS:
            continue
        out.append("w:" + w)
        p = f"#{w}#"
        out.extend("t:" + p[i:i + 3] for i in range(len(p) - 2))
    return out


def posicao(termo: str) -> int:
    return int.from_bytes(hashlib.md5(termo.encode("utf-8")).digest()[:4], "little") % DIM


def ajustar_idf(textos: list[str]) -> dict[str, float]:
    df = Counter(t for x in textos for t in set(termos(x)))
    n = len(textos)
    return {t: math.log(1 + n / (1 + c)) for t, c in df.items()}


def vetores(textos: list[str], idf: dict[str, float]) -> np.ndarray:
    padrao = max(idf.values(), default=1.0)  # termo que o catálogo nunca viu: o peso do termo mais raro
    M = np.zeros((len(textos), DIM), dtype=np.float32)
    for i, x in enumerate(textos):
        for t, c in Counter(termos(x)).items():
            peso = (1 + math.log(c)) * idf.get(t, padrao) * (1.0 if t.startswith("w:") else PESO_TRIGRAMA)
            M[i, posicao(t)] += peso
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    return M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogo", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--conjunto", type=Path, action="append", default=[],
                    help="arquivo de casos (dev, teste) cujas perguntas também precisam de vetor; pode repetir")
    a = ap.parse_args()
    telas = json.loads(a.catalogo.read_text(encoding="utf-8"))["telas"]
    fichas = [texto_ficha(t["ficha"], t.get("descricao")) for t in telas]
    perguntas = [q for t in telas for q in (t.get("perguntas") or [])]
    idf = ajustar_idf(fichas + perguntas)
    a.saida.mkdir(parents=True, exist_ok=True)
    np.save(a.saida / "fichas.npy", vetores(fichas, idf))
    np.save(a.saida / "perguntas.npy", vetores(perguntas, idf))
    contagens = {"fichas": len(fichas), "perguntas": len(perguntas)}
    for p in a.conjunto:
        casos = json.loads(p.read_text(encoding="utf-8"))["casos"]
        np.save(a.saida / f"{p.stem}.npy", vetores([c["pergunta"] for c in casos], idf))
        contagens[p.stem] = len(casos)
    print(json.dumps({"dim": DIM, **contagens}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
