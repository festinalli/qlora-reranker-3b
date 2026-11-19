"""
Avaliação do reranker local sobre listas FIXAS de 40 candidatas por pergunta: a tela certa em 1.º, nas 3 e
nas 5 primeiras, MRR, por grupo. Compara caso a caso com as ordens já medidas de outros sistemas (por exemplo
a busca sem LLM e o rerank de um modelo de API), com o teste de McNemar exato no top-1.

    python src/avaliar.py --conjunto dados/dev_real.json --base                       # o modelo sem treino
    python src/avaliar.py --conjunto dados/dev_real.json --adapter saida/adapter      # o modelo treinado
    python src/avaliar.py --conjunto teste/teste.json --congelado teste/CONGELADO.json \
        --adapter saida/adapter --uma-vez --tau 0.4 [--tambem-base]

Formato do conjunto (ver exemplos/): {"casos": [{"id", "grupo", "pergunta", "certas": [rota, ...],
"candidatas": [40 rotas], "tipo"?: "especifica"|"vaga", "fraco"?: bool, "base"?: {"<sistema>": posição da certa}}]}
Os cartões das telas vêm de --telas (um JSON {rota: cartão}, feito por montar_dataset.py).

Teste congelado: com --congelado, o sha256 do arquivo tem de bater com o do CONGELADO.json; com --uma-vez, o script
recusa se já houver um resultado gravado para aquele teste. O dev é onde se escolhe tudo (checkpoint, limiar do freio).

A nota de cada tela é P(Sim) no 1.º token da resposta; empate = a ordem original das candidatas.
O «freio»: se a melhor nota fica abaixo de um limiar, a resposta seria «não consegui confirmar»; a tela ausente é
simulada tirando as telas certas das 40 (as notas são as mesmas, não custa outra passada).

Adaptado do script original: os caminhos do PC e o carregamento específico do cliente viraram parâmetros.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=FutureWarning)

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import prompt  # noqa: E402

MODELO_PADRAO = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
MAX_SEQ = 768


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pos_certa(ordem, certas):
    return next((j for j, r in enumerate(ordem, 1) if r in certas), None)


def carregar_conjunto(arquivo, congelado):
    if congelado:
        cong = json.load(open(congelado, encoding="utf-8"))
        if sha256(arquivo) != cong["sha256"]:
            raise SystemExit(f"{arquivo}: sha256 diferente do {congelado} — o teste mudou, não meço")
    d = json.load(open(arquivo, encoding="utf-8"))
    return [c for c in d["casos"] if c.get("candidatas")]


class Juiz:
    """O modelo (base ou base + adapter) e a nota P(Sim) de uma lista de (pergunta, cartão)."""

    def __init__(self, nome_ou_adapter, lote):
        from unsloth import FastLanguageModel
        import torch
        self.torch = torch
        self.model, self.tok = FastLanguageModel.from_pretrained(
            model_name=nome_ou_adapter, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(self.model)
        self.tok.padding_side = "left"
        self.lote = lote
        self.id_sim = self.tok.encode(prompt.SIM, add_special_tokens=False)[0]
        self.id_nao = self.tok.encode(prompt.NAO, add_special_tokens=False)[0]
        if self.id_sim == self.id_nao:
            raise SystemExit("«Sim» e «Não» começam pelo mesmo token: mudar prompt.SIM/NAO")

    def notas(self, pares):
        torch = self.torch
        textos = [self.tok.apply_chat_template(prompt.mensagens(q, c), tokenize=False, add_generation_prompt=True)
                  for q, c in pares]
        out = []
        for a in range(0, len(textos), self.lote):
            enc = self.tok(textos[a:a + self.lote], return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_SEQ).to("cuda")
            with torch.no_grad():
                try:
                    logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**enc).logits[:, -1, :]
            dois = logits[:, [self.id_sim, self.id_nao]].float()
            out.extend(torch.softmax(dois, dim=-1)[:, 0].tolist())
        return out


def mcnemar(a_certo, b_certo):
    """McNemar exato (binomial bilateral) no top-1: casos em que só um dos dois acerta."""
    b = sum(1 for x, y in zip(a_certo, b_certo) if x and not y)
    c = sum(1 for x, y in zip(a_certo, b_certo) if y and not x)
    n = b + c
    if n == 0:
        return {"so_local_acerta": 0, "so_base_acerta": 0, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"so_local_acerta": b, "so_base_acerta": c, "p": round(p, 4)}


def medir(casos, ordens):
    por = defaultdict(Counter)
    for c in casos:
        p = pos_certa(ordens[c["id"]], c["certas"])
        grupos = [c.get("grupo") or "sem_grupo", "TOTAL"]
        if c.get("tipo"):
            grupos += ["VAGAS" if c["tipo"] == "vaga" else "ESPECIFICAS"]
            if c["tipo"] != "vaga" and "fraco" in c:
                grupos += ["ESPECIFICAS_FRACAS" if c["fraco"] else "ESPECIFICAS_FORTES"]
        for g in grupos:
            por[g]["n"] += 1
            por[g]["top1"] += p == 1
            por[g]["top3"] += bool(p and p <= 3)
            por[g]["top5"] += bool(p and p <= 5)
            por[g]["rr"] += (1 / p) if p else 0
    return {g: {**{k: v for k, v in d.items() if k != "rr"}, "mrr": round(d["rr"] / d["n"], 3)} for g, d in por.items()}


def freio(casos, notas, tau=None, perda_max=0.072):
    """A tela ausente simulada: sem as certas, a melhor nota é a de uma tela errada. Com o limiar tau, quantas
    ausentes travam (bom) e quantas certas em 1.º travam (perda). No dev, escolhe o limiar que mais trava
    ausentes perdendo no máximo perda_max das certas."""
    pares = []
    for c in casos:
        n = notas[c["id"]]
        certa = max((v for r, v in n.items() if r in c["certas"]), default=None)
        outra = max((v for r, v in n.items() if r not in c["certas"]), default=0.0)
        if certa is None:
            continue
        pares.append((certa, outra))
    acertos = [max(a, b) for a, b in pares if a >= b]
    ausentes = [b for a, b in pares]

    def em(t):
        return {"tau": t, "ausentes_travadas": sum(x < t for x in ausentes), "ausentes": len(ausentes),
                "certas_em_1o_perdidas": sum(x < t for x in acertos), "certas_em_1o": len(acertos)}
    if tau is not None:
        return {"escolhido": em(tau)}
    grade = [em(t) for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
    ok = [g for g in grade if g["certas_em_1o_perdidas"] <= perda_max * max(1, g["certas_em_1o"])]
    return {"grade": grade, "escolhido_no_dev": max(ok, key=lambda g: g["ausentes_travadas"]) if ok else None}


def avaliar(juiz, casos, telas):
    notas, ordens = {}, {}
    t0 = time.time()
    for i, c in enumerate(casos, 1):
        faltam = [r for r in c["candidatas"] if r not in telas]
        if faltam:
            raise SystemExit(f"{c['id']}: candidatas sem cartão: {faltam[:3]}")
        sc = juiz.notas([(c["pergunta"], telas[r]) for r in c["candidatas"]])
        k = c["id"]
        notas[k] = dict(zip(c["candidatas"], sc))
        p = {r: j for j, r in enumerate(c["candidatas"])}
        ordens[k] = sorted(c["candidatas"], key=lambda r: (-notas[k][r], p[r]))
        if i % 20 == 0:
            print(f"   {i}/{len(casos)} ({(time.time() - t0) / i:.2f} s/pergunta)", flush=True)
    return notas, ordens, (time.time() - t0) / max(1, len(casos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conjunto", required=True, help="JSON com os casos (dev ou teste)")
    ap.add_argument("--telas", default=os.path.join(RAIZ, "dados", "telas.json"), help="JSON {rota: cartão}")
    ap.add_argument("--congelado", help="CONGELADO.json do teste: confere o sha256 antes de medir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter", action="append", help="pasta do adapter LoRA; pode repetir")
    g.add_argument("--base", action="store_true", help="o modelo base, sem treino (zero-shot)")
    ap.add_argument("--modelo", default=os.environ.get("MODELO", MODELO_PADRAO), help="o modelo base (com --base)")
    ap.add_argument("--tambem-base", action="store_true", help="no teste: medir também o base, na mesma corrida")
    ap.add_argument("--uma-vez", action="store_true", help="teste congelado: recusa se já houver resultado")
    ap.add_argument("--tau", type=float, action="append", help="limiar do freio escolhido no dev, um por --adapter")
    ap.add_argument("--lote", type=int, default=20)
    ap.add_argument("--limite", type=int, default=0, help="só os N primeiros casos (fumaça; não vale no teste)")
    ap.add_argument("--resultados", default=os.path.join(RAIZ, "resultados"))
    a = ap.parse_args()

    os.makedirs(a.resultados, exist_ok=True)
    rot_conj = os.path.splitext(os.path.basename(a.conjunto))[0]
    if a.congelado:
        if not a.uma_vez or a.base or a.limite:
            raise SystemExit("teste congelado: só com --adapter do modelo escolhido no dev, --uma-vez e sem --limite")
        ja = glob.glob(os.path.join(a.resultados, f"{rot_conj}_*.json"))
        if ja:
            raise SystemExit(f"o {rot_conj} já foi medido ({ja[0]}): mede-se uma vez só")
        if not a.tau or len(a.tau) != len(a.adapter):
            raise SystemExit("teste: passar um --tau por --adapter (o limiar do freio escolhido no dev)")

    casos = carregar_conjunto(a.conjunto, a.congelado)
    if a.limite:
        casos = casos[:a.limite]
    telas = json.load(open(a.telas, encoding="utf-8"))
    print(f">> {rot_conj}: {len(casos)} perguntas, {sum(len(c['candidatas']) for c in casos)} pares", flush=True)

    taus = list(a.tau or [])
    if a.adapter:
        modelos = [(os.path.basename(os.path.normpath(x)), x, (taus[i] if i < len(taus) else None))
                   for i, x in enumerate(a.adapter)]
    else:
        modelos = [("base_sem_treino", a.modelo, None)]
    if a.congelado and a.tambem_base:
        modelos.append(("base_sem_treino", a.modelo, taus[0] if taus else None))

    import torch
    saida = {"conjunto": rot_conj, "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "n": len(casos), "sha256_conjunto": sha256(a.conjunto), "modelos": {}}
    rot = "base" if a.base else os.path.basename(os.path.normpath(a.adapter[0]))
    nome = f"{rot_conj}_{rot}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if a.limite:
        nome = "fumaca_" + nome
    acertos_por_modelo = {}
    for rotulo, caminho, tau in modelos:
        print(f">> Modelo «{rotulo}»: {caminho}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        juiz = Juiz(caminho, a.lote)
        notas, ordens, seg = avaliar(juiz, casos, telas)
        res = {"caminho": caminho, "metricas": medir(casos, ordens), "seg_por_pergunta": round(seg, 3),
               "pico_vram_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
               "freio": freio(casos, notas, tau)}
        certo_local = [ordens[c["id"]][0] in c["certas"] for c in casos]
        acertos_por_modelo[rotulo] = certo_local
        # caso a caso contra as ordens já medidas de outros sistemas (campo «base» de cada caso)
        comp = {}
        sistemas = sorted({b for c in casos for b in (c.get("base") or {})})
        for b in sistemas:
            idx = [i for i, c in enumerate(casos) if b in (c.get("base") or {})]
            certo_b = [casos[i]["base"][b] == 1 for i in idx]
            comp[b] = {"top1_base": sum(certo_b), "top1_local": sum(certo_local[i] for i in idx), "n": len(idx),
                       **mcnemar([certo_local[i] for i in idx], certo_b)}
        res["comparacao_top1"] = comp
        res["casos"] = [{"id": c["id"], "grupo": c.get("grupo"), "certas": c["certas"],
                         "pos_local": pos_certa(ordens[c["id"]], c["certas"]), "base": c.get("base"),
                         "top5": [(r, round(notas[c["id"]][r], 4)) for r in ordens[c["id"]][:5]]}
                        for c in casos]
        saida["modelos"][rotulo] = res
        t = res["metricas"]["TOTAL"]
        print(f">> «{rotulo}» TOTAL: top1 {t['top1']}/{t['n']} · top3 {t['top3']} · top5 {t['top5']} · "
              f"MRR {t['mrr']} · {res['seg_por_pergunta']} s/pergunta · VRAM {res['pico_vram_gb']} GB", flush=True)
        for b, x in comp.items():
            print(f"   vs {b}: base {x['top1_base']} · local {x['top1_local']} (de {x['n']}) · "
                  f"só local {x['so_local_acerta']} · só base {x['so_base_acerta']} · McNemar p={x['p']}", flush=True)
        rot_ = list(acertos_por_modelo)
        saida["entre_modelos"] = {f"{x} vs {y}": {"top1": [sum(acertos_por_modelo[x]), sum(acertos_por_modelo[y])],
                                                  **mcnemar(acertos_por_modelo[x], acertos_por_modelo[y])}
                                  for i, x in enumerate(rot_) for y in rot_[i + 1:]}
        # grava a cada modelo: se o 2.º falhar (memória), o 1.º já está registrado e o teste conta como medido
        with open(os.path.join(a.resultados, nome), "w", encoding="utf-8") as f:
            json.dump(saida, f, ensure_ascii=False, indent=1)
        print(">> Gravado:", os.path.join(a.resultados, nome), flush=True)
        del juiz
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
