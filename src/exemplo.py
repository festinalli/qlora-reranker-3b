"""
O exemplo de ponta a ponta, com dados SINTÉTICOS (exemplos/biblioteca/): um comando só.

    python src/exemplo.py

Etapas, na ordem:
  1. vetores     gerar_vetores.py: os .npy das fichas, das perguntas do catálogo, do dev e do teste
                 (saco de palavras com hashing, só numpy: no experimento vinham de um modelo de embeddings);
  2. candidatas  as 40 candidatas de cada pergunta do dev e do teste, com a mesma fusão do montador
                 (denso na ficha, denso nas perguntas do catálogo, BM25; RRF k=60). A posição da certa nessa
                 ordem fica no caso como o sistema «busca_sem_llm». O teste é congelado aqui (sha256);
  3. dataset     montar_dataset.py, descontaminado contra o dev e o teste;
  4. treino      treinar.py, 1 época (com o catálogo do exemplo dá ~55 passos), adapter gravado;
  5. dev         avaliar.py no dev: o modelo base e o treinado. O limiar do freio sai daqui;
  6. teste       avaliar.py no teste congelado, uma vez: o treinado e o base na mesma corrida.

Tudo vai para exemplos/biblioteca/saida/ (ignorada pelo git) ou para a pasta de EXEMPLO_SAIDA. O teste mede-se
uma vez: se a pasta de saída já existe, o script recusa. Para repetir o exemplo, apague a pasta de saída.

Os números que saem daqui NÃO dizem nada sobre o experimento: o catálogo é inventado e pequeno, os vetores são um
saco de palavras e o treino é curto. O exemplo só mostra que o processo corre de ponta a ponta.
"""
import glob
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import montar_dataset as md  # noqa: E402
from avaliar import mcnemar  # noqa: E402

EX = os.path.join(RAIZ, "exemplos", "biblioteca")
SAIDA = os.path.abspath(os.environ.get("EXEMPLO_SAIDA") or os.path.join(EX, "saida"))
CATALOGO = os.path.join(EX, "catalogo.json")
DEV_ORIG = os.path.join(EX, "dev.json")
TESTE_ORIG = os.path.join(EX, "teste.json")
VET = os.path.join(SAIDA, "vetores")
DADOS = os.path.join(SAIDA, "dados")
RES = os.path.join(SAIDA, "resultados")
DEV = os.path.join(SAIDA, "dev.json")
TESTE_DIR = os.path.join(SAIDA, "teste")
TESTE = os.path.join(TESTE_DIR, "teste.json")
CONGELADO = os.path.join(TESTE_DIR, "CONGELADO.json")
ADAPTER = os.path.join(SAIDA, "adapter")
TAU_SEM_ESCOLHA = 0.5   # só se nenhum limiar do dev respeitar a perda máxima do freio

tempos = {}


def etapa(nome, funcao):
    print(f"\n==== {nome} ====", flush=True)
    t0 = time.time()
    r = funcao()
    tempos[nome] = round(time.time() - t0, 1)
    print(f"==== {nome}: {tempos[nome]} s ====", flush=True)
    return r


def rodar(script, *args, env=None):
    cmd = [sys.executable, os.path.join(AQUI, script), *args]
    r = subprocess.run(cmd, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise SystemExit(f"{script} falhou (código {r.returncode})")


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def gravar(p, doc):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def vetores():
    rodar("gerar_vetores.py", "--catalogo", CATALOGO, "--saida", VET, "--conjunto", DEV_ORIG, "--conjunto", TESTE_ORIG)


def candidatas():
    """A mesma fusão do montador, para perguntas que não estão no índice (as do dev e do teste)."""
    telas, M_f, M_q, n_perg = md.carregar(md.Path(CATALOGO), md.Path(os.path.join(VET, "fichas.npy")),
                                          md.Path(os.path.join(VET, "perguntas.npy")))
    rotas = [t["rota"] for t in telas]
    bm = md.BM25([md.texto_ficha(t["ficha"], t.get("descricao")) for t in telas])
    inicio = np.cumsum([0] + n_perg[:-1])
    com = np.flatnonzero(np.array(n_perg) > 0)

    def montar(origem, vet, destino):
        doc = json.load(open(origem, encoding="utf-8"))
        Q = np.load(vet).astype(np.float32)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        SF, SQ = Q @ M_f.T, Q @ M_q.T
        mel = np.full((len(Q), len(telas)), -1.0, dtype=np.float32)
        mel[:, com] = np.maximum.reduceat(SQ, inicio[com], axis=1)
        for r, c in enumerate(doc["casos"]):
            lf = [int(i) for i in np.argsort(-SF[r], kind="stable")[:md.K_LISTA]]
            lq = [int(i) for i in np.argsort(-mel[r], kind="stable")[:md.K_LISTA]]
            cand = [rotas[i] for i in md.rrf([lf, lq, bm.top(c["pergunta"])])[:md.K_LISTA]]
            faltam = [x for x in c["certas"] if x not in rotas]
            if faltam:
                raise SystemExit(f"{c['id']}: tela certa fora do catálogo: {faltam}")
            c["candidatas"] = cand
            c["base"] = {"busca_sem_llm": next((j for j, x in enumerate(cand, 1) if x in c["certas"]), None)}
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        gravar(destino, doc)
        return doc["casos"]

    dev = montar(DEV_ORIG, os.path.join(VET, "dev.npy"), DEV)
    teste = montar(TESTE_ORIG, os.path.join(VET, "teste.npy"), TESTE)
    gravar(CONGELADO, {"arquivo": "teste.json", "sha256": sha256(TESTE),
                       "congelado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "n": len(teste), "origem": os.path.relpath(TESTE_ORIG, RAIZ), "origem_sha256": sha256(TESTE_ORIG),
                       "aviso": "SINTÉTICO. Teste do exemplo: mede-se uma vez; para repetir, apague a pasta de saída."})
    os.chmod(TESTE, stat.S_IREAD)
    fora = sum(c["base"]["busca_sem_llm"] is None for c in dev + teste)
    print(f">> dev {len(dev)} e teste {len(teste)} perguntas, 40 candidatas cada; certa fora das 40: {fora}")


def dataset():
    rodar("montar_dataset.py", "--catalogo", CATALOGO, "--vetores-fichas", os.path.join(VET, "fichas.npy"),
          "--vetores-perguntas", os.path.join(VET, "perguntas.npy"), "--excluir", DEV_ORIG, "--excluir", TESTE_ORIG,
          "--saida", DADOS)


def treino():
    env = {"DADOS": DADOS, "OUTDIR": os.path.join(SAIDA, "checkpoints"), "LORA": ADAPTER, "RESULTADOS": RES,
           "MAX_PASSOS": "", "MAX_EXEMPLOS": ""}
    rodar("treinar.py", env=env)


def mais_novo(padrao):
    achados = sorted(glob.glob(os.path.join(RES, padrao)), key=os.path.getmtime)
    if not achados:
        raise SystemExit(f"nenhum resultado {padrao} em {RES}")
    return json.load(open(achados[-1], encoding="utf-8"))


def avaliar(*args):
    rodar("avaliar.py", "--telas", os.path.join(DADOS, "telas.json"), "--resultados", RES, "--lote", "40", *args)


def dev():
    avaliar("--conjunto", DEV, "--base")
    avaliar("--conjunto", DEV, "--adapter", ADAPTER)


def escolher_tau():
    esc = mais_novo("dev_adapter_*.json")["modelos"]["adapter"]["freio"].get("escolhido_no_dev")
    return (esc or {}).get("tau") or TAU_SEM_ESCOLHA


def teste(tau):
    avaliar("--conjunto", TESTE, "--congelado", CONGELADO, "--adapter", ADAPTER, "--uma-vez",
            "--tau", str(tau), "--tambem-base")


def linha(nome, pos):
    n = len(pos)
    return {"sistema": nome, "n": n, "top1": sum(p == 1 for p in pos), "top3": sum(bool(p and p <= 3) for p in pos),
            "mrr": round(sum(1 / p for p in pos if p) / max(1, n), 3)}


def resumo(tau):
    out = {"tempos_s": tempos, "tau_do_dev": tau}
    treino_ = json.loads(open(sorted(glob.glob(os.path.join(RES, "treino_*.json")))[-1], encoding="utf-8").read())
    out["treino"] = {"exemplos": treino_["exemplos_treino"], "passos": treino_["passos"],
                     "segundos": round(treino_["segundos_treino"], 1), "pico_vram_gb": treino_["pico_vram_gb"],
                     "perda": [(h["step"], round(h["loss"], 4)) for h in treino_["historico"] if "loss" in h],
                     "perda_dev_final": treino_["metricas"].get("final_eval_loss")}
    d_base = mais_novo("dev_base_*.json")["modelos"]["base_sem_treino"]
    d_tr = mais_novo("dev_adapter_*.json")["modelos"]["adapter"]
    t_doc = mais_novo("teste_adapter_*.json")
    t_tr, t_base = t_doc["modelos"]["adapter"], t_doc["modelos"]["base_sem_treino"]
    for nome, cas_base, m_base, m_tr in (("dev", d_tr["casos"], d_base, d_tr), ("teste", t_tr["casos"], t_base, t_tr)):
        tab = [linha("busca sem LLM (ordem da fusão)", [c["base"]["busca_sem_llm"] for c in cas_base]),
               linha("Qwen 3B sem treino", [c["pos_local"] for c in m_base["casos"]]),
               linha("Qwen 3B com QLoRA (exemplo)", [c["pos_local"] for c in m_tr["casos"]])]
        a = [c["pos_local"] == 1 for c in m_tr["casos"]]
        b = [c["pos_local"] == 1 for c in m_base["casos"]]
        out[nome] = {"tabela": tab, "mcnemar_treinado_vs_base": mcnemar(a, b),
                     "mcnemar_treinado_vs_busca": m_tr["comparacao_top1"].get("busca_sem_llm"),
                     "seg_por_pergunta": {"base": m_base["seg_por_pergunta"], "treinado": m_tr["seg_por_pergunta"]},
                     "pico_vram_gb": {"base": m_base["pico_vram_gb"], "treinado": m_tr["pico_vram_gb"]}}
    gravar(os.path.join(SAIDA, "RESUMO.json"), out)

    print("\n================ RESUMO (dados SINTÉTICOS: os números não dizem nada sobre o experimento) ================")
    print(f"Treino: {out['treino']['exemplos']} exemplos, {out['treino']['passos']} passos, "
          f"{out['treino']['segundos']} s, pico {out['treino']['pico_vram_gb']} GB")
    print("Perda (passo: perda): " + ", ".join(f"{s}: {v}" for s, v in out["treino"]["perda"]))
    for nome in ("dev", "teste"):
        x = out[nome]
        print(f"\n{nome} ({x['tabela'][0]['n']} perguntas)")
        print(f"  {'sistema':34s} {'top-1':>6s} {'top-3':>6s} {'MRR':>6s}")
        for r in x["tabela"]:
            print(f"  {r['sistema']:34s} {r['top1']:>6d} {r['top3']:>6d} {r['mrr']:>6.3f}")
        m = x["mcnemar_treinado_vs_base"]
        print(f"  McNemar treinado × sem treino: só o treinado acerta {m['so_local_acerta']}, "
              f"só o sem treino {m['so_base_acerta']}, p = {m['p']}")
        m = x["mcnemar_treinado_vs_busca"] or {}
        if m:
            print(f"  McNemar treinado × busca: só o treinado acerta {m['so_local_acerta']}, "
                  f"só a busca {m['so_base_acerta']}, p = {m['p']}")
    print("\nTempos (s): " + ", ".join(f"{k} {v}" for k, v in tempos.items()) + f" · total {sum(tempos.values()):.0f}")
    print(f"Limiar do freio escolhido no dev: {tau}")
    print(f"Tudo em: {SAIDA}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if os.path.exists(SAIDA):
        raise SystemExit(f"{SAIDA} já existe: o teste do exemplo já foi medido e mede-se uma vez.\n"
                         f"Para repetir o exemplo do zero, apague essa pasta (rm -rf \"{SAIDA}\").")
    os.makedirs(RES)
    t0 = time.time()
    etapa("vetores", vetores)
    etapa("candidatas", candidatas)
    etapa("dataset", dataset)
    etapa("treino", treino)
    etapa("dev", dev)
    tau = escolher_tau()
    etapa("teste", lambda: teste(tau))
    resumo(tau)
    print(f"EXEMPLO_OK em {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
