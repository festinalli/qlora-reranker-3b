"""
Junta as perguntas dos escritores com o veredito dos conferentes e CONGELA o teste cego.

    python src/congelar_teste.py --rascunho teste/rascunho --telas dados/telas.json --saida teste

Espera em --rascunho, por lote L: perguntas_L.json (lista de {pergunta, tela_gabarito, grupo, fraco?}) e
conferido_L.json (mesma ordem: {pergunta, veredito: "ok"|"recusa", certas: [rotas], motivo}). Um lote chamado
«vagas» marca as perguntas como tipo «vaga»; os outros, «especifica».

Entra no teste: veredito «ok», pelo menos uma tela certa que existe no catálogo e nenhuma citação literal do nome
técnico da tela (vazamento). Grava teste.json, CONGELADO.json (sha256, data, contagens, sha256 do rascunho) e deixa
o teste só de leitura. Recusa congelar duas vezes: o teste congelado não se reescreve.

Adaptado do script original: os caminhos do PC viraram parâmetros.
"""
import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vazou(pergunta, rota):
    nome = rota.split("/")[-1]
    if sum(ch.isupper() for ch in nome) < 2:      # nome que é palavra comum não conta como vazamento
        return False
    return nome.lower() in re.sub(r"\s+", "", pergunta.lower())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rascunho", type=Path, required=True)
    ap.add_argument("--telas", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    saida, congelado = a.saida / "teste.json", a.saida / "CONGELADO.json"
    if congelado.exists():
        raise SystemExit(f"o teste já foi congelado ({congelado}): não se reescreve")
    telas = json.loads(a.telas.read_text(encoding="utf-8"))
    casos, recusas, avisos = [], Counter(), []
    lotes = sorted(p.stem.split("_", 1)[1] for p in a.rascunho.glob("perguntas_*.json"))
    for lote in lotes:
        tipo = "vaga" if lote == "vagas" else "especifica"
        perg = json.loads((a.rascunho / f"perguntas_{lote}.json").read_text(encoding="utf-8"))
        conf = json.loads((a.rascunho / f"conferido_{lote}.json").read_text(encoding="utf-8"))
        if len(perg) != len(conf):
            raise SystemExit(f"lote {lote}: {len(perg)} perguntas e {len(conf)} conferidas")
        for p, c in zip(perg, conf):
            if c.get("pergunta") != p["pergunta"]:
                avisos.append(f"{lote}: o conferente mudou o texto de uma pergunta (fica o original)")
            gab = p.get("tela_gabarito")
            if c.get("veredito") != "ok":
                recusas["conferente"] += 1
                continue
            certas = [r for r in dict.fromkeys(c.get("certas") or []) if r in telas]
            if gab in telas and gab not in certas:
                avisos.append(f"{lote}: «ok» sem a tela do gabarito nas certas ({gab}); fica a lista do conferente")
            if not certas:
                recusas["sem_tela_certa_no_catalogo"] += 1
                continue
            if any(vazou(p["pergunta"], r) for r in certas):
                recusas["vazamento_nome_da_tela"] += 1
                continue
            casos.append({"grupo": p.get("grupo") or gab.split("/")[0], "tipo": tipo, "fraco": bool(p.get("fraco")),
                          "pergunta": p["pergunta"], "certas": certas, "tela_do_caso": gab,
                          "motivo_conferente": c.get("motivo", "")})
    for i, c in enumerate(casos, 1):
        c["id"] = f"N{i:03d}"

    a.saida.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "teste-cego-reranker/2",
           "congelado": "TESTE CEGO: nunca usar para treinar nem para escolher nada (checkpoint, limiar, prompt). "
                        "Mede-se uma vez.",
           "casos": casos}
    saida.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    cong = {
        "arquivo": saida.name, "sha256": sha256(saida),
        "congelado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(casos), "por_grupo": dict(Counter(c["grupo"] for c in casos)),
        "por_tipo": dict(Counter(c["tipo"] for c in casos)), "fracos": sum(c["fraco"] for c in casos),
        "com_mais_de_uma_certa": sum(len(c["certas"]) > 1 for c in casos),
        "recusadas": dict(recusas), "avisos": avisos,
        "rascunho_sha256": {p.name: sha256(p) for p in sorted(a.rascunho.glob("*.json"))},
    }
    congelado.write_text(json.dumps(cong, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    os.chmod(saida, stat.S_IREAD)
    print(json.dumps({k: v for k, v in cong.items() if k != "rascunho_sha256"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
