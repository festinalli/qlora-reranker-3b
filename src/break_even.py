"""
Ponto de equilíbrio: a partir de quantas perguntas por mês uma GPU dedicada, ligada 24x7, custa menos do que
pagar o rerank à API por pergunta.

    python src/break_even.py --usd-hora 0.894 --usd-por-mil 2.80

Só faz a conta; os preços são parâmetros. Os valores padrão são os usados no experimento (ver docs/break_even.md):
eles NÃO são atualizados sozinhos e precisam ser reconferidos antes de qualquer decisão.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd-hora", type=float, default=0.894, help="preço da VM com GPU, US$ por hora")
    ap.add_argument("--horas-mes", type=float, default=730, help="horas ligada por mês (730 = 24x7)")
    ap.add_argument("--usd-por-mil", type=float, default=2.80, help="custo da API, US$ por mil perguntas")
    a = ap.parse_args()
    mensal = a.usd_hora * a.horas_mes
    equilibrio = mensal / (a.usd_por_mil / 1000)
    print(f"GPU: US$ {a.usd_hora}/h x {a.horas_mes:g} h = US$ {mensal:,.1f} por mês")
    print(f"API: US$ {a.usd_por_mil} por mil perguntas")
    print(f"Ponto de equilíbrio: ~{equilibrio:,.0f} perguntas por mês")
    print("Abaixo disso a API sai mais barata; acima, a GPU (sem contar operação, energia, redundância e pessoas).")


if __name__ == "__main__":
    main()
