"""
Treino QLoRA do reranker local: dada uma pergunta de usuário e o cartão de UMA tela, responder
«Sim» (é aqui que se faz) ou «Não». Rodou em WSL2/Ubuntu numa GPU de notebook com 8 GB (Blackwell).

    python src/treinar.py

Variáveis (opcionais):
    MODELO=unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit   o modelo base em 4 bits
    DADOS=dados                                          pasta com treino.jsonl e dev_sintetico.jsonl
    MAX_PASSOS=30                                        só para medir velocidade e memória (teste de fumaça)
    EPOCAS=1
    MAX_EXEMPLOS=8000                                    corta o treino (os primeiros N, já embaralhados)
    OUTDIR=saida/checkpoints                             onde ficam os checkpoints (a cada 250 passos)
    LORA=saida/adapter                                   onde grava o adapter final

O dataset vem de montar_dataset.py. O teste congelado não está em DADOS e este script não o lê.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

# Reduz fragmentacao de VRAM (importante em 8 GB)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

RAIZ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL     = os.environ.get("MODELO", "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit")
MAX_SEQ   = 768                     # o maior exemplo tem ~500 tokens
DADOS     = os.path.join(RAIZ, os.environ.get("DADOS", "dados"))
DATA      = os.path.join(DADOS, "treino.jsonl")
DEV       = os.path.join(DADOS, "dev_sintetico.jsonl")
N_DEV     = 400                     # exemplos do dev sintético para a perda de validação
OUTDIR    = os.environ.get("OUTDIR", os.path.join(RAIZ, "saida", "checkpoints"))
LORA      = os.environ.get("LORA", os.path.join(RAIZ, "saida", "adapter"))
RESULTADOS = os.path.join(RAIZ, "resultados")
MAX_PASSOS   = int(os.environ.get("MAX_PASSOS") or -1)
EPOCAS       = float(os.environ.get("EPOCAS") or 1)
MAX_EXEMPLOS = int(os.environ.get("MAX_EXEMPLOS") or 0)
FUMACA    = MAX_PASSOS > 0

t0 = time.time()
print(">> Carregando modelo base 4-bit:", MODEL, flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL,
    max_seq_length = MAX_SEQ,
    load_in_4bit   = True,
    dtype          = None,   # auto -> bf16 na Blackwell
)

print(">> Aplicando LoRA (QLoRA r=16)", flush=True)
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 3407,
)

print(">> Carregando dataset:", DATA, flush=True)
ds = load_dataset("json", data_files=DATA, split="train")
if MAX_EXEMPLOS:
    ds = ds.select(range(min(MAX_EXEMPLOS, len(ds))))
dev = load_dataset("json", data_files=DEV, split="train").shuffle(seed=3407).select(range(N_DEV))
print(">> Exemplos: treino", len(ds), "| dev", len(dev), flush=True)


def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt, remove_columns=list(ds.column_names))
dev = dev.map(fmt, remove_columns=list(dev.column_names))

trainer = SFTTrainer(
    model           = model,
    processing_class = tokenizer,
    train_dataset   = ds,
    eval_dataset    = dev,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ,
        packing                     = False,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 8,
        gradient_accumulation_steps = 2,
        warmup_steps                = 30,
        num_train_epochs            = EPOCAS,
        max_steps                   = MAX_PASSOS,
        learning_rate               = 2e-4,
        logging_steps               = 10,
        eval_strategy               = "no" if FUMACA else "steps",
        eval_steps                  = 250,
        save_strategy               = "no" if FUMACA else "steps",
        save_steps                  = 250,
        save_total_limit            = 5,      # todos os checkpoints (250, 500, 750, 1000): escolhe-se no dev
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        bf16                        = is_bfloat16_supported(),
        fp16                        = not is_bfloat16_supported(),
        output_dir                  = OUTDIR,
        report_to                   = "none",
    ),
)

# a perda só na resposta («Sim»/«Não»): a pergunta e o cartão são contexto, não o que o modelo deve escrever
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

torch.cuda.reset_peak_memory_stats()
print(">> Iniciando treino..." + (f" (fumaça: {MAX_PASSOS} passos)" if FUMACA else ""), flush=True)
stats = trainer.train()
pico_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
metricas = dict(stats.metrics)
if not FUMACA:
    metricas.update({f"final_{k}": v for k, v in trainer.evaluate().items()})

passos = trainer.state.global_step
seg = metricas.get("train_runtime") or 0
resumo = {
    "quando": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "modelo_base": MODEL, "fumaca": FUMACA, "exemplos_treino": len(ds), "passos": passos,
    "lote_efetivo": 16, "segundos_treino": seg, "seg_por_passo": round(seg / max(1, passos), 2),
    "estimativa_1_epoca_min": round(seg / max(1, passos) * (len(ds) / 16) / 60, 1),
    "pico_vram_gb": round(pico_gb, 2), "metricas": metricas,
    "historico": trainer.state.log_history,
}
os.makedirs(RESULTADOS, exist_ok=True)
nome = ("fumaca_" if FUMACA else "treino_") + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
with open(os.path.join(RESULTADOS, nome), "w", encoding="utf-8") as f:
    json.dump(resumo, f, ensure_ascii=False, indent=1)
print(">> Resumo:", json.dumps({k: v for k, v in resumo.items() if k not in ("historico", "metricas")},
                               ensure_ascii=False), flush=True)

if FUMACA:
    print("FUMACA_OK (adapter não salvo)", flush=True)
    sys.exit(0)

print(">> Salvando adapter LoRA em:", LORA, flush=True)
model.save_pretrained(LORA)
tokenizer.save_pretrained(LORA)
print("TREINO_OK", flush=True)
