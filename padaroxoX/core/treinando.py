"""
PARADOXO X — Treinando v2
==========================
Orquestrador de treino com:
  - Retomada real de onde parou (pula exemplos já vistos)
  - Barra de progresso com % e tempo estimado
  - Salva progresso a cada checkpoint
  - Trata Ctrl+C com salvamento automático
"""

import os
import sys
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.treino import treinar, AdamOtimizador, calcular_loss, forward_com_cache, Backward
from core.transformer import ParadoxoTransformer
from core.tokenizer import Tokenizer

# ═══════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════

CAMINHO_VOCAB    = os.path.join('memory', 'vocab.json')
CAMINHO_MODELO   = os.path.join('core', 'modelo_paradoxox.json')
CAMINHO_PROGRESSO = os.path.join('core', 'progresso_treino.json')
DB_PATH          = os.path.join('memory', 'paradoxox_dataset.db')

EPOCAS           = 1
BATCH_SIZE       = 8
SEQ_LEN          = 128
LR               = 3e-4
LOG_CADA         = 50
SALVAR_CADA      = 500


# ═══════════════════════════════════════════════════════
# PROGRESSO
# ═══════════════════════════════════════════════════════

def carregar_progresso() -> dict:
    if os.path.exists(CAMINHO_PROGRESSO):
        try:
            with open(CAMINHO_PROGRESSO, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"total_exemplos_vistos": 0, "epoca_atual": 1}

def salvar_progresso(total_vistos: int, epoca: int):
    with open(CAMINHO_PROGRESSO, 'w') as f:
        json.dump({
            "total_exemplos_vistos": total_vistos,
            "epoca_atual": epoca
        }, f)

def resetar_progresso():
    if os.path.exists(CAMINHO_PROGRESSO):
        os.remove(CAMINHO_PROGRESSO)


# ═══════════════════════════════════════════════════════
# BARRA DE PROGRESSO
# ═══════════════════════════════════════════════════════

def barra_progresso(atual: int, total: int, loss: float, inicio: float, largura: int = 28):
    pct = atual / max(total, 1)
    preenchido = int(largura * pct)
    barra = "█" * preenchido + "░" * (largura - preenchido)

    decorrido = time.time() - inicio
    if atual > 0:
        restante = (decorrido / atual) * (total - atual)
        h = int(restante // 3600)
        m = int((restante % 3600) // 60)
        s = int(restante % 60)
        if h > 0:
            tempo_str = f"{h}h{m:02d}m"
        elif m > 0:
            tempo_str = f"{m}m{s:02d}s"
        else:
            tempo_str = f"{s}s"
    else:
        tempo_str = "--"

    print(
        f"\r  [{barra}] {pct:.1%} | Ex {atual:,}/{total:,} | "
        f"Loss {loss:.4f} | Resta: {tempo_str}   ",
        end="", flush=True
    )


# ═══════════════════════════════════════════════════════
# LOOP DE TREINO PRÓPRIO (com skip de exemplos já vistos)
# ═══════════════════════════════════════════════════════

def treinar_com_progresso(modelo, tokenizer, progresso: dict):
    import sys, os, math
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _root)
    from memory.dataloader import DataLoader

    total_vistos = progresso["total_exemplos_vistos"]
    epoca_atual  = progresso["epoca_atual"]

    otim = AdamOtimizador(modelo, lr=LR)
    loader = DataLoader(
        tokenizer  = tokenizer,
        batch_size = BATCH_SIZE,
        seq_len    = SEQ_LEN,
        db_path    = DB_PATH,
        shuffle    = "global",
        verbose    = True,
    )

    # Total estimado de exemplos na época
    total_epoca = 202441  # usa o valor do banco
    total_geral = total_epoca * EPOCAS

    print(f"\n  ⏭️  Pulando {total_vistos:,} exemplos já vistos..." if total_vistos > 0 else "")

    soma_loss = 0.0
    conta_loss = 0
    exemplos_sessao = 0
    inicio = time.time()

    try:
        for batch in loader.gerar_batches(epocas=EPOCAS):
            for i in range(batch.n_textos):
                ids_entrada = batch.input_ids[i].tolist()
                ids_alvo    = batch.target_ids[i].tolist()

                if not ids_entrada or not ids_alvo:
                    continue

                # Pula exemplos já vistos na sessão anterior
                if exemplos_sessao + total_vistos < total_vistos:
                    exemplos_sessao += 1
                    continue

                # Forward
                logits, cache = forward_com_cache(modelo, ids_entrada)

                # Loss
                token_alvo = ids_alvo[-1]
                loss = calcular_loss(logits, token_alvo)
                soma_loss += loss
                conta_loss += 1
                exemplos_sessao += 1
                total_vistos += 1

                # Backward
                grads = Backward.calcular(
                    modelo      = modelo,
                    ids_entrada = ids_entrada,
                    token_alvo  = token_alvo,
                    cache       = cache,
                )
                otim.step(grads)

                # Log e barra de progresso
                if total_vistos % LOG_CADA == 0:
                    media = soma_loss / conta_loss
                    barra_progresso(total_vistos, total_geral, media, inicio)
                    soma_loss  = 0.0
                    conta_loss = 0

                # Checkpoint
                if total_vistos % SALVAR_CADA == 0:
                    modelo.salvar(CAMINHO_MODELO)
                    salvar_progresso(total_vistos, batch.epoch)
                    print(f"\n  💾 Checkpoint salvo — {total_vistos:,} exemplos")

    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  Interrompido pelo usuário.")
        print(f"  💾 Salvando modelo e progresso...")
        modelo.salvar(CAMINHO_MODELO)
        salvar_progresso(total_vistos, 1)
        print(f"  ✅ Salvo! {total_vistos:,} exemplos processados nesta sessão.")
        print(f"     Rode novamente para continuar.")
        sys.exit(0)

    modelo.salvar(CAMINHO_MODELO)
    resetar_progresso()
    print(f"\n\n  ✅ Época concluída! Total: {total_vistos:,} exemplos")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 52)
    print("  ⚛️   PARADOXO X — Treinamento v2")
    print("═" * 52)

    # Tokenizer
    tok = Tokenizer()
    if os.path.exists(CAMINHO_VOCAB):
        tok.carregar(CAMINHO_VOCAB)
        print(f"  📖 Vocab: {len(tok.vocab):,} tokens")
    else:
        print(f"  ❌ Vocab não encontrado: {CAMINHO_VOCAB}")
        sys.exit(1)

    # Modelo
    modelo = ParadoxoTransformer(tamanho_vocab=len(tok.vocab))
    if os.path.exists(CAMINHO_MODELO):
        try:
            modelo.carregar(CAMINHO_MODELO)
            print(f"  🔄 Modelo carregado: {CAMINHO_MODELO}")
        except Exception as e:
            print(f"  ⚠️  Erro ao carregar modelo: {e} — iniciando do zero")
    else:
        print(f"  🆕 Modelo novo criado")

    # Progresso
    progresso = carregar_progresso()
    if progresso["total_exemplos_vistos"] > 0:
        print(f"  ⏭️  Progresso anterior: {progresso['total_exemplos_vistos']:,} exemplos vistos")
        print(f"     Continuando de onde parou...")
    else:
        print(f"  🚀 Iniciando do zero")

    print(f"\n  Épocas: {EPOCAS} | Batch: {BATCH_SIZE} | Seq: {SEQ_LEN} | LR: {LR}")
    print("═" * 52 + "\n")

    treinar_com_progresso(modelo, tok, progresso)

    print(f"\n  💾 Modelo salvo em: {CAMINHO_MODELO}")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()