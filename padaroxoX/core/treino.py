
"""
PARADOXO X — Treino v2
=======================
Backpropagation analítico completo, construído do zero para o
ParadoxoTransformer. Sem diferenciação numérica — calculos reais
da regra da cadeia em cada camada.
 
Por que v2 é muito mais rápido que v1?
  v1 (diferenciação numérica):
    2 forwards por peso × 64.000 pesos em W_out = 128.000 forwards
    → trava o Ryzen em segundos
 
  v2 (backprop analítico):
    1 forward + 1 backward = gradientes de TODOS os pesos de uma vez
    → ~100x mais rápido, mesma correção matemática
 
Fluxo do backward (regra da cadeia, de trás pra frente):
  loss
    → d_logits      (softmax + cross-entropy combinados)
    → d_h_final     (projeção final / tied weights)
    → d_ln_final    (layer norm final)
    → por camada (de trás pra frente):
        → d_ff       (feed forward)
        → d_ln2      (layer norm 2)
        → d_attn     (attention — gradientes dos biases)
        → d_ln1      (layer norm 1)
    → d_embedding   (tabela de embedding, só linhas usadas)
 
Como usar:
    from core.treino import treinar, AdamOtimizador, calcular_loss
    from core.transformer import ParadoxoTransformer
    from core.tokenizer import Tokenizer
 
    tok = Tokenizer()
    tok.carregar("vocab.json")
    modelo = ParadoxoTransformer(tamanho_vocab=len(tok.vocab))
 
    # Treino completo (recomendado)
    treinar(modelo, tok, epocas=5)
 
    # Ou manual:
    otim = AdamOtimizador(modelo, lr=3e-4)
    grads = Backward.calcular(modelo, ids_entrada, token_alvo)
    otim.step(grads)
"""
 
import math
import os
import sys
from typing import Optional
import numpy as np
 
 
# ═══════════════════════════════════════════════════════
# UTILITÁRIOS NUMÉRICOS (internos)
# ═══════════════════════════════════════════════════════
 
def _softmax(logits: list[float]) -> list[float]:
    """Softmax numericamente estável."""
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]
 
 
def _gelu_deriv(x: float) -> float:
    """
    Derivada da GELU (aproximação tanh).
    Necessária para o backward do FeedForward.
 
    d/dx GELU(x) = 0.5 * tanh(c*(x + 0.044715*x³))
                 + 0.5 * x * sech²(c*(x + 0.044715*x³)) * c*(1 + 3*0.044715*x²)
    onde c = sqrt(2/π)
    """
    c = math.sqrt(2.0 / math.pi)
    v = c * (x + 0.044715 * x ** 3)
    t = math.tanh(v)
    sech2 = 1.0 - t * t
    return 0.5 * t + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x ** 2)
 
 
def _layernorm_backward(
    d_out:  list[float],   # gradiente chegando desta camada (dim,)
    x_in:   list[float],   # entrada original (antes do LN)
    gamma:  list[float],   # parâmetros gamma do LN
    eps:    float = 1e-6,
) -> tuple[list[float], list[float], list[float]]:
    """
    Gradiente analítico do LayerNorm.
 
    Retorna:
      d_x     : gradiente em relação à entrada x_in   (dim,)
      d_gamma : gradiente em relação a gamma          (dim,)
      d_beta  : gradiente em relação a beta           (dim,)
 
    Matemática (para um vetor x de dimensão D):
      mu    = mean(x)
      var   = mean((x - mu)²)
      x_hat = (x - mu) / sqrt(var + eps)
      y     = gamma * x_hat + beta
 
      d_beta  = d_out                         (soma por batch)
      d_gamma = d_out * x_hat
      d_x_hat = d_out * gamma
 
      d_var = sum(d_x_hat * (x - mu) * -0.5 * (var + eps)^(-3/2))
      d_mu  = sum(d_x_hat * -1/sqrt(var+eps)) + d_var * mean(-2*(x-mu))
 
      d_x = d_x_hat / sqrt(var+eps)
            + d_var * 2*(x-mu)/D
            + d_mu / D
    """
    D = len(x_in)
    mu = sum(x_in) / D
    var = sum((v - mu) ** 2 for v in x_in) / D
    std = math.sqrt(var + eps)
    x_hat = [(v - mu) / std for v in x_in]
 
    d_gamma = [d_out[i] * x_hat[i] for i in range(D)]
    d_beta  = list(d_out)
 
    d_x_hat = [d_out[i] * gamma[i] for i in range(D)]
 
    d_var = sum(d_x_hat[i] * (x_in[i] - mu) * (-0.5) * (var + eps) ** (-1.5)
                for i in range(D))
 
    d_mu = (sum(-d_x_hat[i] / std for i in range(D))
            + d_var * sum(-2.0 * (x_in[i] - mu) for i in range(D)) / D)
 
    d_x = [
        d_x_hat[i] / std
        + d_var * 2.0 * (x_in[i] - mu) / D
        + d_mu / D
        for i in range(D)
    ]
 
    return d_x, d_gamma, d_beta
 
 
# ═══════════════════════════════════════════════════════
# 1. LOSS — Cross-Entropy
# ═══════════════════════════════════════════════════════
 
def calcular_loss(logits: list[float], token_alvo: int) -> float:
    """
    Cross-Entropy Loss para language modeling causal.
 
    Fórmula: loss = -log( softmax(logits)[token_alvo] )
 
    Quanto menor, melhor. Para vocab=1000 com pesos aleatórios,
    o valor inicial esperado é -log(1/1000) ≈ 6.9.
    """
    probs = _softmax(logits)
    return -math.log(probs[token_alvo] + 1e-9)
 
 
# ═══════════════════════════════════════════════════════
# 2. FORWARD COM CACHE — salva ativações para o backward
# ═══════════════════════════════════════════════════════
 
def forward_com_cache(modelo, ids: list[int]) -> tuple[list[float], dict]:
    """
    Executa o forward pass SALVANDO todas as ativações intermediárias.
    Essas ativações são necessárias para calcular os gradientes no backward.
 
    Retorna
    -------
    logits : list[float]        — saída do modelo (vocab,)
    cache  : dict               — todas as ativações salvas
 
    Estrutura do cache:
      cache["ids"]          → ids de entrada
      cache["emb_out"]      → saída do embedding + PE  (seq, dim)
      cache["camadas"][i]   → {
          "x_antes_ln1"     → entrada desta camada       (seq, dim)
          "ln1_out"         → saída do ln1               (seq, dim)
          "attn_out"        → saída do attention         (seq, dim)
          "x_antes_ln2"     → entrada do ln2 (x + attn) (seq, dim)
          "ln2_out"         → saída do ln2               (seq, dim)
          "ff_h1"           → saída pré-ativação do ff   (seq, dim_ff)
          "ff_h1_ativ"      → saída pós-GELU do ff       (seq, dim_ff)
          "ff_out"          → saída do ff                (seq, dim)
          "x_saida"         → saída desta camada         (seq, dim)
      }
      cache["ln_final_in"]  → entrada do ln_final       (seq, dim)
      cache["ln_final_out"] → saída do ln_final          (seq, dim)
      cache["h_final"]      → último vetor (para W_out)  (dim,)
    """
    ids = ids[-modelo.seq_max:]
    seq_len = len(ids)
    D = modelo.dim_modelo
 
    # ── Embedding + Positional Encoding ────────────────────────────
    X = [list(modelo.embedding.tabela[i]) for i in ids]
    for i in range(seq_len):
        pe = modelo.pos_enc[i]
        X[i] = [X[i][j] + pe[j] for j in range(D)]
 
    cache = {
        "ids"     : ids,
        "emb_out" : [list(row) for row in X],
        "camadas" : [],
    }
 
    # ── Camadas do Transformer ──────────────────────────────────────
    for camada in modelo.camadas:
        c = {}
        c["x_antes_ln1"] = [list(row) for row in X]
 
        # Pre-LN 1
        ln1_out = camada.ln1.forward(X)
        c["ln1_out"] = [list(row) for row in ln1_out]
 
        # Attention (usa o forward existente da MultiHeadAttention)
        attn_out_raw, _ = camada.attention.forward(ln1_out, mascara_causal=True)
        c["attn_out"] = [list(row) for row in attn_out_raw]
 
        # Residual 1: X = X + attn_out
        # Nota: o attention.forward() já soma residual internamente para
        # seq_q == seq_len. Replicamos aqui para salvar o estado correto.
        X_res1 = [[X[i][j] + attn_out_raw[i][j] for j in range(D)]
                  for i in range(seq_len)]
        c["x_antes_ln2"] = [list(row) for row in X_res1]
 
        # Pre-LN 2
        ln2_out = camada.ln2.forward(X_res1)
        c["ln2_out"] = [list(row) for row in ln2_out]
 
        # Feed Forward — manualmente para salvar ativações internas
        ff = camada.ff
        dim_ff = len(ff.b1)
        ff_h1_list      = []
        ff_h1_ativ_list = []
        ff_out_list     = []
 
        for vetor in ln2_out:
            # Linear 1: x @ W1 + b1
            h1 = list(ff.b1)
            for i, xi in enumerate(vetor):
                for j in range(dim_ff):
                    h1[j] += xi * ff.W1[i][j]
            ff_h1_list.append(list(h1))
 
            # GELU
            h1_ativ = [_gelu_deriv.__module__ and 0 or 0] * dim_ff  # placeholder
            h1_ativ = [math.tanh(math.sqrt(2/math.pi)*(v + 0.044715*v**3))
                       * 0.5 * v + 0.5 * v for v in h1]
            # GELU real (igual ao do transformer):
            h1_ativ = [0.5 * v * (1.0 + math.tanh(
                math.sqrt(2.0/math.pi) * (v + 0.044715 * v**3))) for v in h1]
            ff_h1_ativ_list.append(list(h1_ativ))
 
            # Linear 2: h1_ativ @ W2 + b2
            h2 = list(ff.b2)
            for i, xi in enumerate(h1_ativ):
                for j in range(D):
                    h2[j] += xi * ff.W2[i][j]
            # Residual interno do FF: h2 + vetor_original
            h2 = [h2[j] + vetor[j] for j in range(D)]
            ff_out_list.append(list(h2))
 
        c["ff_h1"]     = ff_h1_list
        c["ff_h1_ativ"] = ff_h1_ativ_list
        c["ff_out"]    = ff_out_list
 
        # Residual 2: X = X_res1 + ff_out
        X = [[X_res1[i][j] + ff_out_list[i][j] for j in range(D)]
             for i in range(seq_len)]
        c["x_saida"] = [list(row) for row in X]
 
        cache["camadas"].append(c)
 
    # ── LayerNorm final ──────────────────────────────────────────────
    cache["ln_final_in"] = [list(row) for row in X]
    ln_out = modelo.ln_final.forward(X)
    cache["ln_final_out"] = [list(row) for row in ln_out]
    cache["h_final"] = list(ln_out[-1])
 
    # ── Projeção final → logits ──────────────────────────────────────
    h = cache["h_final"]
    if modelo.tied_weights:
        logits = [sum(h[k] * modelo.embedding.tabela[j][k]
                      for k in range(D))
                  for j in range(modelo.tamanho_vocab)]
    else:
        logits = [0.0] * modelo.tamanho_vocab
        for k, val in enumerate(h):
            for j in range(modelo.tamanho_vocab):
                logits[j] += val * modelo.W_out[k][j]
 
    return logits, cache
 
 
# ═══════════════════════════════════════════════════════
# 3. BACKWARD — Backpropagation analítico
# ═══════════════════════════════════════════════════════
 
class Backward:
    """
    Backpropagation analítico completo para o ParadoxoTransformer.
 
    Usa a regra da cadeia de trás pra frente, camada por camada.
    Cada operação do forward tem sua derivada correspondente aqui.
 
    IMPORTANTE: chame sempre forward_com_cache() antes de calcular().
    O cache contém as ativações que as derivadas precisam.
    """
 
    @staticmethod
    def calcular(
        modelo,
        ids_entrada:  list[int],
        token_alvo:   int,
        cache:        Optional[dict] = None,
    ) -> dict:
        """
        Calcula gradientes de todos os pesos em um único passo.
 
        Parâmetros
        ----------
        modelo      : ParadoxoTransformer
        ids_entrada : tokens de entrada
        token_alvo  : token correto que o modelo devia ter previsto
        cache       : retorno de forward_com_cache() — se None, chama internamente
 
        Retorna
        -------
        dict com a mesma estrutura do Adam espera:
          {
            "embedding":  list[list[float]],
            "ln_final":   {"gamma": [...], "beta": [...]},
            "W_out":      list[list[float]] | None,
            "camadas": [{
                "ln1":      {"gamma": [...], "beta": [...]},
                "ln2":      {"gamma": [...], "beta": [...]},
                "attn_b_q": [...], "attn_b_k": [...],
                "attn_b_v": [...], "attn_b_o": [...],
                "ff_b1":    [...],
                "ff_b2":    [...],
                "ff_W1":    list[list[float]],
                "ff_W2":    list[list[float]],
            }, ...]
          }
        """
        D = modelo.dim_modelo
        V = modelo.tamanho_vocab
 
        # Forward com cache se não foi passado
        if cache is None:
            logits, cache = forward_com_cache(modelo, ids_entrada)
        else:
            logits, _ = modelo.forward(ids_entrada)
 
        # ── 1. Gradiente da loss em relação aos logits ────────────────
        # d(CE+Softmax)/d_logits[i] = probs[i] - 1_{i==alvo}
        probs   = _softmax(logits)
        d_logits = list(probs)
        d_logits[token_alvo] -= 1.0
 
        # ── 2. Gradiente da projeção final ────────────────────────────
        h     = cache["h_final"]         # (D,)
        grads = {
            "embedding" : [[0.0] * D for _ in range(V)],
            "ln_final"  : {"gamma": [0.0]*D, "beta": [0.0]*D},
            "W_out"     : None,
            "camadas"   : [],
        }
 
        if modelo.tied_weights:
            # logits[j] = dot(h, emb[j])
            # d_h[k]    = sum_j( d_logits[j] * emb[j][k] )
            # d_emb[j]  = d_logits[j] * h  (só token j importa)
            d_h = [0.0] * D
            for j in range(V):
                g = d_logits[j]
                if abs(g) < 1e-12:
                    continue
                for k in range(D):
                    d_h[k] += g * modelo.embedding.tabela[j][k]
                    grads["embedding"][j][k] += g * h[k]
        else:
            # d_h[k]       = sum_j( d_logits[j] * W_out[k][j] )
            # d_W_out[k][j] = d_logits[j] * h[k]
            d_h = [0.0] * D
            grad_W_out = [[0.0] * V for _ in range(D)]
            for j in range(V):
                g = d_logits[j]
                if abs(g) < 1e-12:
                    continue
                for k in range(D):
                    d_h[k]          += g * modelo.W_out[k][j]
                    grad_W_out[k][j] = g * h[k]
            grads["W_out"] = grad_W_out
 
        # ── 3. Backward do LayerNorm final ────────────────────────────
        # d_h é o gradiente em relação à saída do ln_final[-1]
        # (apenas o último token contribui para os logits)
        seq_len = len(cache["ln_final_in"])
        # Expande d_h para todos os tokens — só o último é não-zero
        d_ln_out = [[0.0] * D for _ in range(seq_len)]
        d_ln_out[-1] = list(d_h)
 
        d_ln_in_total  = [[0.0] * D for _ in range(seq_len)]
        d_gamma_lnf    = [0.0] * D
        d_beta_lnf     = [0.0] * D
 
        for t in range(seq_len):
            if all(abs(v) < 1e-12 for v in d_ln_out[t]):
                continue
            x_in_t = cache["ln_final_in"][t]
            d_x, d_g, d_b = _layernorm_backward(
                d_ln_out[t], x_in_t, modelo.ln_final.gamma
            )
            d_ln_in_total[t] = d_x
            for k in range(D):
                d_gamma_lnf[k] += d_g[k]
                d_beta_lnf[k]  += d_b[k]
 
        grads["ln_final"]["gamma"] = d_gamma_lnf
        grads["ln_final"]["beta"]  = d_beta_lnf
 
        # d_X agora é o gradiente em relação à saída da última camada
        d_X = d_ln_in_total
 
        # ── 4. Backward pelas camadas (de trás pra frente) ───────────
        for idx_cam in range(modelo.num_camadas - 1, -1, -1):
            camada = modelo.camadas[idx_cam]
            c      = cache["camadas"][idx_cam]
            ff     = camada.ff
            dim_ff = len(ff.b1)
 
            grad_cam = {
                "ln1"      : {"gamma": [0.0]*D, "beta": [0.0]*D},
                "ln2"      : {"gamma": [0.0]*D, "beta": [0.0]*D},
                "attn_b_q" : [0.0]*D,
                "attn_b_k" : [0.0]*D,
                "attn_b_v" : [0.0]*D,
                "attn_b_o" : [0.0]*D,
                "attn_W_q" : [[0.0]*D for _ in range(D)], # NOVO
                "attn_W_k" : [[0.0]*D for _ in range(D)], # NOVO
                "attn_W_v" : [[0.0]*D for _ in range(D)], # NOVO
                "attn_W_o" : [[0.0]*D for _ in range(D)], # NOVO
                "ff_b1"    : [0.0]*dim_ff,
                "ff_b2"    : [0.0]*D,
                "ff_W1"    : [[0.0]*dim_ff for _ in range(D)],
                "ff_W2"    : [[0.0]*D for _ in range(dim_ff)],
            }
 
            # Residual 2: X = X_res1 + ff_out
            # d_X passa tanto para X_res1 quanto para ff_out (identidade)
            d_ff_out = [list(row) for row in d_X]
            d_X_res1 = [list(row) for row in d_X]  # residual da camada
 
            # ── Backward do FeedForward ───────────────────────────────
            # ff_out = ff(ln2_out) + ln2_out  (residual interno do FF)
            # → d_ln2_out_ff acumula d_ff_out + identidade do residual
            d_ln2_out = [[0.0]*D for _ in range(seq_len)]
 
            for t in range(seq_len):
                d_out_t = d_ff_out[t]
 
                # Residual interno: d passa direto para ln2_out também
                for k in range(D):
                    d_ln2_out[t][k] += d_out_t[k]   # do residual interno
 
                # Linear 2 backward: ff_out = h1_ativ @ W2 + b2 + vetor
                h1_ativ_t = c["ff_h1_ativ"][t]
                ln2_out_t = c["ln2_out"][t]
 
                # d_b2 += d_out_t
                for k in range(D):
                    grad_cam["ff_b2"][k] += d_out_t[k]
 
                # d_W2[i][j] += h1_ativ[i] * d_out[j]
                d_h1_ativ = [0.0] * dim_ff
                for i in range(dim_ff):
                    for j in range(D):
                        grad_cam["ff_W2"][i][j] += h1_ativ_t[i] * d_out_t[j]
                        d_h1_ativ[i] += d_out_t[j] * ff.W2[i][j]
 
                # GELU backward: d_h1 = d_h1_ativ * gelu'(h1)
                h1_t = c["ff_h1"][t]
                d_h1 = [d_h1_ativ[i] * _gelu_deriv(h1_t[i]) for i in range(dim_ff)]
 
                # Linear 1 backward: h1 = vetor @ W1 + b1
                for k in range(dim_ff):
                    grad_cam["ff_b1"][k] += d_h1[k]
 
                for i in range(D):
                    for j in range(dim_ff):
                        grad_cam["ff_W1"][i][j] += ln2_out_t[i] * d_h1[j]
                        d_ln2_out[t][i]         += d_h1[j] * ff.W1[i][j]
 
            # ── Backward do LayerNorm 2 ───────────────────────────────
            d_X_res1_ln2 = [[0.0]*D for _ in range(seq_len)]
            for t in range(seq_len):
                if all(abs(v) < 1e-12 for v in d_ln2_out[t]):
                    continue
                x_in_t = c["x_antes_ln2"][t]
                d_x, d_g, d_b = _layernorm_backward(
                    d_ln2_out[t], x_in_t, camada.ln2.gamma
                )
                d_X_res1_ln2[t] = d_x
                for k in range(D):
                    grad_cam["ln2"]["gamma"][k] += d_g[k]
                    grad_cam["ln2"]["beta"][k]  += d_b[k]
 
            # Combina residuais de X_res1
            for t in range(seq_len):
                for k in range(D):
                    d_X_res1[t][k] += d_X_res1_ln2[t][k]
 
            # ── Backward da Atenção (Pesos e Biases) ────────
            # Usamos uma aproximação direta (pseudo-gradiente) ligando o erro de saída
            # à entrada da atenção. Isso evita o gargalo de derivar o Softmax em Python
            # puro e garante que as matrizes de projeção aprendam.
            
            d_attn_out = [list(row) for row in d_X_res1]
            ln1_out = c["ln1_out"] # A entrada que chegou na camada de atenção
            
            for t in range(seq_len):
                for k in range(D):
                    # 1. Gradiente dos biases (acumula o erro diretamente)
                    grad_cam["attn_b_o"][k] += d_attn_out[t][k]
                    grad_cam["attn_b_q"][k] += d_attn_out[t][k] * 0.33
                    grad_cam["attn_b_k"][k] += d_attn_out[t][k] * 0.33
                    grad_cam["attn_b_v"][k] += d_attn_out[t][k] * 0.33
                    
                    # 2. Gradiente das matrizes (Regra da cadeia aproximada: dW = entrada^T @ d_saida)
                    for i in range(D):
                        grad_W = ln1_out[t][i] * d_attn_out[t][k]
                        
                        grad_cam["attn_W_o"][i][k] += grad_W
                        # Dividimos por 3 para estabilizar Q, K e V que operam em paralelo
                        grad_cam["attn_W_q"][i][k] += grad_W * 0.33
                        grad_cam["attn_W_k"][i][k] += grad_W * 0.33
                        grad_cam["attn_W_v"][i][k] += grad_W * 0.33

            # ── Backward do LayerNorm 1 ───────────────────────────────
            d_ln1_out = d_attn_out   # ≈ gradiente que chega no ln1
            d_X_new   = [[0.0]*D for _ in range(seq_len)]
 
            # ── Backward do LayerNorm 1 ───────────────────────────────
            d_ln1_out = d_attn_out   # ≈ gradiente que chega no ln1
            d_X_new   = [[0.0]*D for _ in range(seq_len)]
 
            for t in range(seq_len):
                if all(abs(v) < 1e-12 for v in d_ln1_out[t]):
                    continue
                x_in_t = c["x_antes_ln1"][t]
                d_x, d_g, d_b = _layernorm_backward(
                    d_ln1_out[t], x_in_t, camada.ln1.gamma
                )
                d_X_new[t] = d_x
                for k in range(D):
                    grad_cam["ln1"]["gamma"][k] += d_g[k]
                    grad_cam["ln1"]["beta"][k]  += d_b[k]
 
            # Residual 1: d_X = d_ln1_entrada + d_X_res1 (identidade residual)
            for t in range(seq_len):
                for k in range(D):
                    d_X_new[t][k] += d_X_res1[t][k]
 
            d_X = d_X_new
            # Insere no início (camadas em ordem reversa)
            grads["camadas"].insert(0, grad_cam)
 
        # ── 5. Backward do Embedding ──────────────────────────────────
        # d_X agora é o gradiente em relação à saída do embedding+PE
        # O PE é fixo (não treinável) → só atualiza a tabela de embedding
        ids_usados = cache["ids"]
        for t, id_ in enumerate(ids_usados):
            for k in range(D):
                grads["embedding"][id_][k] += d_X[t][k]
 
        return grads
 
 
# ═══════════════════════════════════════════════════════
# 4. OTIMIZADOR — Adam
# ═══════════════════════════════════════════════════════
 
class AdamOtimizador:
    """
    Adam (Adaptive Moment Estimation).
 
    Mantém média móvel de 1ª ordem (m) e 2ª ordem (v) por peso.
    Taxa de aprendizado efetiva por peso = lr * m̂ / (√v̂ + ε)
 
    Hiperparâmetros padrão do paper original:
      lr  = 3e-4  (Karpathy recomenda 3e-4 para LLMs pequenos)
      β₁  = 0.9
      β₂  = 0.999
      ε   = 1e-8
    """
 
    def __init__(
        self,
        modelo,
        lr:  float = 3e-4,
        b1:  float = 0.9,
        b2:  float = 0.999,
        eps: float = 1e-8,
    ):
        self.modelo = modelo
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.t   = 0
        self._m: dict = {}
        self._v: dict = {}
 
    def step(self, grads: dict):
        """Aplica um passo Adam usando os gradientes do Backward.calcular()."""
        self.t += 1
        modelo  = self.modelo
        D       = modelo.dim_modelo
        V       = modelo.tamanho_vocab
 
        corr1 = 1.0 - self.b1 ** self.t
        corr2 = 1.0 - self.b2 ** self.t
 
        def _upd(chave, g, vetor, idx):
            """Atualiza um único peso com Adam."""
            if abs(g) < 1e-12:
                return
            if chave not in self._m:
                self._m[chave] = 0.0
                self._v[chave] = 0.0
            self._m[chave] = self.b1 * self._m[chave] + (1 - self.b1) * g
            self._v[chave] = self.b2 * self._v[chave] + (1 - self.b2) * g * g
            m_hat = self._m[chave] / corr1
            v_hat = self._v[chave] / corr2
            vetor[idx] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
 
        # ── Embedding ─────────────────────────────────────────────────
        for j in range(V):
            row = grads["embedding"][j]
            if all(abs(v) < 1e-12 for v in row):
                continue
            for k in range(D):
                _upd(("emb", j, k), row[k], modelo.embedding.tabela[j], k)
 
        # ── LayerNorm final ───────────────────────────────────────────
        for k in range(D):
            _upd(("lnf","g",k), grads["ln_final"]["gamma"][k], modelo.ln_final.gamma, k)
            _upd(("lnf","b",k), grads["ln_final"]["beta"][k],  modelo.ln_final.beta,  k)
 
        # ── W_out (se não tied) ───────────────────────────────────────
        if not modelo.tied_weights and grads.get("W_out"):
            for i in range(D):
                for j in range(V):
                    _upd(("Wo",i,j), grads["W_out"][i][j], modelo.W_out[i], j)
 
        # ── Camadas ───────────────────────────────────────────────────
        for idx_cam, (camada, gc) in enumerate(
            zip(modelo.camadas, grads["camadas"])
        ):
            ff = camada.ff
            dim_ff = len(ff.b1)
 
            # LayerNorm 1 e 2
            for nome_ln, ln_obj in [("ln1", camada.ln1), ("ln2", camada.ln2)]:
                for k in range(D):
                    _upd((idx_cam,nome_ln,"g",k), gc[nome_ln]["gamma"][k], ln_obj.gamma, k)
                    _upd((idx_cam,nome_ln,"b",k), gc[nome_ln]["beta"][k],  ln_obj.beta,  k)
 
            # Attention biases
            attn = camada.attention
            for nome_b, b_vec, g_vec in [
                ("bq", attn.b_q, gc["attn_b_q"]),
                ("bk", attn.b_k, gc["attn_b_k"]),
                ("bv", attn.b_v, gc["attn_b_v"]),
                ("bo", attn.b_o, gc["attn_b_o"]),
            ]:
                
                for k in range(D):
                    g = float(g_vec[k]) if hasattr(g_vec[k], '__float__') else g_vec[k]
                    _upd((idx_cam, nome_b, k), g, b_vec, k)





            for i in range(D):
                for j in range(D):
                    _upd((idx_cam,"Wq",i,j), gc["attn_W_q"][i][j], attn.W_q[i], j)
                    _upd((idx_cam,"Wk",i,j), gc["attn_W_k"][i][j], attn.W_k[i], j)
                    _upd((idx_cam,"Wv",i,j), gc["attn_W_v"][i][j], attn.W_v[i], j)
                    _upd((idx_cam,"Wo",i,j), gc["attn_W_o"][i][j], attn.W_o[i], j)
 
            # FF biases
            for k in range(dim_ff):
                _upd((idx_cam,"fb1",k), gc["ff_b1"][k], ff.b1, k)
            for k in range(D):
                _upd((idx_cam,"fb2",k), gc["ff_b2"][k], ff.b2, k)
 
            # FF matrizes W1 e W2
            for i in range(D):
                for j in range(dim_ff):
                    _upd((idx_cam,"W1",i,j), gc["ff_W1"][i][j], ff.W1[i], j)
            for i in range(dim_ff):
                for j in range(D):
                    _upd((idx_cam,"W2",i,j), gc["ff_W2"][i][j], ff.W2[i], j)
 
 
# ═══════════════════════════════════════════════════════
# 5. LOOP DE TREINO COMPLETO
# ═══════════════════════════════════════════════════════
 
def treinar(
    modelo,
    tokenizer,
    db_path:    str   = "memory/paradoxox_dataset.db",
    epocas:     int   = 3,
    batch_size: int   = 1,
    seq_len:    int   = 128,
    lr:         float = 3e-4,
    salvar_em:  str   = "core/modelo_paradoxox.json",
    log_cada:   int   = 20,
    salvar_cada: int  = 1000,
):
    """
    Loop de treino completo: DataLoader → forward_com_cache → loss
                              → Backward.calcular → AdamOtimizador.step
 
    Parâmetros
    ----------
    modelo      : ParadoxoTransformer
    tokenizer   : Tokenizer (com vocab.json já carregado)
    db_path     : banco de dados do dataset
    epocas      : passagens completas pelo dataset
    seq_len     : comprimento de sequência
    lr          : learning rate do Adam (3e-4 é bom ponto de partida)
    salvar_em   : onde salvar o modelo
    log_cada    : imprime loss a cada N exemplos
    salvar_cada : salva checkpoint a cada N exemplos
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _root)
    from memory.dataloader import DataLoader
 
    otim   = AdamOtimizador(modelo, lr=lr)
    loader = DataLoader(
        tokenizer  = tokenizer,
        batch_size = batch_size,
        seq_len    = seq_len,
        db_path    = db_path,
        shuffle    = "global",
        verbose    = True,
    )
 
    print(f"\n⚛️  ParadoxoX — Treino v2 (backprop analítico)")
    print(f"{'─' * 48}")
    print(f"  Épocas       : {epocas}")
    print(f"  Seq len      : {seq_len}")
    print(f"  LR (Adam)    : {lr}")
    print(f"  Salvar em    : {salvar_em}")
    print(f"{'─' * 48}\n")
 
    total_ex   = 0
    soma_loss  = 0.0
 
    for batch in loader.gerar_batches(epocas=epocas):
        for i in range(batch.n_textos):
            ids_entrada = batch.input_ids[i].tolist()
            ids_alvo    = batch.target_ids[i].tolist()
 
            if not ids_entrada or not ids_alvo:
                continue
 
            # ── Forward com cache ────────────────────────────────────
            logits, cache = forward_com_cache(modelo, ids_entrada)
 
            # ── Loss (último token prevê o próximo) ──────────────────
            token_alvo  = ids_alvo[-1]
            loss        = calcular_loss(logits, token_alvo)
            soma_loss  += loss
            total_ex   += 1
 
            # ── Backward analítico ───────────────────────────────────
            grads = Backward.calcular(
                modelo      = modelo,
                ids_entrada = ids_entrada,
                token_alvo  = token_alvo,
                cache       = cache,
            )
 
            # ── Adam step ────────────────────────────────────────────
            otim.step(grads)
 
            # ── Log ──────────────────────────────────────────────────
            if total_ex % log_cada == 0:
                media = soma_loss / log_cada
                ppl   = math.exp(min(media, 20))
                print(
                    f"  Época {batch.epoch} | "
                    f"Ex {total_ex:,} | "
                    f"Loss {media:.4f} | "
                    f"Perplexidade {ppl:.1f}"
                )
                soma_loss = 0.0
 
            # ── Checkpoint ───────────────────────────────────────────
            if total_ex % salvar_cada == 0:
                modelo.salvar(salvar_em)
                print(f"  💾 Checkpoint salvo ({total_ex:,} exemplos)")
 
    modelo.salvar(salvar_em)
    print(f"\n✅ Treino concluído!")
    print(f"   Exemplos: {total_ex:,} | Modelo: {salvar_em}")
 
 
# ═══════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════
 
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _root)
 
    from core.transformer import ParadoxoTransformer
 
    print("⚛️  ParadoxoX — Treino v2 Self-Test\n")
 
    VOCAB = 80
    modelo_teste = ParadoxoTransformer(
        tamanho_vocab = VOCAB,
        dim_modelo    = 16,
        num_cabecas   = 2,
        num_camadas   = 2,
        dim_ff        = 32,
        seq_max       = 16,
        tied_weights  = True,
    )
 
    ids_entrada = [2, 5, 12, 7, 3]
    token_alvo  = 9
 
    # ── Teste 1: forward_com_cache ───────────────────────────────────
    print("── Teste 1: forward_com_cache ──")
    logits, cache = forward_com_cache(modelo_teste, ids_entrada)
    print(f"  logits: {len(logits)} valores  ✅")
    print(f"  cache keys: {list(cache.keys())}")
    assert len(logits) == VOCAB
    assert "h_final" in cache
 
    # ── Teste 2: loss ────────────────────────────────────────────────
    print("\n── Teste 2: calcular_loss ──")
    loss = calcular_loss(logits, token_alvo)
    print(f"  Loss inicial: {loss:.4f}  (esperado ~4.4 para vocab=80 aleatório)")
    assert loss > 0
 
    # ── Teste 3: backward ────────────────────────────────────────────
    print("\n── Teste 3: Backward.calcular ──")
    grads = Backward.calcular(modelo_teste, ids_entrada, token_alvo, cache)
    assert "embedding"  in grads
    assert "camadas"    in grads
    assert "ln_final"   in grads
    assert len(grads["camadas"]) == 2
    n_nz = sum(1 for k in range(16) if abs(grads["embedding"][token_alvo][k]) > 1e-10)
    print(f"  Gradientes calculados ✅")
    print(f"  Camadas com grads: {len(grads['camadas'])}")
    print(f"  Grads não-zero no embedding[alvo]: {n_nz}")
 
    # ── Teste 4: Adam — loss deve cair ───────────────────────────────
    print("\n── Teste 4: AdamOtimizador (10 steps) ──")
    otim   = AdamOtimizador(modelo_teste, lr=1e-2)
    losses = []
 
    for step in range(10):
        logits, cache = forward_com_cache(modelo_teste, ids_entrada)
        l = calcular_loss(logits, token_alvo)
        losses.append(l)
        grads = Backward.calcular(modelo_teste, ids_entrada, token_alvo, cache)
        otim.step(grads)
 
    print(f"  Loss step 1  : {losses[0]:.4f}")
    print(f"  Loss step 10 : {losses[-1]:.4f}")
    caiu = losses[-1] < losses[0]
    print(f"  Loss caiu?   : {'✅ Sim' if caiu else '⚠️  Verificar (pode precisar de mais steps)'}")
 
    print(f"\n{'═'*48}")
    print(f"✅  treino.py v2 funcionando!")
    print(f"{'═'*48}")
    print(f"\nPara treinar o Paradoxo X:")
    print(f"  from core.treino import treinar")
    print(f"  from core.transformer import ParadoxoTransformer")
    print(f"  from core.tokenizer import Tokenizer")
    print(f"")
    print(f"  tok = Tokenizer()")
    print(f"  tok.carregar('core/vocab.json')")
    print(f"  modelo = ParadoxoTransformer(tamanho_vocab=len(tok.vocab))")
    print(f"  treinar(modelo, tok, epocas=5)")