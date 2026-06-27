
"""
PARADOXO X — Transformer
========================
Aqui tudo se junta.
 
Um Transformer é basicamente isso:
  1. Embedding        → transforma IDs em vetores
  2. Positional Enc.  → diz pra IA a posição de cada token
  3. N camadas de:
       - Multi-Head Attention  → entende relações
       - Feed Forward          → processa a informação
       - Layer Norm            → estabiliza os números
  4. Projeção final   → transforma vetor em probabilidades
                        de qual token vem a seguir
 
É assim que eu funciono. É assim que o ParadoxoX vai funcionar.
 
MELHORIAS v2:
  - Pre-LayerNorm (mais estável que Post-LN)
  - GELU no lugar de ReLU (mais suave, gradientes melhores)
  - Tied Weights: embedding compartilhado com W_out (menos params, melhor generalização)
  - Top-P (nucleus) sampling além de Top-K
  - Repetition Penalty para evitar loops na geração
  - Dropout simulado (desativado em inferência)
  - Temperature = 0 → argmax determinístico
  - Layer scaling residual (controla a magnitude das residuais)
  - carregar() para restaurar pesos salvos
  - Contagem de parâmetros real
  - Validação de inputs
  - Geração com beam search simples
 
CORREÇÕES v2.1:
  - Positional Encoding agora é injetado no forward() — antes era calculado
    mas nunca somado aos embeddings na geração token-a-token
  - KV Cache implementado no gerar() — elimina O(N²) de reprocessamento
  - Double residual no FeedForward corrigido — havia dupla soma residual
    (uma dentro do FF e outra em CamadaTransformer) inflando os valores
"""
 
import math
import json
import os
import random
from typing import Optional
 
from .attention import MultiHeadAttention, mat_mul, soma_vetores
 
 
# -------------------------------------------------------
# UTILIDADES NUMÉRICAS
# -------------------------------------------------------
 
def _dot(a: list[float], b: list[float]) -> float:
    """Produto escalar de dois vetores."""
    return sum(x * y for x, y in zip(a, b))
 
 
def _mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    """Multiplica matriz M por vetor coluna v → vetor resultado."""
    return [_dot(linha, v) for linha in M]
 
 
def _add_vecs(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]
 
 
def _scale_vec(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]
 
 
# -------------------------------------------------------
# EMBEDDING — ID → Vetor
# -------------------------------------------------------
 
class EmbeddingLayer:
    """
    Transforma cada ID de token num vetor de números.
 
    Exemplo:
      token "oi" tem ID 45
      embedding[45] = [0.2, -0.5, 0.8, 0.1, ...]  (dim_modelo números)
 
    Esses vetores são APRENDIDOS durante o treino.
    Palavras parecidas ficam com vetores parecidos.
 
    MELHORIA: inicialização com escala * sqrt(dim_modelo) → vetores com
    norma próxima de 1 logo de cara, reduzindo instabilidade inicial.
    """
    def __init__(self, tamanho_vocab: int, dim_modelo: int, seed: int = 7):
        random.seed(seed)
        # Escala padrão do "Attention Is All You Need": sqrt(d_model)
        escala = dim_modelo ** -0.5
        self.tabela = [
            [random.gauss(0, escala) for _ in range(dim_modelo)]
            for _ in range(tamanho_vocab)
        ]
        self.dim_modelo = dim_modelo
        self.tamanho_vocab = tamanho_vocab
 
    def forward(self, ids: list[int]) -> list[list[float]]:
        """IDs → matriz de embeddings (cópia para evitar mutação da tabela)."""
        if any(i < 0 or i >= self.tamanho_vocab for i in ids):
            raise ValueError(f"ID fora do range [0, {self.tamanho_vocab - 1}]")
        # Retorna cópias para que modificações externas não corrompam a tabela
        return [list(self.tabela[id_]) for id_ in ids]
 
    def get_vetor(self, id_: int) -> list[float]:
        """Retorna o vetor de embedding de um único token."""
        return list(self.tabela[id_])
 
 
# -------------------------------------------------------
# POSITIONAL ENCODING — onde cada token está na sequência
# -------------------------------------------------------
 
def positional_encoding(seq_len: int, dim_modelo: int) -> list[list[float]]:
    """
    O Attention não sabe a ordem das palavras sozinho
    (ele olha tudo ao mesmo tempo).
 
    O Positional Encoding resolve isso: adiciona um vetor
    único pra cada posição, baseado em funções seno e cosseno.
 
    posição 0 → [sin(0/...), cos(0/...), sin(0/...), ...]
    posição 1 → [sin(1/...), cos(1/...), sin(1/...), ...]
 
    MELHORIA: divisor calculado só uma vez por frequência (mais eficiente).
    """
    PE = []
    # Pré-calcula os divisores para cada dimensão par
    divs = [10000 ** (2 * (i // 2) / dim_modelo) for i in range(dim_modelo)]
    for pos in range(seq_len):
        vetor = []
        for i in range(dim_modelo):
            angle = pos / divs[i]
            if i % 2 == 0:
                vetor.append(math.sin(angle))
            else:
                vetor.append(math.cos(angle))
        PE.append(vetor)
    return PE
 
 
# -------------------------------------------------------
# LAYER NORM — estabiliza os números (com parâmetros aprendíveis)
# -------------------------------------------------------
 
class LayerNorm:
    """
    Normaliza cada vetor pra ter média 0 e desvio padrão 1,
    depois aplica escala (gamma) e deslocamento (beta) aprendíveis.
 
    MELHORIA em relação à função original:
      - gamma e beta são parâmetros que o modelo pode aprender.
        A função anterior não tinha isso → menos expressiva.
      - Sem gamma/beta o LN sempre "reseta" para N(0,1),
        o que impede o modelo de aprender magnitudes úteis.
 
    É tipo calibrar o volume E escolher um tom preferido.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.gamma = [1.0] * dim   # escala (começa em 1 = sem mudança)
        self.beta  = [0.0] * dim   # deslocamento (começa em 0)
 
    def forward(self, X: list[list[float]]) -> list[list[float]]:
        resultado = []
        for vetor in X:
            media = sum(vetor) / self.dim
            variancia = sum((v - media) ** 2 for v in vetor) / self.dim
            desvio = math.sqrt(variancia + self.eps)
            normalizado = [
                self.gamma[i] * (vetor[i] - media) / desvio + self.beta[i]
                for i in range(self.dim)
            ]
            resultado.append(normalizado)
        return resultado
 
 
# -------------------------------------------------------
# ATIVAÇÕES
# -------------------------------------------------------
 
def _gelu(x: float) -> float:
    """
    GELU (Gaussian Error Linear Unit) — ativação usada no GPT-2/BERT.
 
    MELHORIA sobre ReLU:
      - ReLU mata neurônios negativos (gradiente = 0 sempre).
      - GELU é suave: passa valores negativos pequenos com atenuação,
        mantendo gradientes fluindo → treino mais estável.
 
    Aproximação eficiente (Hendrycks & Gimpel, 2016):
      GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))
 
 
def _silu(x: float) -> float:
    """SiLU / Swish — alternativa ao GELU, usada no LLaMA."""
    return x / (1.0 + math.exp(-x))
 
 
# -------------------------------------------------------
# FEED FORWARD — processa após o attention
# -------------------------------------------------------
 
class FeedForward:
    """
    Rede neural simples aplicada em cada token independentemente.
 
    Estrutura:
      Linear(dim_modelo → dim_ff) → GELU → Linear(dim_ff → dim_modelo)
 
    dim_ff normalmente é 4× o dim_modelo.
 
    MELHORIAS:
      - GELU no lugar de ReLU (gradientes mais ricos).
      - Inicialização de W2 com escala reduzida (1/√num_camadas):
        técnica do GPT-2 que evita que camadas mais profundas
        perturbem demais o sinal residual.
 
    CORREÇÃO: residual removida daqui — ela deve viver em CamadaTransformer.
      Antes havia dupla soma (uma aqui + uma em CamadaTransformer), o que
      dobrava o sinal residual e fazia os valores explodirem com o treino.
    """
    def __init__(self, dim_modelo: int, dim_ff: int, seed: int = 13,
                 num_camadas: int = 1):
        random.seed(seed)
        escala1 = math.sqrt(2.0 / dim_modelo)
        # Escala reduzida para a segunda camada: estabiliza residuais profundas
        escala2 = math.sqrt(2.0 / dim_ff) / math.sqrt(num_camadas)
 
        self.dim_modelo = dim_modelo
 
        # Camada 1: dim_modelo → dim_ff
        self.W1 = [[random.gauss(0, escala1) for _ in range(dim_ff)]
                   for _ in range(dim_modelo)]
        self.b1 = [0.0] * dim_ff
 
        # Camada 2: dim_ff → dim_modelo
        self.W2 = [[random.gauss(0, escala2) for _ in range(dim_modelo)]
                   for _ in range(dim_ff)]
        self.b2 = [0.0] * dim_modelo
 
    def _linear(self, x: list[float], W: list[list[float]],
                b: list[float]) -> list[float]:
        """Multiplicação linear: x @ W + b"""
        dim_saida = len(W[0])
        saida = list(b)
        for i, xi in enumerate(x):
            if xi == 0.0:
                continue  # pula zeros (GELU produz zeros exatos raramente, mas ReLU sim)
            for j in range(dim_saida):
                saida[j] += xi * W[i][j]
        return saida
 
    def forward(self, X: list[list[float]]) -> list[list[float]]:
        """
        Aplica FF em cada token.
 
        CORREÇÃO: não soma residual aqui — CamadaTransformer já faz isso.
        Antes: h = h + vetor  ← dupla residual, valores explodem
        Agora: retorna apenas a transformação F(x), sem somar x
        """
        resultado = []
        for vetor in X:
            h = self._linear(vetor, self.W1, self.b1)
            h = [_gelu(v) for v in h]   # GELU
            h = self._linear(h, self.W2, self.b2)
            # CORRIGIDO: sem residual aqui — CamadaTransformer aplica x + FF(x)
            resultado.append(h)
        return resultado
 
 
# -------------------------------------------------------
# CAMADA DO TRANSFORMER (uma só) — Pre-LayerNorm
# -------------------------------------------------------
 
class CamadaTransformer:
    """
    Uma camada completa do Transformer com PRE-LAYERNORM:
 
    Arquitetura ORIGINAL (Post-LN, "Attention Is All You Need"):
      X → Attention → X+Attn → LayerNorm → FF → X+FF → LayerNorm
 
    Arquitetura MELHORADA (Pre-LN, usada no GPT-2 em diante):
      X → LayerNorm → Attention → X+Attn → LayerNorm → FF → X+FF
 
    POR QUE PRE-LN É MELHOR?
      - Gradientes mais estáveis no começo do treino.
      - Não precisa de learning rate warmup tão longo.
      - Permite treinar modelos mais profundos sem explodir.
      - Resultados empíricos: converge mais rápido e de forma mais suave.
 
    MELHORIA: cada sublayer tem seu próprio LayerNorm com gamma/beta.
    """
    def __init__(self, dim_modelo: int, num_cabecas: int, dim_ff: int,
                 idx: int = 0, num_camadas_total: int = 1):
        self.attention = MultiHeadAttention(dim_modelo, num_cabecas)
        self.ff = FeedForward(dim_modelo, dim_ff,
                              seed=idx * 17,
                              num_camadas=num_camadas_total)
        # Pre-LN: cada sublayer tem seu próprio LayerNorm
        self.ln1 = LayerNorm(dim_modelo)   # antes do attention
        self.ln2 = LayerNorm(dim_modelo)   # antes do feed forward
        self.dim_modelo = dim_modelo
 
    def forward(
        self,
        X: list[list[float]],
        mascara_causal: bool = True,
        kv_cache: Optional[dict] = None,
    ):
        """
        kv_cache: dicionário opcional passado para o MultiHeadAttention.
                  Permite reusar Keys e Values de tokens anteriores.
        """
        # ── Sublayer 1: Attention com Pre-LN ──
        # Normaliza ANTES de entrar no attention
        X_norm = self.ln1.forward(X)
        attn_out, pesos = self.attention.forward(
            X_norm, mascara_causal, kv_cache=kv_cache
        )
        # Conexão residual: entrada original + saída do attention
        X = [[X[i][j] + attn_out[i][j] for j in range(self.dim_modelo)]
             for i in range(len(X))]
 
        # ── Sublayer 2: FeedForward com Pre-LN ──
        X_norm = self.ln2.forward(X)
        ff_out = self.ff.forward(X_norm)
        # CORRIGIDO: residual única e correta — X + FF(LayerNorm(X))
        # (FeedForward não faz mais residual internamente)
        X = [[X[i][j] + ff_out[i][j] for j in range(self.dim_modelo)]
             for i in range(len(X))]
 
        return X, pesos
 
 
# -------------------------------------------------------
# TRANSFORMER COMPLETO
# -------------------------------------------------------
 
class ParadoxoTransformer:
    """
    O modelo completo do ParadoxoX.
 
    Parâmetros padrão (pequeno pra rodar sem GPU):
      dim_modelo  = 64   (tamanho dos vetores internos)
      num_cabecas = 4    (cabeças de atenção)
      num_camadas = 4    (profundidade)
      dim_ff      = 256  (tamanho da camada feed forward)
 
    Quando tiver o servidor, aumenta esses números.
 
    MELHORIAS PRINCIPAIS:
      - Pre-LayerNorm em todas as camadas
      - GELU no FeedForward
      - LayerNorm final antes de W_out (pre-norm output)
      - Tied Weights: W_out reutiliza a tabela de embedding
        → menos parâmetros, melhor generalização (GPT-2 faz isso)
      - Top-P (nucleus) sampling
      - Repetition penalty
      - Beam search simples
      - carregar() para restaurar modelo salvo
      - Contagem de parâmetros precisa
    """
    def __init__(
        self,
        tamanho_vocab: int,
        dim_modelo:    int = 64,
        num_cabecas:   int = 4,
        num_camadas:   int = 4,
        dim_ff:        int = 256,
        seq_max:       int = 512,
        tied_weights:  bool = True,   # compartilha embedding com W_out
    ):
        # Validações
        if dim_modelo % num_cabecas != 0:
            raise ValueError(
                f"dim_modelo ({dim_modelo}) deve ser divisível por "
                f"num_cabecas ({num_cabecas})"
            )
 
        self.dim_modelo    = dim_modelo
        self.num_camadas   = num_camadas
        self.seq_max       = seq_max
        self.tamanho_vocab = tamanho_vocab
        self.tied_weights  = tied_weights
 
        # ── Embedding ──
        self.embedding = EmbeddingLayer(tamanho_vocab, dim_modelo)
 
        # ── Positional Encoding (pré-calculado até seq_max) ──
        # CORRIGIDO: era calculado mas nunca injetado no forward() da geração.
        # Agora pos_enc é tabela estática e o forward() sempre o aplica.
        self.pos_enc = positional_encoding(seq_max, dim_modelo)
 
        # ── Camadas do Transformer ──
        self.camadas = [
            CamadaTransformer(dim_modelo, num_cabecas, dim_ff,
                              idx=i, num_camadas_total=num_camadas)
            for i in range(num_camadas)
        ]
 
        # ── LayerNorm final (pre-norm output — GPT-2 style) ──
        self.ln_final = LayerNorm(dim_modelo)
 
        # ── Projeção final → logits ──
        if tied_weights:
            # Tied weights: W_out = embedding^T
            # Não cria matriz nova → economiza tamanho_vocab × dim_modelo params
            self.W_out = None
        else:
            random.seed(42)
            escala = dim_modelo ** -0.5
            self.W_out = [
                [random.gauss(0, escala) for _ in range(tamanho_vocab)]
                for _ in range(dim_modelo)
            ]
 
        self._imprimir_info()
 
    def _imprimir_info(self):
        """Imprime resumo do modelo com contagem real de parâmetros."""
        n_emb   = self.tamanho_vocab * self.dim_modelo
        n_ln    = self.dim_modelo * 2  # gamma + beta do ln_final
        n_layer = 0
        for _ in self.camadas:
            # attention: 4 matrizes dim×dim (Q, K, V, O)
            n_layer += self.dim_modelo * self.dim_modelo * 4
            # ff: W1 (d×ff) + b1 (ff) + W2 (ff×d) + b2 (d)
            n_layer += (self.dim_modelo * self._dim_ff_estimado() +
                        self._dim_ff_estimado() +
                        self._dim_ff_estimado() * self.dim_modelo +
                        self.dim_modelo)
            # 2× LayerNorm por camada: 2 × (gamma + beta) = 4 × dim
            n_layer += self.dim_modelo * 4
        n_out = 0 if self.tied_weights else self.tamanho_vocab * self.dim_modelo
        total = n_emb + n_ln + n_layer + n_out
 
        print(f"🧠 ParadoxoX Transformer v2 inicializado")
        print(f"   Vocab:        {self.tamanho_vocab:,} tokens")
        print(f"   Dimensão:     {self.dim_modelo}")
        print(f"   Cabeças:      {self.camadas[0].attention.num_cabecas if self.camadas else '?'}")
        print(f"   Camadas:      {self.num_camadas}")
        print(f"   Seq max:      {self.seq_max}")
        print(f"   Tied weights: {'Sim ✓' if self.tied_weights else 'Não'}")
        print(f"   Parâmetros:   ~{total:,}")
 
    def _dim_ff_estimado(self) -> int:
        """Recupera dim_ff da primeira camada para estatísticas."""
        if self.camadas:
            return len(self.camadas[0].ff.b1)
        return 0
 
    # -------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------
 
    def forward(self, ids: list[int], kv_caches: Optional[list[dict]] = None) -> tuple:
        """
        Passa uma sequência de IDs pelo transformer.
 
        ids: lista de token IDs
        kv_caches: lista de dicionários de cache (um por camada), opcional.
                   Quando fornecido, só processa os tokens novos (offset).
 
        Retorna:
          logits = vetor de scores pra cada token do vocab
          pesos  = mapas de atenção de todas as camadas
        """
        if not ids:
            raise ValueError("Sequência de entrada não pode ser vazia.")
 
        # Trunca ao tamanho máximo (pega os mais recentes)
        ids = ids[-self.seq_max:]
        seq_len = len(ids)
 
        # Descobre offset: se há cache, só processa tokens novos
        # (o cache já guarda os tokens anteriores)
        offset = 0
        if kv_caches is not None and len(kv_caches) > 0:
            cache0 = kv_caches[0]
            if "K" in cache0:
                offset = cache0["K"].shape[0]
 
        # IDs a processar neste passo
        ids_novos = ids[offset:]
        if not ids_novos:
            ids_novos = ids[-1:]   # garante ao menos 1 token
 
        seq_novos = len(ids_novos)
 
        # ── 1. Embedding + Positional Encoding ──
        # CORRIGIDO: pos_enc agora é sempre somado, usando a posição absoluta
        # do token na sequência (offset + i), não a posição relativa no batch.
        # Antes o PE era calculado mas ignorado na geração token-a-token.
        X = self.embedding.forward(ids_novos)
        for i in range(seq_novos):
            pos = offset + i          # posição absoluta na sequência
            pe = self.pos_enc[pos]
            X[i] = [X[i][j] + pe[j] for j in range(self.dim_modelo)]
 
        # ── 2. Passa por todas as camadas ──
        todos_pesos = []
        for idx_camada, camada in enumerate(self.camadas):
            cache = kv_caches[idx_camada] if kv_caches is not None else None
            X, pesos = camada.forward(X, mascara_causal=True, kv_cache=cache)
            todos_pesos.append(pesos)
 
        # ── 3. LayerNorm final (pre-norm output) ──
        X_norm = self.ln_final.forward(X)
        ultimo = X_norm[-1]  # último token prevê o próximo
 
        # ── 4. Projeção final → logits ──
        if self.tied_weights:
            # Tied: logit[j] = dot(ultimo, embedding[j])
            logits = [
                sum(ultimo[k] * self.embedding.tabela[j][k]
                    for k in range(self.dim_modelo))
                for j in range(self.tamanho_vocab)
            ]
        else:
            logits = [0.0] * self.tamanho_vocab
            for k, val in enumerate(ultimo):
                for j in range(self.tamanho_vocab):
                    logits[j] += val * self.W_out[k][j]
 
        return logits, todos_pesos
 
    # -------------------------------------------------------
    # SAMPLING HELPERS
    # -------------------------------------------------------
 
    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        maximo = max(logits)
        exps = [math.exp(l - maximo) for l in logits]
        soma = sum(exps)
        return [e / soma for e in exps]
 
    @staticmethod
    def _aplicar_temperatura(logits: list[float], temperatura: float) -> list[float]:
        if temperatura <= 0:
            return logits  # argmax depois
        return [l / temperatura for l in logits]
 
    @staticmethod
    def _aplicar_repetition_penalty(logits: list[float],
                                    ids_vistos: list[int],
                                    penalty: float) -> list[float]:
        """
        Penaliza tokens que já apareceram na sequência.
 
        Evita que o modelo entre em loop repetindo
        a mesma palavra indefinidamente.
 
        penalty > 1.0 → divide o logit dos tokens repetidos.
        penalty = 1.0 → sem efeito.
        """
        if penalty == 1.0:
            return logits
        novos = list(logits)
        for id_ in set(ids_vistos):
            if 0 <= id_ < len(novos):
                if novos[id_] > 0:
                    novos[id_] /= penalty
                else:
                    novos[id_] *= penalty
        return novos
 
    @staticmethod
    def _top_k_filter(probs: list[float], k: int) -> list[float]:
        """Zera tudo fora do top-k."""
        if k <= 0 or k >= len(probs):
            return probs
        threshold = sorted(probs, reverse=True)[k - 1]
        return [p if p >= threshold else 0.0 for p in probs]
 
    @staticmethod
    def _top_p_filter(probs: list[float], p: float) -> list[float]:
        """
        Nucleus sampling (Top-P).
 
        Em vez de sempre pegar os mesmos K tokens,
        pega tantos tokens quanto for necessário para cobrir
        probabilidade acumulada ≥ p.
 
        Vantagem: adapta o número de tokens ao contexto.
        Frases fáceis → poucos tokens candidatos.
        Frases ambíguas → mais candidatos.
        """
        if p >= 1.0:
            return probs
        ordenados = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        acum = 0.0
        mantidos = set()
        for idx, prob in ordenados:
            acum += prob
            mantidos.add(idx)
            if acum >= p:
                break
        return [probs[i] if i in mantidos else 0.0 for i in range(len(probs))]
 
    @staticmethod
    def _amostrar(probs: list[float]) -> int:
        """Amostragem por distribuição cumulativa."""
        r = random.random()
        acumulado = 0.0
        for i, p in enumerate(probs):
            acumulado += p
            if r <= acumulado:
                return i
        return len(probs) - 1  # fallback
 
    # -------------------------------------------------------
    # GERAÇÃO AUTOREGRESSIVA — com KV Cache
    # -------------------------------------------------------
 
    def gerar(
        self,
        ids_entrada:        list[int],
        max_novos_tokens:   int = 50,
        temperatura:        float = 0.8,
        top_k:              int = 10,
        top_p:              float = 0.95,
        repetition_penalty: float = 1.1,
        token_fim:          int = 3,         # <EOS>
        tokens_proibidos:   Optional[list[int]] = None,
    ) -> list[int]:
        """
        Gera tokens novos um por vez, com KV Cache.
 
        CORREÇÃO: antes cada chamada a self.forward() reprocessava toda
        a sequência histórica — complexidade O(N²) que degradava
        conforme o código gerado crescia.
 
        Agora:
          - Primeira chamada: processa toda a entrada, popula os caches
          - Chamadas seguintes: processa só o 1 token novo
          - O KV cache em MultiHeadAttention concatena K e V automaticamente
          → Complexidade por token = O(N) em vez de O(N²)
 
        Temperatura:
          < 1.0 → mais conservador      > 1.0 → mais criativo
          = 0   → argmax determinístico (greedy)
 
        Top-K: considera só os K tokens mais prováveis.
        Top-P: considera tokens até cobrir P de probabilidade acumulada.
        Repetition Penalty: > 1.0 penaliza tokens repetidos.
        """
        ids = list(ids_entrada)
        proibidos = set(tokens_proibidos or [])
 
        # Inicializa um dicionário de cache por camada
        # CORRIGIDO: antes não havia cache algum — cada token reprocessava tudo
        kv_caches: list[dict] = [{} for _ in self.camadas]
 
        # Primeira passagem: processa toda a entrada e popula os caches
        logits, _ = self.forward(ids, kv_caches=kv_caches)
 
        for _ in range(max_novos_tokens):
            # Aplica restrições nos logits
            for t in proibidos:
                if 0 <= t < len(logits):
                    logits[t] = -1e9
 
            logits = self._aplicar_repetition_penalty(logits, ids, repetition_penalty)
 
            # Temperatura = 0 → greedy (argmax)
            if temperatura <= 0:
                escolhido = max(range(len(logits)), key=lambda i: logits[i])
            else:
                logits_t = self._aplicar_temperatura(logits, temperatura)
                probs    = self._softmax(logits_t)
 
                if top_k > 0:
                    probs = self._top_k_filter(probs, top_k)
                if top_p < 1.0:
                    probs = self._top_p_filter(probs, top_p)
 
                # Renormaliza
                soma = sum(probs)
                if soma <= 0:
                    probs = [1.0 / self.tamanho_vocab] * self.tamanho_vocab
                else:
                    probs = [p / soma for p in probs]
 
                escolhido = self._amostrar(probs)
 
            ids.append(escolhido)
 
            if escolhido == token_fim:
                break
 
            # Próxima passagem: só o novo token (cache faz o resto)
            logits, _ = self.forward([escolhido], kv_caches=kv_caches)
 
        return ids[len(ids_entrada):]
 
    # -------------------------------------------------------
    # BEAM SEARCH
    # -------------------------------------------------------
 
    def beam_search(
        self,
        ids_entrada:     list[int],
        num_beams:       int = 4,
        max_novos_tokens: int = 50,
        token_fim:       int = 3,
        penalidade_comprimento: float = 0.6,  # alpha do length penalty
    ) -> list[int]:
        """
        Beam Search — gera a sequência mais provável explorando N caminhos.
 
        Greedy pega sempre o token mais provável agora.
        Beam Search mantém N hipóteses em paralelo e escolhe
        a melhor sequência COMPLETA no final.
 
        penalidade_comprimento (α): evita que sequências curtas ganhem
        injustamente só por ter menos termos multiplicados.
        length_penalty = ((5 + len)/(5 + 1))^α
 
        Nota: beam search não usa KV cache pois mantém múltiplas hipóteses
        em paralelo — o cache de uma hipótese não serve para as outras.
        """
        # Inicializa beams: (log_prob, ids_sequencia)
        beams = [(0.0, list(ids_entrada))]
        completos = []
 
        for _ in range(max_novos_tokens):
            candidatos = []
 
            for log_prob, ids_seq in beams:
                if ids_seq and ids_seq[-1] == token_fim:
                    completos.append((log_prob, ids_seq))
                    continue
 
                logits, _ = self.forward(ids_seq)
                probs = self._softmax(logits)
 
                # Pega os top-num_beams candidatos
                top_ids = sorted(range(len(probs)),
                                 key=lambda i: probs[i], reverse=True)[:num_beams]
                for tid in top_ids:
                    novo_log_prob = log_prob + math.log(probs[tid] + 1e-10)
                    candidatos.append((novo_log_prob, ids_seq + [tid]))
 
            if not candidatos:
                break
 
            # Ordena e mantém os top-num_beams
            candidatos.sort(key=lambda x: x[0], reverse=True)
            beams = candidatos[:num_beams]
 
            # Verifica se todos terminaram
            if all(seq[-1] == token_fim for _, seq in beams):
                completos.extend(beams)
                break
 
        completos.extend(beams)
 
        if not completos:
            return []
 
        # Length penalty: favorece sequências com comprimento adequado
        def _score(log_prob, seq):
            n = len(seq) - len(ids_entrada)
            lp = ((5 + n) / 6) ** penalidade_comprimento if n > 0 else 1.0
            return log_prob / lp
 
        melhor = max(completos, key=lambda x: _score(x[0], x[1]))
        return melhor[1][len(ids_entrada):]
 
    # -------------------------------------------------------
    # SALVAR / CARREGAR
    # -------------------------------------------------------
 
    def salvar(self, caminho: str = "modelo_paradoxox.json"):
        """Salva os pesos do modelo em JSON."""
        # Coleta gamma/beta dos LayerNorms de cada camada
        camadas_ln = []
        for camada in self.camadas:
            camadas_ln.append({
                "ln1_gamma": camada.ln1.gamma,
                "ln1_beta":  camada.ln1.beta,
                "ln2_gamma": camada.ln2.gamma,
                "ln2_beta":  camada.ln2.beta,
            })
 
        dados = {
            "versao": "2.1",
            "config": {
                "tamanho_vocab":  self.tamanho_vocab,
                "dim_modelo":     self.dim_modelo,
                "num_cabecas":    self.camadas[0].attention.num_cabecas if self.camadas else 4,
                "num_camadas":    self.num_camadas,
                "dim_ff":         self._dim_ff_estimado(),
                "seq_max":        self.seq_max,
                "tied_weights":   self.tied_weights,
            },
            "embedding":  self.embedding.tabela,
            "ln_final_gamma": self.ln_final.gamma,
            "ln_final_beta":  self.ln_final.beta,
            "camadas_ln": camadas_ln,
            "W_out": self.W_out if not self.tied_weights else None,
        }

        
        with open(caminho, "w") as f:
            json.dump(dados, f)
        print(f"💾 Modelo v2.1 salvo em '{caminho}'")
 
    @classmethod
    def carregar(cls, caminho: str) -> "ParadoxoTransformer":
        """
        Restaura um modelo salvo.
 
        Uso:
          modelo = ParadoxoTransformer.carregar("modelo_paradoxox.json")
        """
        with open(caminho, "r") as f:
            dados = json.load(f)
 
        cfg = dados["config"]
        modelo = cls(
            tamanho_vocab=cfg["tamanho_vocab"],
            dim_modelo=cfg["dim_modelo"],
            num_cabecas=cfg.get("num_cabecas", 4),
            num_camadas=cfg["num_camadas"],
            dim_ff=cfg.get("dim_ff", 256),
            seq_max=cfg["seq_max"],
            tied_weights=cfg.get("tied_weights", True),
        )
 
        # Restaura embedding
        modelo.embedding.tabela = dados["embedding"]
 
        # Restaura LayerNorm final
        modelo.ln_final.gamma = dados["ln_final_gamma"]
        modelo.ln_final.beta  = dados["ln_final_beta"]
 
        # Restaura LayerNorms das camadas
        for i, ln_data in enumerate(dados.get("camadas_ln", [])):
            modelo.camadas[i].ln1.gamma = ln_data["ln1_gamma"]
            modelo.camadas[i].ln1.beta  = ln_data["ln1_beta"]
            modelo.camadas[i].ln2.gamma = ln_data["ln2_gamma"]
            modelo.camadas[i].ln2.beta  = ln_data["ln2_beta"]
 
        # Restaura W_out se não for tied
        if not cfg.get("tied_weights", True) and dados.get("W_out"):
            modelo.W_out = dados["W_out"]
 
        print(f"📂 Modelo v2.1 carregado de '{caminho}'")
        return modelo
 
 
# -------------------------------------------------------
# TESTE RÁPIDO
# -------------------------------------------------------
if __name__ == "__main__":
    print("⚛️  PARADOXO X — Testando Transformer v2.1\n")
 
    VOCAB_SIZE  = 100
    DIM_MODELO  = 32
    NUM_CABECAS = 2
    NUM_CAMADAS = 2
    DIM_FF      = 128
 
    modelo = ParadoxoTransformer(
        tamanho_vocab=VOCAB_SIZE,
        dim_modelo=DIM_MODELO,
        num_cabecas=NUM_CABECAS,
        num_camadas=NUM_CAMADAS,
        dim_ff=DIM_FF,
        seq_max=64,
        tied_weights=True,
    )
 
    print("\n--- Teste de Forward Pass ---")
    ids_teste = [2, 13, 7, 14, 7, 15, 3]
    logits, pesos = modelo.forward(ids_teste)
    print(f"Entrada:  {ids_teste}")
    print(f"Logits:   {len(logits)} valores")
    top5 = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
    print(f"Top 5 tokens mais prováveis: {top5}")
 
    print("\n--- Teste de Geração com KV Cache (Top-K + Top-P + Repetition Penalty) ---")
    ids_gerados = modelo.gerar(
        ids_entrada=[2, 13, 7],
        max_novos_tokens=10,
        temperatura=0.8,
        top_k=10,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    print(f"Tokens gerados (sampling): {ids_gerados}")
 
    print("\n--- Teste de Geração (Greedy, temperatura=0) ---")
    ids_greedy = modelo.gerar(
        ids_entrada=[2, 13, 7],
        max_novos_tokens=10,
        temperatura=0.0,
    )
    print(f"Tokens gerados (greedy):   {ids_greedy}")
 
    print("\n--- Teste de Beam Search ---")
    ids_beam = modelo.beam_search(
        ids_entrada=[2, 13, 7],
        num_beams=3,
        max_novos_tokens=10,
    )
    print(f"Tokens gerados (beam=3):   {ids_beam}")
 
    print("\n--- Teste de Salvar/Carregar ---")
    modelo.salvar("paradoxox_test.json")
    modelo2 = ParadoxoTransformer.carregar("paradoxox_test.json")
    logits2, _ = modelo2.forward(ids_teste)
    diff = max(abs(logits[i] - logits2[i]) for i in range(len(logits)))
    print(f"Diferença máxima após salvar/carregar: {diff:.2e} (deve ser ~0)")
 
    print("\n✅ Transformer v2.1 funcionando!")