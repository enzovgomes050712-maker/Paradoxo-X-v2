
"""
PARADOXO X — Attention
=======================
Mecanismo de atenção multi-cabeça (Multi-Head Attention).
 
O que é atenção?
  Cada token "olha" pra todos os outros e decide
  o quanto cada um importa pra ele.
 
  "o gato que comeu o rato sumiu"
   ↑                              ↑
   "sumiu" presta muita atenção em "gato" (sujeito)
   e pouca em "que", "o", "o"
 
Por que multi-cabeça?
  Cada cabeça aprende um TIPO diferente de relação:
    cabeça 1 → relação gramatical (sujeito-verbo)
    cabeça 2 → relação semântica (significado parecido)
    cabeça 3 → relação de posição (palavras próximas)
    cabeça 4 → relação de referência (pronomes)
 
Usando numpy pra performance — puro Python seria 100x mais lento.
"""
 
import math
import numpy as np
from typing import Optional
 
 
# -------------------------------------------------------
# FUNÇÕES UTILITÁRIAS (mantidas para compatibilidade)
# -------------------------------------------------------
 
def mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Multiplicação de matrizes pura Python.
    Mantida por compatibilidade — internamente usamos numpy.
    """
    a = np.array(A, dtype=np.float32)
    b = np.array(B, dtype=np.float32)
    return (a @ b).tolist()
 
 
def soma_vetores(a: list[float], b: list[float]) -> list[float]:
    """Soma dois vetores elemento a elemento."""
    return [x + y for x, y in zip(a, b)]
 
 
# -------------------------------------------------------
# ATENÇÃO ESCALADA (o núcleo)
# -------------------------------------------------------
 
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mascara: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Atenção escalada por produto escalar.
 
    Fórmula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) × V
 
    Por que dividir por sqrt(d_k)?
      Pra evitar que os produtos escalares fiquem grandes demais
      quando d_k é grande — isso faria o softmax saturar
      e os gradientes sumirem.
 
    Q: (seq_q, d_k)  — perguntas
    K: (seq_k, d_k)  — chaves
    V: (seq_k, d_v)  — valores
 
    CORREÇÃO: K.T só funciona para tensores 2-D e quebra quando
    seq_q != seq_kv (modo KV cache). Usamos K.transpose(-2, -1)
    que é seguro para qualquer número de dimensões.
    """
    d_k = Q.shape[-1]
 
    # scores: (seq_q, seq_k)
    # CORRIGIDO: era K.T — só funciona para 2-D exatos e quebrava no modo KV cache.
    # K aqui já é (seq_k, d_k) vindo de split_cabecas, então swapaxes(0,1) é
    # equivalente a .T mas explícito e seguro para qualquer shape 2-D.
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / math.sqrt(d_k)
 
    # Máscara causal: token só vê tokens anteriores (autoregressive)
    if mascara is not None:
        scores = scores + mascara  # mascara tem -inf onde não pode olhar
 
    # Softmax numericamente estável
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    pesos = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
 
    saida = pesos @ V
    return saida, pesos
 
 
def _criar_mascara_causal(seq_len: int) -> np.ndarray:
    """
    Cria a máscara triangular inferior.
    Posições futuras recebem -inf → softmax as zera.
 
    Exemplo pra seq_len=4:
      [[  0, -inf, -inf, -inf],
       [  0,    0, -inf, -inf],
       [  0,    0,    0, -inf],
       [  0,    0,    0,    0]]
    """
    mascara = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    return mascara
 
 
# -------------------------------------------------------
# MULTI-HEAD ATTENTION
# -------------------------------------------------------
 
class MultiHeadAttention:
    """
    Atenção multi-cabeça.
 
    Divide o espaço em num_cabecas sub-espaços menores,
    aplica atenção em cada um, depois junta tudo.
 
    Por que dividir?
      dim_modelo=64, num_cabecas=4 → cada cabeça usa dim=16
      Cada cabeça aprende um TIPO diferente de relação.
      Juntas, capturam relações muito mais ricas.
 
    Parâmetros:
      W_q, W_k, W_v: projeções para Query, Key, Value
      W_o: projeção de saída (junta as cabeças)
    """
 
    def __init__(self, dim_modelo: int, num_cabecas: int, seed: int = 3):
        assert dim_modelo % num_cabecas == 0, \
            f"dim_modelo ({dim_modelo}) deve ser divisível por num_cabecas ({num_cabecas})"
 
        self.dim_modelo  = dim_modelo
        self.num_cabecas = num_cabecas
        self.dim_cabeca  = dim_modelo // num_cabecas  # d_k por cabeça
 
        rng = np.random.default_rng(seed)
        escala = math.sqrt(2.0 / dim_modelo)
 
        # Pesos de projeção: (dim_modelo, dim_modelo) cada
        self.W_q = rng.normal(0, escala, (dim_modelo, dim_modelo)).astype(np.float32)
        self.W_k = rng.normal(0, escala, (dim_modelo, dim_modelo)).astype(np.float32)
        self.W_v = rng.normal(0, escala, (dim_modelo, dim_modelo)).astype(np.float32)
        self.W_o = rng.normal(0, escala, (dim_modelo, dim_modelo)).astype(np.float32)
 
        # Biases
        self.b_q = np.zeros(dim_modelo, dtype=np.float32)
        self.b_k = np.zeros(dim_modelo, dtype=np.float32)
        self.b_v = np.zeros(dim_modelo, dtype=np.float32)
        self.b_o = np.zeros(dim_modelo, dtype=np.float32)
 
    def forward(
        self,
        X: list[list[float]],
        mascara_causal: bool = True,
        kv_cache: Optional[dict] = None,
    ) -> tuple[list[list[float]], list]:
        """
        Forward pass da atenção multi-cabeça.
 
        X: sequência de vetores (seq_len, dim_modelo)
        mascara_causal: se True, tokens só veem o passado
        kv_cache: dicionário opcional pra reusar K,V computados antes
                  (acelera geração token-por-token)
 
        Retorna:
          saida: (seq_len, dim_modelo)
          pesos: lista de mapas de atenção por cabeça
        """
        X_np = np.array(X, dtype=np.float32)  # (seq, dim)
        seq_len = X_np.shape[0]
 
        # Projeções Q, K, V → (seq, dim_modelo)
        Q = X_np @ self.W_q + self.b_q
        K = X_np @ self.W_k + self.b_k
        V = X_np @ self.W_v + self.b_v
 
        # KV Cache: concatena com cache anterior se existir
        if kv_cache is not None:
            if "K" in kv_cache:
                K = np.concatenate([kv_cache["K"], K], axis=0)
                V = np.concatenate([kv_cache["V"], V], axis=0)
            kv_cache["K"] = K
            kv_cache["V"] = V
 
        seq_q  = Q.shape[0]
        seq_kv = K.shape[0]
 
        # Reshape para múltiplas cabeças: (seq, num_cabecas, dim_cabeca)
        def split_cabecas(tensor, seq):
            return tensor.reshape(seq, self.num_cabecas, self.dim_cabeca).transpose(1, 0, 2)
            # → (num_cabecas, seq, dim_cabeca)
 
        Q_h = split_cabecas(Q, seq_q)   # (H, seq_q, d_k)
        K_h = split_cabecas(K, seq_kv)  # (H, seq_kv, d_k)
        V_h = split_cabecas(V, seq_kv)  # (H, seq_kv, d_v)
 
        # Máscara causal
        # CORRIGIDO: com KV cache seq_q != seq_kv (Q é 1 token, K/V são maiores)
        # nesse caso removemos a máscara — o token final pode ver todo o histórico
        mascara = None
        if mascara_causal and seq_q == seq_kv:
            mascara = _criar_mascara_causal(seq_q)
 
        # Atenção por cabeça
        saidas_cabecas = []
        pesos_cabecas  = []
        for h in range(self.num_cabecas):
            saida_h, pesos_h = scaled_dot_product_attention(
                Q_h[h], K_h[h], V_h[h], mascara
            )  # (seq_q, d_v), (seq_q, seq_kv)
            saidas_cabecas.append(saida_h)
            pesos_cabecas.append(pesos_h.tolist())
 
        # Concat cabeças: (seq_q, dim_modelo)
        concat = np.concatenate(saidas_cabecas, axis=-1)
 
        # Projeção de saída
        saida = concat @ self.W_o + self.b_o  # (seq_q, dim_modelo)
 
        # Conexão residual
        if seq_q == seq_len:  # sem KV cache ou primeira passagem
            saida = saida + X_np
 
        return saida.tolist(), pesos_cabecas
 
    def state_dict(self) -> dict:
        """Exporta pesos para salvar."""
        return {
            "W_q": self.W_q.tolist(),
            "W_k": self.W_k.tolist(),
            "W_v": self.W_v.tolist(),
            "W_o": self.W_o.tolist(),
            "b_q": self.b_q.tolist(),
            "b_k": self.b_k.tolist(),
            "b_v": self.b_v.tolist(),
            "b_o": self.b_o.tolist(),
        }
 
    def load_state_dict(self, sd: dict):
        """Carrega pesos salvos."""
        self.W_q = np.array(sd["W_q"], dtype=np.float32)
        self.W_k = np.array(sd["W_k"], dtype=np.float32)
        self.W_v = np.array(sd["W_v"], dtype=np.float32)
        self.W_o = np.array(sd["W_o"], dtype=np.float32)
        self.b_q = np.array(sd["b_q"], dtype=np.float32)
        self.b_k = np.array(sd["b_k"], dtype=np.float32)
        self.b_v = np.array(sd["b_v"], dtype=np.float32)
        self.b_o = np.array(sd["b_o"], dtype=np.float32)