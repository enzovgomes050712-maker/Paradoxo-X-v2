"""
PARADOXO X — Brain
===================
O cérebro do ParadoxoX. Conecta todos os módulos.

Fluxo de uma mensagem:
  1. Usuário fala algo
  2. Brain detecta a INTENÇÃO (analisar? corrigir? criar? conversar?)
  3. Chama o módulo certo (analyzer, refactor, ou conversa)
  4. Salva tudo na memory
  5. Devolve a resposta

Intenções reconhecidas:
  ANALISAR   → "analisa isso", "o que tem de errado", "revisa"
  CORRIGIR   → "corrige", "arruma", "conserta"
  MELHORAR   → "melhora", "refatora", "otimiza"
  CRIAR      → "cria", "faz", "gera", "escreve um código"
  CONVERSA   → qualquer outra coisa (resposta via templates + contexto)

Por enquanto as respostas de CONVERSA são via templates inteligentes.
Quando o transformer tiver treino real, troca essa parte e o resto
continua funcionando igual — o brain não precisa mudar.

MELHORIAS v2:
  - DetectorIntencao: scores ponderados por peso (padrões mais específicos
    valem mais), desambiguação por presença de código, fallback menos agressivo.
  - DetectorIntencao: suporte a intenção "explicar" ("explica", "como funciona").
  - DetectorIntencao: detecção de linguagem de programação no texto.
  - GeradorResposta: respostas de erro/frustração ("não tá funcionando").
  - GeradorResposta: reconhece pedidos de ajuda vagos e pede contexto certeiro.
  - GeradorResposta: histórico de conversa evita repetir a mesma resposta.
  - ParadoxoBrain: carregamento via ParadoxoTransformer.carregar() (v2).
  - ParadoxoBrain: _handle_analisar aceita código vindo da memory.
  - ParadoxoBrain: _handle_criar aceita linguagem detectada pelo detector.
  - ParadoxoBrain: novo comando /explicar e intenção "explicar".
  - ParadoxoBrain: mensagens de entrada validadas e sanitizadas.
  - ParadoxoBrain: proteção contra crash nos handlers (try/except específico).
  - ParadoxoBrain: método chat() retorna dict opcional com metadados.
  - main(): suporte a /explicar, /limpar, exibição de confiança no debug.
"""

import re
import os
import sys
import random
import json
import math
from typing import Optional, List, Dict
# Acrescente no __init__ do ParadoxoBrain:


# --- CONFIGURAÇÃO DE CAMINHO ---
_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- IMPORTS ---
from code_engine.analyzer import CodeAnalyzer
from code_engine.refactor import CodeRefactor
from memory.Memory import MemoryManager
from core.tokenizer import Tokenizer
from core.transformer import ParadoxoTransformer
from core.vision import VisionProcessor



# -------------------------------------------------------
# CONSTANTES GLOBAIS
# -------------------------------------------------------

# Mapeamento de extensão → nome de linguagem (para detecção)
_EXT_PARA_LANG = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "java": "java", "go": "go", "rs": "rust", "cpp": "cpp",
    "c": "c", "cs": "csharp", "rb": "ruby", "php": "php",
    "swift": "swift", "kt": "kotlin", "sh": "bash", "sql": "sql",
}

# Palavras que indicam linguagem no texto
_PALAVRAS_LANG = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript", "node": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "java": "java",
    "golang": "go", "go": "go",
    "rust": "rust",
    "c++": "cpp", "cpp": "cpp",
    "c#": "csharp", "csharp": "csharp",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "bash": "bash", "shell": "bash",
    "sql": "sql",
}


# -------------------------------------------------------
# RAG — Retrieval-Augmented Generation
# -------------------------------------------------------

class RAGEngine:
    """
    Motor de RAG (Retrieval-Augmented Generation) para o ParadoxoX.

    Mantém uma base de documentos indexados por TF-IDF simples
    (sem dependências externas). Ao receber uma consulta, recupera
    os K trechos mais relevantes e os injeta no contexto antes de
    chamar o modelo ou o handler de intenção.

    Uso:
        rag = RAGEngine()
        rag.indexar("nome_doc", conteudo_texto)
        trechos = rag.recuperar("como funciona a função X?", top_k=3)
        # trechos → lista de dicts {"doc": str, "trecho": str, "score": float}

    A base pode ser salva/carregada em JSON para persistir entre sessões:
        rag.salvar("memory/rag_index.json")
        rag.carregar("memory/rag_index.json")
    """

    def __init__(self, tamanho_trecho: int = 300, passo_trecho: int = 150):
        """
        Args:
            tamanho_trecho: caracteres por trecho ao indexar documentos.
            passo_trecho:   salto entre trechos (< tamanho → sobreposição).
        """
        self.tamanho_trecho = tamanho_trecho
        self.passo_trecho   = passo_trecho

        # índice: lista de {"doc": str, "trecho": str, "tf": dict}
        self._index: List[Dict] = []
        # frequência de documento por token (para IDF)
        self._df: Dict[str, int] = {}
        # número total de trechos
        self._n_docs: int = 0

    # ── Indexação ───────────────────────────────────────────

    def indexar(self, nome_doc: str, conteudo: str) -> int:
        """
        Divide *conteudo* em trechos sobrepostos e os adiciona ao índice.

        Retorna o número de trechos adicionados.
        """
        trechos = self._dividir_em_trechos(conteudo)
        adicionados = 0
        for trecho in trechos:
            tf = self._calcular_tf(trecho)
            self._index.append({"doc": nome_doc, "trecho": trecho, "tf": tf})
            # Atualiza DF: cada token conta uma vez por trecho
            for token in set(tf.keys()):
                self._df[token] = self._df.get(token, 0) + 1
            adicionados += 1
        self._n_docs += adicionados
        return adicionados

    def remover_doc(self, nome_doc: str) -> int:
        """Remove todos os trechos de um documento do índice."""
        antes = len(self._index)
        removidos = [e for e in self._index if e["doc"] == nome_doc]
        self._index = [e for e in self._index if e["doc"] != nome_doc]
        # Recalcula DF do zero após remoção
        self._df = {}
        for entrada in self._index:
            for token in set(entrada["tf"].keys()):
                self._df[token] = self._df.get(token, 0) + 1
        self._n_docs = len(self._index)
        return antes - len(self._index)

    def limpar(self):
        """Apaga toda a base de conhecimento."""
        self._index.clear()
        self._df.clear()
        self._n_docs = 0

    # ── Recuperação ─────────────────────────────────────────

    def recuperar(self, consulta: str, top_k: int = 3) -> List[Dict]:
        """
        Retorna os *top_k* trechos mais relevantes para *consulta*.

        Cada item do resultado é um dict:
            {"doc": str, "trecho": str, "score": float}

        Retorna lista vazia se o índice estiver vazio.
        """
        if not self._index:
            return []

        tf_consulta = self._calcular_tf(consulta)
        if not tf_consulta:
            return []

        scores = []
        for entrada in self._index:
            score = self._similaridade_tfidf(tf_consulta, entrada["tf"])
            if score > 0:
                scores.append({
                    "doc":    entrada["doc"],
                    "trecho": entrada["trecho"],
                    "score":  round(score, 4),
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def contexto_para_prompt(self, consulta: str, top_k: int = 3) -> str:
        """
        Versão conveniente de recuperar(): devolve uma string pronta
        para ser concatenada ao prompt do modelo.

        Formato:
            [RAG]
            [doc: nome_arquivo] trecho relevante...

            [doc: outro_arquivo] outro trecho...
        """
        resultados = self.recuperar(consulta, top_k=top_k)
        if not resultados:
            return ""
        linhas = ["[RAG]"]
        for r in resultados:
            linhas.append(f"[doc: {r['doc']}] {r['trecho']}")
            linhas.append("")   # linha em branco entre trechos
        return "\n".join(linhas).strip()

    # ── Persistência ────────────────────────────────────────

    def salvar(self, caminho: str) -> bool:
        """Salva o índice completo em JSON. Retorna True se bem-sucedido."""
        try:
            os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
            with open(caminho, "w", encoding="utf-8") as f:
                json.dump({
                    "tamanho_trecho": self.tamanho_trecho,
                    "passo_trecho":   self.passo_trecho,
                    "df":             self._df,
                    "n_docs":         self._n_docs,
                    "index":          self._index,
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def carregar(self, caminho: str) -> bool:
        """Carrega índice de um JSON salvo. Retorna True se bem-sucedido."""
        if not os.path.exists(caminho):
            return False
        try:
            with open(caminho, encoding="utf-8") as f:
                dados = json.load(f)
            self.tamanho_trecho = dados.get("tamanho_trecho", self.tamanho_trecho)
            self.passo_trecho   = dados.get("passo_trecho",   self.passo_trecho)
            self._df            = dados.get("df",    {})
            self._n_docs        = dados.get("n_docs", 0)
            self._index         = dados.get("index",  [])
            return True
        except Exception:
            return False

    # ── Internos ────────────────────────────────────────────

    def _dividir_em_trechos(self, texto: str) -> List[str]:
        """Divide texto em trechos sobrepostos."""
        trechos = []
        inicio = 0
        while inicio < len(texto):
            fim = inicio + self.tamanho_trecho
            trechos.append(texto[inicio:fim].strip())
            if fim >= len(texto):
                break
            inicio += self.passo_trecho
        return [t for t in trechos if t]

    @staticmethod
    def _tokenizar(texto: str) -> List[str]:
        """Tokenização simples: lowercase + split por não-alfanumérico."""
        return re.findall(r'[a-záéíóúàâêôãõüñçA-Z0-9_]+', texto.lower())

    def _calcular_tf(self, texto: str) -> Dict[str, float]:
        """Term Frequency normalizada."""
        tokens = self._tokenizar(texto)
        if not tokens:
            return {}
        contagem: Dict[str, int] = {}
        for t in tokens:
            contagem[t] = contagem.get(t, 0) + 1
        total = len(tokens)
        return {t: c / total for t, c in contagem.items()}

    def _idf(self, token: str) -> float:
        """Inverse Document Frequency com suavização."""
        df = self._df.get(token, 0)
        if df == 0 or self._n_docs == 0:
            return 0.0
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def _similaridade_tfidf(
        self, tf_consulta: Dict[str, float], tf_doc: Dict[str, float]
    ) -> float:
        """
        Similaridade coseno entre vetores TF-IDF da consulta e do trecho.
        """
        tokens_comuns = set(tf_consulta) & set(tf_doc)
        if not tokens_comuns:
            return 0.0

        dot = 0.0
        for t in tokens_comuns:
            idf = self._idf(t)
            dot += tf_consulta[t] * idf * tf_doc[t] * idf

        def norma(tf_vec):
            return math.sqrt(sum((v * self._idf(t)) ** 2 for t, v in tf_vec.items()))

        denom = norma(tf_consulta) * norma(tf_doc)
        return dot / denom if denom > 0 else 0.0


# -------------------------------------------------------
# DETECTOR DE INTENÇÃO
# -------------------------------------------------------

class DetectorIntencao:
    """
    Lê a mensagem do usuário e decide o que ele quer.

    Retorna um dict:
      {
        "tipo":      "analisar" | "corrigir" | "melhorar" | "criar"
                     | "explicar" | "conversa",
        "codigo":    "..." ou None,     ← código extraído da mensagem
        "texto":     "...",             ← mensagem sem o bloco de código
        "confianca": 0.0 a 1.0,
        "linguagem": "python" | None,  ← linguagem detectada no texto
      }

    MELHORIAS v2:
      - Pesos por padrão: padrões mais específicos valem mais que genéricos,
        evitando que "cria um arquivo" bata em "criar" quando o contexto
        real é "corrigir".
      - Nova intenção "explicar": "explica", "como funciona", "o que é", etc.
      - Detecção de linguagem na mensagem para passar ao handler de criar.
      - Desambiguação: corrigir/melhorar têm prioridade sobre analisar
        quando presença de código é forte.
      - Pontuação de código inline mais precisa (evita falsos positivos
        em frases que contêm "import" como substantivo).
    """

    # (padrão, peso) — peso > 1 = mais específico / mais forte
    PADROES: dict[str, list[tuple[str, float]]] = {
        "analisar": [
            (r"\banalis[ae]\b",                      1.0),
            (r"\brevisa\b",                          1.0),
            (r"\bverifica\b",                        0.8),
            (r"\brevise\b",                          1.0),
            (r"\baudit[ae]\b",                       1.2),
            (r"\bdiagnosti[cq]\b",                   1.2),
            (r"\bo que (tem|h[aá]) de errado\b",     1.5),
            (r"\btem (algum |alguma )?erro\b",        1.2),
            (r"\bo que (est[aá] errado|est[aá] ruim)\b", 1.2),
            (r"\bverifica pra mim\b",                1.0),
            (r"\bprocura (erro|bug|problema)\b",     1.3),
        ],
        "corrigir": [
            (r"\bcorrig[ei]\b",                      1.0),
            (r"\barruma\b",                          1.0),
            (r"\bconserta\b",                        1.0),
            (r"\bfix\b",                             0.8),   # palavra curta, peso menor
            (r"\bcorrige\b",                         1.0),
            (r"\bcorrija\b",                         1.2),
            (r"\bapaga o erro\b",                    1.5),
            (r"\bresolve o (erro|bug|problema)\b",   1.5),
            (r"\btira o erro\b",                     1.3),
            (r"\bfaz (funcionar|rodar)\b",           1.2),
            (r"\bconserta o bug\b",                  1.5),
        ],
        "melhorar": [
            (r"\bmelhora\b",                         1.0),
            (r"\brefatora\b",                        1.2),
            (r"\botimiza\b",                         1.2),
            (r"\bdeixa (mais limpo|melhor)\b",       1.3),
            (r"\brefactor\b",                        1.0),
            (r"\bimprove\b",                         0.8),
            (r"\baprimora\b",                        1.0),
            (r"\bclean( up)?\b",                     0.8),
            (r"\bdeixa (mais (eficiente|rápido|performático))\b", 1.3),
            (r"\bsimplifica\b",                      1.1),
            (r"\breenomeia\b",                       1.0),
        ],
        "criar": [
            (r"\bcria\b",                            1.0),
            (r"\bfaz (um|uma|o|a)\b",                0.9),
            (r"\bgera\b",                            1.0),
            (r"\bescreve (um|uma)\b",                1.0),
            (r"\bimplementa\b",                      1.2),
            (r"\bcreate\b",                          0.8),
            (r"\bgenerate\b",                        0.8),
            (r"\bconstrui\b",                        1.0),
            (r"\bmonta\b",                           0.9),
            (r"\bdesenvol[vw]\b",                    1.1),
            (r"\bescr[ea]ve (o |a |um |uma )?(código|função|classe|script)\b", 1.5),
            (r"\bpreciso de (um|uma) (código|função|classe|script)\b", 1.4),
        ],
        "explicar": [
            (r"\bexplica\b",                         1.2),
            (r"\bcomo funciona\b",                   1.3),
            (r"\bo que [eé] (esse|um|uma|o|a)\b",    1.0),
            (r"\bpara que serve\b",                  1.2),
            (r"\bme (fala|diz|conta) (o que|como)\b",1.0),
            (r"\bme ajuda a entender\b",             1.3),
            (r"\bnão (entend[oi]|entendo)\b",        1.1),
            (r"\bsignifica\b",                       1.0),
            (r"\bdiferença entre\b",                 1.2),
        ],
    }

    def detectar(self, mensagem: str) -> dict:
        """Detecta a intenção da mensagem."""
        msg_lower = mensagem.lower()

        # Extrai bloco de código delimitado por ```
        codigo = self._extrair_codigo(mensagem)
        texto_limpo = self._remover_codigo(mensagem).strip()

        # Detecta código inline se não tem bloco
        if not codigo:
            codigo = self._detectar_codigo_inline(mensagem)

        # Detecta linguagem mencionada no texto
        linguagem = self._detectar_linguagem(texto_limpo or msg_lower)

        # Calcula score ponderado por intenção
        scores: dict[str, float] = {}
        for intencao, padroes in self.PADROES.items():
            score = 0.0
            for padrao, peso in padroes:
                if re.search(padrao, msg_lower):
                    score += peso
            if score > 0:
                scores[intencao] = score

        # Decide intenção principal
        if scores:
            melhor = max(scores, key=scores.get)
            # Normaliza: score 3.0 → confiança 1.0
            confianca = min(1.0, scores[melhor] / 3.0)

            # Desambiguação: se há empate entre analisar e corrigir/melhorar,
            # corrigir/melhorar ganham (ação > diagnóstico)
            for prioritario in ("corrigir", "melhorar"):
                if (prioritario in scores and "analisar" in scores
                        and scores[prioritario] >= scores["analisar"] * 0.8):
                    melhor = prioritario
                    break
        else:
            melhor = "conversa"
            confianca = 0.9

        # Regra: tem código mas nenhuma intenção de ação clara → analisar
        if melhor == "conversa" and codigo:
            melhor = "analisar"
            confianca = 0.7

        return {
            "tipo":      melhor,
            "codigo":    codigo,
            "texto":     texto_limpo,
            "confianca": round(confianca, 2),
            "linguagem": linguagem,
        }

    # ── Extração de código ──

    def _extrair_codigo(self, texto: str) -> Optional[str]:
        """Extrai código de blocos ```...``` (com ou sem nome de linguagem)."""
        match = re.search(r'```[\w]*\n?([\s\S]*?)```', texto)
        return match.group(1).strip() if match else None

    def _remover_codigo(self, texto: str) -> str:
        """Remove todos os blocos ``` do texto."""
        return re.sub(r'```[\s\S]*?```', '', texto).strip()

    def _detectar_codigo_inline(self, texto: str) -> Optional[str]:
        """
        Detecta código sem marcadores ```.

        MELHORIA v2: verifica se há pelo menos 2 linhas com indicadores
        de código E que a mensagem tem mais código que prosa. Evita
        falsos positivos em frases curtas como "importa essa ideia".
        """
        indicadores = [
            r'^\s*def\s+\w+\s*\(',
            r'^\s*class\s+\w+[\s:(]',
            r'^\s*function\s+\w+\s*\(',
            r'^\s*const\s+\w+\s*=',
            r'^\s*(?:import|from)\s+\w+',
            r'^\s*#include\s*[<"]',
            r'^\s*public\s+(class|static)\s+\w+',
            r'^\s*fn\s+\w+\s*\(',
            r'^\s*func\s+\w+\s*\(',
            r'^\s*(?:var|let)\s+\w+\s*[=:]',
        ]
        linhas = texto.splitlines()
        if len(linhas) < 2:
            return None  # mensagem de uma linha raramente é só código

        hits = sum(
            1 for linha in linhas
            if any(re.match(ind, linha) for ind in indicadores)
        )
        # Exige pelo menos 1 indicador forte E mais de 1 linha
        if hits >= 1 and len(linhas) > 1:
            return texto
        return None

    # ── Detecção de linguagem ──

    def _detectar_linguagem(self, texto: str) -> Optional[str]:
        """
        Detecta linguagem de programação mencionada no texto.

        NOVA FEATURE v2: passa essa informação para o handler de criar,
        que pode usá-la no prompt de geração.
        """
        texto_lower = texto.lower()
        for palavra, lang in _PALAVRAS_LANG.items():
            # Busca como palavra inteira para evitar "go" em "vou"
            if re.search(r'\b' + re.escape(palavra) + r'\b', texto_lower):
                return lang
        # Tenta extensão de arquivo mencionada ("arquivo.py", "main.ts")
        match = re.search(r'\b\w+\.(' + '|'.join(_EXT_PARA_LANG) + r')\b', texto_lower)
        if match:
            ext = match.group(1)
            return _EXT_PARA_LANG.get(ext)
        return None


# -------------------------------------------------------
# GERADOR DE RESPOSTAS (conversa)
# -------------------------------------------------------

class GeradorResposta:
    """
    Gera respostas de conversa (quando não é código).

    Por enquanto usa templates + contexto da memory.
    Quando o transformer tiver treino, substitui o método
    `_gerar_com_modelo()` e o resto continua igual.

    MELHORIAS v2:
      - Respostas para frustração/erro ("não tá funcionando", "tá quebrando").
      - Reconhecimento de pedidos de ajuda vagos → pede contexto específico.
      - Anti-repetição: guarda as últimas respostas dadas e evita repetir.
      - Respostas de despedida.
      - Detecção de confirmação ("sim", "pode", "vai") para contextos de
        continuação de fluxo (ex: "quer que eu corrija?" → "sim").
    """

    SAUDACOES = [
        "oi", "olá", "ola", "hey", "eai", "e aí", "bom dia",
        "boa tarde", "boa noite", "salve", "fala", "opa", "oi oi",
    ]

    RESPOSTAS_SAUDACAO = [
        "Fala, {nome}! Pode mandar o código ou me dizer o que você precisa.",
        "Oi, {nome}! Tô aqui. Vai ter código hoje ou é só papo?",
        "E aí, {nome}! Manda ver.",
        "Olá, {nome}! Pronto pra trabalhar.",
        "Opa, {nome}! Cola o código ou me diz o que precisa.",
    ]

    PERGUNTAS_CAPACIDADE = [
        r"o que você (faz|sabe|consegue|pode)",
        r"quais (são suas|suas) (funções|capacidades|habilidades)",
        r"me (fala|diz|conta) (sobre você|o que você é)",
        r"como você funciona",
        r"para que (você )?serve",
        r"o que (é|és) você",
    ]

    RESPOSTA_CAPACIDADE = """Sou o ParadoxoX. Trabalho com código.

O que eu faço:
  🔍 ANALISAR  → manda um código, te digo tudo que tem de errado
  🔧 CORRIGIR  → mando de volta corrigido
  ⚡ MELHORAR  → deixo mais limpo, mais eficiente
  ✨ CRIAR     → descreve o que quer, eu gero do zero
  📖 EXPLICAR  → explico como funciona qualquer trecho de código

Linguagens: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, e mais.

Pode mandar o código direto ou usar ``` pra blocos maiores."""

    # Padrões de frustração/erro reportado pelo usuário
    _FRUSTRACOES = [
        r"não (tá|está|funciona|roda|executa|compila)",
        r"tá (quebrando|crashando|bugado|com erro|dando erro)",
        r"deu (erro|bug|problema|exception|traceback)",
        r"n[ãa]o (consigo|sei) (fazer|entender|resolver)",
        r"help",
        r"socorro",
        r"que (merda|droga|inferno)",
    ]

    # Padrões de confirmação (o usuário diz "sim" pra algo que o bot perguntou)
    _CONFIRMACOES = [
        r"^(sim|pode|vai|s|yes|manda|faz isso|pode ser|claro|obvio|óbvio|bora)\.?$",
    ]

    # Padrões de despedida
    _DESPEDIDAS = [
        r"\b(tchau|xau|até|flw|falou|bye|adeus|até mais|até logo)\b",
    ]

    RESPOSTAS_FRUSTRACAO = [
        "Cola o código aqui com o erro que eu resolvo.",
        "Manda o código e a mensagem de erro. Vou olhar.",
        "Sem ver o código e o traceback fica difícil. Manda os dois.",
        "Traz o código e o erro. A gente descobre o que tá acontecendo.",
    ]

    RESPOSTAS_DESPEDIDA = [
        "Tmj, {nome}. Qualquer código, é só chamar.",
        "Até mais, {nome}. Manda o código quando precisar.",
        "Falou, {nome}. Tô aqui quando precisar.",
    ]

    def __init__(self, nome_usuario: str = "Usuário"):
        self.nome_usuario = nome_usuario
        self._ultimas_respostas: list[str] = []   # anti-repetição

    def gerar(self, mensagem: str, contexto: dict) -> str:
        """Gera resposta de conversa baseada na mensagem e contexto."""
        msg_lower = mensagem.lower().strip()

        # Saudação
        if any(msg_lower.startswith(s) for s in self.SAUDACOES) and len(mensagem) < 35:
            return self._escolher(self.RESPOSTAS_SAUDACAO).format(
                nome=self.nome_usuario
            )

        # Despedida
        for p in self._DESPEDIDAS:
            if re.search(p, msg_lower):
                return random.choice(self.RESPOSTAS_DESPEDIDA).format(
                    nome=self.nome_usuario
                )

        # Confirmação de ação anterior ("sim", "pode", "vai")
        for p in self._CONFIRMACOES:
            if re.match(p, msg_lower):
                return self._resposta_confirmacao(contexto)

        # "O que você faz?"
        for p in self.PERGUNTAS_CAPACIDADE:
            if re.search(p, msg_lower):
                return self.RESPOSTA_CAPACIDADE

        # Frustração / erro reportado
        for p in self._FRUSTRACOES:
            if re.search(p, msg_lower):
                return self._escolher(self.RESPOSTAS_FRUSTRACAO)

        # Pergunta sobre linguagem preferida
        if re.search(r'(qual|que) (linguagem|lang)', msg_lower):
            lang = contexto.get("preferencias", {}).get("linguagem_favorita", "python")
            return f"Pela sua história, você mais usa **{lang}**. Quer trocar? É só falar."

        # Agradecimento
        if re.search(r'\b(obrigad[ao]|valeu|vlw|thanks|thx|grat[ao])\b', msg_lower):
            return f"Tmj, {self.nome_usuario}. Manda mais quando precisar."

        # Pergunta genérica
        if re.search(r'\b(como|o que|qual|quando|onde|por que|porque)\b', msg_lower):
            return self._resposta_pergunta_generica(mensagem, contexto)

        # Fallback
        return (
            f"Entendi, {self.nome_usuario}. "
            "Se tiver código pra analisar, corrigir ou melhorar, manda aí. "
            "Se quiser que eu crie algo, descreve o que você precisa."
        )

    def _escolher(self, opcoes: list[str]) -> str:
        """
        Escolhe uma resposta evitando repetir a última dada.
        Anti-repetição simples mas eficaz.
        """
        disponiveis = [r for r in opcoes if r not in self._ultimas_respostas[-2:]]
        escolhida = random.choice(disponiveis if disponiveis else opcoes)
        self._ultimas_respostas.append(escolhida)
        if len(self._ultimas_respostas) > 10:
            self._ultimas_respostas.pop(0)
        return escolhida

    def _resposta_confirmacao(self, contexto: dict) -> str:
        """
        NOVA v2: o usuário disse "sim" / "pode" para algo.
        Olha o histórico pra saber o que foi perguntado.
        """
        historico = contexto.get("historico_recente", [])
        if historico:
            ultima_ia = next(
                (m for m in reversed(historico) if getattr(m, "papel", None) == "ia"),
                None,
            )
            if ultima_ia:
                resumo = getattr(ultima_ia, "resumo", lambda: "")()
                if "corrig" in resumo.lower():
                    return "Manda o código então. Pode ser direto ou entre ```."
                if "melhora" in resumo.lower() or "otimiz" in resumo.lower():
                    return "Manda o código. Vou deixar mais limpo."
                if "analis" in resumo.lower():
                    return "Cola o código aqui."
        return "Pode mandar o código."

    def _resposta_pergunta_generica(self, mensagem: str, contexto: dict) -> str:
        """Responde perguntas genéricas usando o contexto disponível."""
        historico = contexto.get("historico_recente", [])
        arquivos  = contexto.get("arquivos_relevantes", [])

        if arquivos:
            nomes = ", ".join(
                a["arquivo"] for a in arquivos[:3] if a.get("arquivo")
            )
            return (
                f"Tô com contexto dos seus arquivos ({nomes}). "
                "Quer que eu analise, corrija ou melhore algum deles? "
                "Ou manda o código que você tem em mente."
            )

        if len(historico) > 2:
            return (
                "Pode elaborar mais? Se tiver código envolvido, manda ele "
                "que fica mais fácil de te ajudar de verdade."
            )

        return (
            "Boa pergunta. Pra te responder direito, precisa de mais contexto. "
            "Manda o código ou descreve melhor o que você precisa."
        )


# -------------------------------------------------------
# BRAIN — O CÉREBRO PRINCIPAL
# -------------------------------------------------------

class ParadoxoBrain:
    """
    Ponto de entrada do ParadoxoX.

    Uso:
        brain = ParadoxoBrain()
        resposta = brain.chat("analisa esse código: def f(x): return x*2")
        print(resposta)

    Com código em bloco:
        brain.chat('''
            corrige isso:
            ```python
            import os
            import os
            password = "123"
            ```
        ''')

    Criação com linguagem explícita:
        brain.chat("cria uma classe Produto com nome e preço em Python")

    Explicação:
        brain.chat("explica como funciona esse código: ...")

    MELHORIAS v2:
      - _carregar_modelo() usa ParadoxoTransformer.carregar() (v2 do transformer).
      - _handle_analisar() aceita código da memory quando não tem na mensagem.
      - _handle_criar() repassa a linguagem detectada pelo detector.
      - Novo _handle_explicar() para a intenção "explicar".
      - chat() valida e sanitiza entrada antes de processar.
      - Handlers protegidos com try/except específico — um erro num módulo
        não trava o bot inteiro.
      - chat() aceita parâmetro retornar_meta=True para retornar dict com
        intenção, confiança e outros metadados (útil para debug/API).
      - _formatar_analisar() extrai a formatação para método separado
        (código mais limpo e reutilizável).
    """
    # Tamanho máximo de entrada aceita (caracteres)
    _MAX_INPUT = 50_000

    def __init__(
        self,
        # O vocab está na pasta memory
        caminho_vocab:  str = "memory/vocab.json", 
        # O modelo está na pasta core
        caminho_modelo: str = "core/modelo_paradoxox.json",
        caminho_db:     str = "memory/paradoxox_memory.db",
    ):
        print("⚛️  Iniciando ParadoxoX...\n")

        # Memory — primeira a iniciar
        self.mem = MemoryManager(caminho_db)

        # Detector de intenção
        self.detector = DetectorIntencao()

        # Code engine
        self.analyzer = CodeAnalyzer()

        # Gerador de respostas
        self.gerador = GeradorResposta(
            nome_usuario=self.mem.prefs.nome_usuario
        )

        # Tokenizer
        self.tokenizer = Tokenizer()
        if os.path.exists(caminho_vocab):
            self.tokenizer.carregar(caminho_vocab)
            print(f"📖 Vocab carregado: {len(self.tokenizer.vocab)} tokens")
        else:
            print("📖 Vocab novo (sem arquivo salvo ainda)")

        # Transformer
        self.transformer: Optional[ParadoxoTransformer] = None
        self._modelo_carregado = False
        if os.path.exists(caminho_modelo):
            try:
                self.transformer = self._carregar_modelo(caminho_modelo)
                self._modelo_carregado = True
                print("🧠 Modelo carregado!")
            except Exception as e:
                print(f"⚠️  Modelo não carregado ({e}) — usando modo template")
        else:
            print("🧠 Modelo não treinado ainda — usando modo template")

        self.refactor = CodeRefactor(
            transformer=self.transformer,
            tokenizer=self.tokenizer
        )

        # RAG — base de conhecimento recuperável
        self._caminho_rag = os.path.join(
            os.path.dirname(caminho_db), "rag_index.json"
        )
        self.rag = RAGEngine()
        if self.rag.carregar(self._caminho_rag):
            print(f"📚 RAG carregado: {self.rag._n_docs} trechos indexados")
        else:
            print("📚 RAG inicializado (base vazia)")

        # Vision
        self.vision = VisionProcessor()
        
        # O PRINT QUE VOCÊ QUERIA FICA AQUI, NO FINAL DO INIT
        print(f"\n✅ ParadoxoX pronto! Olá, {self.mem.prefs.nome_usuario}!\n")

    # --- MÉTODOS DA CLASSE (Todos com 4 espaços de recuo) ---

    def _extrair_nome_arquivo(self, texto: str) -> str:
        """Extrai nome de arquivo da mensagem se mencionado."""
        exts = "|".join(_EXT_PARA_LANG.keys())
        match = re.search(rf'\b([\w\-]+\.(?:{exts}))\b', texto)
        return match.group(1) if match else ""

    def _ultimo_codigo(self, contexto: dict) -> Optional[str]:
        """Pega o último código registrado na memory."""
        arquivos = contexto.get("arquivos_relevantes", [])
        if arquivos:
            return arquivos[0].get("conteudo") or arquivos[0].get("resumo")
        return None

    def stats(self):
        """Exibe estatísticas da sessão."""
        self.mem.exibir_stats()

    def nova_sessao(self):
        """Inicia uma nova conversa do zero."""
        self.mem.nova_sessao()
        print("🆕 Nova sessão iniciada.")

    def historico(self, n: int = 10):
        """Exibe as últimas N mensagens."""
        msgs = self.mem.historico_recente(n)
        print(f"\n📜 Últimas {len(msgs)} mensagens:")
        for m in msgs:
            print(f"  {m.resumo()}")

    def fechar(self):
        """Encerra o brain corretamente."""
        self.mem.fechar()
        self.rag.salvar(self._caminho_rag)
        print("👋 ParadoxoX encerrado.")

    def indexar_documento(self, nome: str, conteudo: str) -> int:
        """
        Adiciona um documento à base de conhecimento do RAG.

        Args:
            nome:     Identificador do documento (ex: "main.py", "docs/api.md").
            conteudo: Texto completo a indexar.

        Returns:
            Número de trechos adicionados ao índice.
        """
        n = self.rag.indexar(nome, conteudo)
        print(f"📚 RAG: {n} trecho(s) indexado(s) de '{nome}'")
        return n

        print(f"\n✅ ParadoxoX pronto! Olá, {self.mem.prefs.nome_usuario}!\n")

    # -------------------------------------------------------
    # CARREGAMENTO DO MODELO
    # -------------------------------------------------------

    def _carregar_modelo(self, caminho: str) -> ParadoxoTransformer:
        """
        Carrega o transformer salvo.

        MELHORIA v2: tenta primeiro o formato novo (ParadoxoTransformer.carregar),
        e faz fallback para o formato legado se não encontrar a chave 'versao'.
        """
        import json
        with open(caminho) as f:
            dados = json.load(f)

        # Formato v2 (tem chave "versao")
        if dados.get("versao"):
            return ParadoxoTransformer.carregar(caminho)

        # Formato legado (v1)
        cfg = dados["config"]
        modelo = ParadoxoTransformer(
            tamanho_vocab=cfg["tamanho_vocab"],
            dim_modelo=cfg["dim_modelo"],
            num_camadas=cfg["num_camadas"],
            seq_max=cfg["seq_max"],
        )
        modelo.embedding.tabela = dados["embedding"]
        if dados.get("W_out"):
            modelo.W_out = dados["W_out"]
        return modelo

    # -------------------------------------------------------
    # CHAT — interface principal
    # -------------------------------------------------------

    def chat(
        self,
        mensagem: str,
        retornar_meta: bool = False,
    ) -> "str | dict":
        """
        Recebe uma mensagem, processa, e retorna a resposta.
        Salva tudo na memory automaticamente.

        Args:
          mensagem:      Texto do usuário.
          retornar_meta: Se True, retorna dict com 'resposta', 'intencao',
                         'confianca' e 'linguagem'. Útil para debug ou API.

        Returns:
          str com a resposta, ou dict se retornar_meta=True.
        """
        # ── Sanitização ──
        mensagem = mensagem.strip()
        if not mensagem:
            return "Manda algo." if not retornar_meta else {
                "resposta": "Manda algo.", "intencao": "vazio",
                "confianca": 1.0, "linguagem": None,
            }
        if len(mensagem) > self._MAX_INPUT:
            mensagem = mensagem[:self._MAX_INPUT]
            print(f"⚠️  Entrada truncada para {self._MAX_INPUT} caracteres.")

        # Salva mensagem do usuário
        self.mem.salvar_mensagem("usuario", mensagem)

        # Detecta intenção
        intencao = self.detector.detectar(mensagem)

        # Mantém nome do gerador sincronizado
        self.gerador.nome_usuario = self.mem.prefs.nome_usuario

        # Monta contexto da memory
        contexto = self.mem.contexto_para_modelo(mensagem)

        # ── RAG: recupera trechos relevantes e enriquece o contexto ──
        rag_contexto = self.rag.contexto_para_prompt(mensagem, top_k=3)
        if rag_contexto:
            contexto["rag"] = rag_contexto

        # Roteia para o handler correto
        tipo = intencao["tipo"]
        try:
                # === CÉREBRO TOTALMENTE DESBLOQUEADO ===
            if self._modelo_carregado and self.transformer is not None:
                print(f"\n🧠 [ParadoxoX está pensando na intenção '{intencao}' usando o Transformer...]")
                
                # Monta o texto para o modelo ler
                prompt = formatar_contexto_para_prompt(contexto)
                
                # Chama o modelo para gerar a resposta
                try:
                    resposta = self.transformer.gerar(prompt) 
                except (TypeError, AttributeError):
                    ids_entrada = self.tokenizer.encode(prompt)
                    ids_saida = self.transformer.gerar(ids_entrada) 
                    resposta = self.tokenizer.decode(ids_saida)
                    
            # === FALLBACK (SE O JSON SUMIR) ===
            else:
                if tipo == "analisar":
                    resposta = self._handle_analisar(intencao, contexto)
                elif tipo == "corrigir":
                    resposta = self._handle_corrigir(intencao, contexto)
                elif tipo == "melhorar":
                    resposta = self._handle_melhorar(intencao, contexto)
                elif tipo == "criar":
                    resposta = self._handle_criar(intencao, contexto)
                elif tipo == "explicar":
                    resposta = self._handle_explicar(intencao, contexto)
                elif tipo == "visao":
                    resposta = self._handle_visao(intencao, contexto)
                else:
                    resposta = self.gerador.gerar(mensagem, contexto)
        except Exception as e:
            # Protege o loop principal — nenhum erro de módulo trava o bot
            resposta = (
                f"Eita, algo deu errado internamente ({type(e).__name__}: {e}). "
                "Tenta mandar o código de novo?"
            )

        # Salva resposta na memory
        self.mem.salvar_mensagem("ia", resposta)

        if retornar_meta:
            return {
                "resposta":   resposta,
                "intencao":   tipo,
                "confianca":  intencao["confianca"],
                "linguagem":  intencao.get("linguagem"),
            }
        return resposta

    # -------------------------------------------------------
    # HANDLERS DE INTENÇÃO
    # -------------------------------------------------------

    def _handle_analisar(self, intencao: dict, contexto: dict) -> str:
        """
        Analisa o código e devolve relatório.

        MELHORIA v2: aceita código da memory quando a mensagem não tem código,
        mas o usuário claramente quer análise ("analisa o último", etc.).
        """
        codigo = intencao["codigo"] or self._ultimo_codigo(contexto)
        mensagem_original = intencao["texto"]

        if not codigo:
            return (
                "Preciso do código pra analisar. "
                "Manda ele aqui, pode ser direto ou entre ```."
            )

        nome_arquivo = self._extrair_nome_arquivo(mensagem_original)
        resultado = self.analyzer.analisar(codigo, nome_arquivo)
        m = resultado.metricas

        self.mem.salvar_contexto_codigo(
            nome_arquivo=nome_arquivo or "snippet",
            linguagem=m.linguagem,
            conteudo=codigo,
            score=m.score_qualidade,
            problemas=[str(p) for p in resultado.problemas],
        )

        return self._formatar_analise(resultado)

    def _formatar_analise(self, resultado) -> str:
        """
        NOVA v2: extrai a formatação do resultado de análise para método
        separado — reutilizável e mais fácil de modificar.
        """
        m = resultado.metricas
        erros  = [p for p in resultado.problemas if p.tipo == "erro"]
        avisos = [p for p in resultado.problemas if p.tipo == "aviso"]
        sugest = [p for p in resultado.problemas if p.tipo == "sugestao"]

        linhas = [
            f"Analisei o código {m.linguagem.upper()}. "
            f"Score: **{m.score_qualidade:.1f}/10**\n"
        ]

        if not resultado.problemas:
            linhas.append("✅ Código limpo! Nenhum problema encontrado.")
        else:
            if erros:
                linhas.append(f"🔴 **{len(erros)} erro(s) crítico(s):**")
                for p in erros[:3]:
                    linhas.append(f"  • Linha {p.linha}: {p.mensagem}")
                    linhas.append(f"    → {p.sugestao}")
                if len(erros) > 3:
                    linhas.append(f"  • ...e mais {len(erros) - 3} erro(s)")

            if avisos:
                linhas.append(f"\n🟡 **{len(avisos)} aviso(s):**")
                for p in avisos[:3]:
                    linhas.append(f"  • Linha {p.linha}: {p.mensagem}")
                if len(avisos) > 3:
                    linhas.append(f"  • ...e mais {len(avisos) - 3} aviso(s)")

            if sugest:
                linhas.append(f"\n🔵 **{len(sugest)} sugestão(ões):**")
                for p in sugest[:2]:
                    linhas.append(f"  • {p.mensagem}")

        if resultado.sugestoes_gerais:
            linhas.append("\n📋 **Geral:**")
            for s in resultado.sugestoes_gerais:
                linhas.append(f"  {s}")

        linhas.append(
            f"\n📊 {m.total_linhas} linhas | "
            f"{m.num_funcoes} funções | "
            f"Complexidade: {m.complexidade_estimada}"
        )

        if resultado.problemas:
            linhas.append(
                "\nQuer que eu **corrija** ou **melhore** esse código?"
            )

        return "\n".join(linhas)

    def _handle_corrigir(self, intencao: dict, contexto: dict) -> str:
        """Corrige o código — usa o código da mensagem ou o último da memory."""
        codigo = intencao["codigo"] or self._ultimo_codigo(contexto)

        if not codigo:
            return (
                "Preciso do código pra corrigir. "
                "Manda ele aqui, pode ser direto ou entre ```."
            )

        nome_arquivo = self._extrair_nome_arquivo(intencao["texto"])
        resultado = self.refactor.corrigir(codigo, nome_arquivo)

        self.mem.salvar_contexto_codigo(
            nome_arquivo=(nome_arquivo or "snippet") + "_corrigido",
            linguagem=resultado.linguagem,
            conteudo=resultado.codigo_resultado,
            score=resultado.score_depois,
            tipo="arquivo",
        )

        delta = resultado.score_depois - resultado.score_antes
        sinal = "+" if delta >= 0 else ""
        linhas = [
            f"✅ Código corrigido! "
            f"Score: {resultado.score_antes:.1f} → **{resultado.score_depois:.1f}** "
            f"({sinal}{delta:.1f})\n"
        ]

        if resultado.mudancas:
            linhas.append(f"**{len(resultado.mudancas)} correção(ões) aplicada(s):**")
            for m in resultado.mudancas:
                linhas.append(f"  ✅ {m}")

        linhas += [
            "\n**Código corrigido:**",
            f"```{resultado.linguagem}",
            resultado.codigo_resultado.strip(),
            "```",
        ]
        return "\n".join(linhas)

    def _handle_melhorar(self, intencao: dict, contexto: dict) -> str:
        """Melhora o código — vai além da correção."""
        codigo = intencao["codigo"] or self._ultimo_codigo(contexto)

        if not codigo:
            return (
                "Preciso do código pra melhorar. "
                "Manda aqui, pode ser direto ou entre ```."
            )

        nome_arquivo = self._extrair_nome_arquivo(intencao["texto"])
        resultado = self.refactor.melhorar(codigo, nome_arquivo)

        self.mem.salvar_contexto_codigo(
            nome_arquivo=(nome_arquivo or "snippet") + "_melhorado",
            linguagem=resultado.linguagem,
            conteudo=resultado.codigo_resultado,
            score=resultado.score_depois,
            tipo="arquivo",
        )

        delta = resultado.score_depois - resultado.score_antes
        sinal = "+" if delta >= 0 else ""
        linhas = [
            f"⚡ Código melhorado! "
            f"Score: {resultado.score_antes:.1f} → **{resultado.score_depois:.1f}** "
            f"({sinal}{delta:.1f})\n"
        ]

        if resultado.mudancas:
            linhas.append(f"**{len(resultado.mudancas)} melhoria(s) aplicada(s):**")
            for m in resultado.mudancas:
                linhas.append(f"  ✅ {m}")

        linhas += [
            "\n**Código melhorado:**",
            f"```{resultado.linguagem}",
            resultado.codigo_resultado.strip(),
            "```",
        ]
        return "\n".join(linhas)

    def _handle_criar(self, intencao: dict) -> str:
        """
        Cria código novo a partir da descrição.

        MELHORIA v2: repassa a linguagem detectada pelo detector ao módulo
        de criação, para que ele gere código na linguagem certa sem precisar
        de heurísticas adicionais no refactor.
        """
        descricao = intencao["texto"]
        linguagem = intencao.get("linguagem")  # NOVO: pode ser None

        if len(descricao.strip()) < 5:
            return (
                "Descreve melhor o que você quer criar. "
                "Exemplo: 'cria uma classe Produto com nome e preço em Python'"
            )

        # Tenta passar linguagem se o CodeRefactor.criar() aceitar o parâmetro
        try:
            resultado = self.refactor.criar(descricao, linguagem=linguagem)
        except TypeError:
            # Versão antiga do refactor não aceita linguagem — fallback
            resultado = self.refactor.criar(descricao)

        self.mem.salvar_contexto_codigo(
            nome_arquivo="gerado_" + re.sub(r'\W+', '_', descricao[:20]).strip('_'),
            linguagem=resultado.linguagem,
            conteudo=resultado.codigo_resultado,
            score=resultado.score_depois,
            tipo="snippet",
        )

        linhas = [
            f"✨ Código gerado em **{resultado.linguagem.upper()}**! "
            f"Score: {resultado.score_depois:.1f}/10\n",
            f"```{resultado.linguagem}",
            resultado.codigo_resultado.strip(),
            "```",
            "\nQuer que eu **analise**, **corrija** ou **melhore** esse código?",
        ]
        return "\n".join(linhas)

    def _handle_explicar(self, intencao: dict, contexto: dict) -> str:
        """
        NOVA INTENÇÃO v2: explica um trecho de código ou conceito.

        Se tiver código → análise + explicação em linguagem simples.
        Se não tiver código → tenta usar o último da memory.
        Se não tiver nada → pede o código.
        """
        codigo = intencao["codigo"] or self._ultimo_codigo(contexto)

        if not codigo:
            return (
                "Manda o código que você quer entender. "
                "Pode ser direto ou entre ```."
            )

        # Usa o analyzer para obter métricas e contexto
        nome_arquivo = self._extrair_nome_arquivo(intencao["texto"])
        resultado = self.analyzer.analisar(codigo, nome_arquivo)
        m = resultado.metricas

        linhas = [
            f"📖 **Explicação do código {m.linguagem.upper()}** "
            f"({m.total_linhas} linhas, {m.num_funcoes} função(ões)):\n"
        ]

        # Adiciona sugestões gerais como "explicação"
        if resultado.sugestoes_gerais:
            for s in resultado.sugestoes_gerais:
                linhas.append(f"  {s}")
        else:
            linhas.append(
                "O código parece bem estruturado. "
                "Pergunta algo específico que eu explico melhor."
            )

        if resultado.problemas:
            linhas.append(
                f"\n⚠️  Vi {len(resultado.problemas)} ponto(s) que valem atenção. "
                "Quer que eu detalhe ou corrija?"
            )

        return "\n".join(linhas)
    







def formatar_contexto_para_prompt(ctx: dict, max_historico: int = 6) -> str:
    """
    Converte o dicionário retornado por memory.contexto_para_modelo()
    numa string estruturada pronta para o Tokenizer.

    Estrutura do prompt gerado:
        [SISTEMA]
        Usuário: João | Lang: python | Estilo: tecnico

        [HISTORICO]
        usuario: ...
        ia: ...

        [CODIGO]
        arquivo: main.py (python, score=0.9)
        ...

        [PERGUNTA]
        <pergunta atual>

    Parâmetros
    ----------
    ctx          : dict retornado por memory.contexto_para_modelo()
    max_historico: quantas mensagens do histórico incluir no prompt
                   (mais mensagens = prompt maior = mais tokens)
    """
    partes = []

    # ── 1. Linha de sistema (preferências) ───────────────────────────
    prefs = ctx.get("preferencias", {})
    nome  = prefs.get("nome_usuario", "Usuário")
    lang  = prefs.get("linguagem_favorita", "python")
    estilo = prefs.get("estilo_resposta", "tecnico")
    partes.append(f"[SISTEMA]\nUsuário: {nome} | Lang: {lang} | Estilo: {estilo}")

    # ── 2. Histórico de mensagens ─────────────────────────────────────
    historico = ctx.get("historico_recente", [])
    if historico:
        # Pega as N mensagens mais recentes
        recentes = historico[-max_historico:]
        linhas_hist = []
        for msg in recentes:
            # Mensagem pode ser objeto Mensagem ou dict
            if hasattr(msg, "papel"):
                papel    = msg.papel
                conteudo = msg.conteudo
            else:
                papel    = msg.get("papel", "?")
                conteudo = msg.get("conteudo", "")
            # Trunca mensagens longas para não explodir o contexto
            if len(conteudo) > 300:
                conteudo = conteudo[:300] + "..."
            linhas_hist.append(f"{papel}: {conteudo}")
        partes.append("[HISTORICO]\n" + "\n".join(linhas_hist))

    # ── 3. Contexto de código ─────────────────────────────────────────
    codigos = ctx.get("contextos_codigo", [])
    if codigos:
        linhas_cod = []
        for c in codigos[:3]:   # máximo 3 arquivos no prompt
            if hasattr(c, "nome_arquivo"):
                nome_arq = c.nome_arquivo
                linguagem = c.linguagem
                score     = c.score_qualidade
                conteudo  = c.conteudo
            else:
                nome_arq  = c.get("nome_arquivo", "?")
                linguagem = c.get("linguagem", "?")
                score     = c.get("score_qualidade", 0)
                conteudo  = c.get("conteudo", "")
            # Só inclui um trecho do código para não lotar o prompt
            trecho = conteudo[:200] + "..." if len(conteudo) > 200 else conteudo
            linhas_cod.append(
                f"arquivo: {nome_arq} ({linguagem}, score={score:.1f})\n{trecho}"
            )
        partes.append("[CODIGO]\n" + "\n\n".join(linhas_cod))

    # ── 4. Trechos recuperados pelo RAG ──────────────────────────────
    rag = ctx.get("rag", "").strip()
    if rag:
        partes.append(rag)

    # ── 5. Pergunta atual ─────────────────────────────────────────────
    pergunta = ctx.get("pergunta", "").strip()
    if pergunta:
        partes.append(f"[PERGUNTA]\n{pergunta}")

    # Junta tudo com separador duplo de linha
    return "\n\n".join(partes)

 




# -------------------------------------------------------
# LOOP INTERATIVO — roda direto no terminal
# -------------------------------------------------------

def main():
    """
    Loop de conversa interativo no terminal.
    Roda com: python brain.py

    Comandos especiais:
      /sair        → encerra
      /stats       → estatísticas da sessão
      /historico N → últimas N mensagens (padrão: 10)
      /nova        → nova sessão
      /debug       → toggle: mostra intenção e confiança detectadas
      /limpar      → limpa a tela
    """
    brain = ParadoxoBrain()
    debug_mode = False

    print("=" * 55)
    print("  PARADOXO X — Terminal")
    print("  Comandos especiais:")
    print("  /sair        → encerra")
    print("  /stats       → estatísticas da sessão")
    print("  /historico N → últimas N mensagens")
    print("  /nova        → nova sessão")
    print("  /debug       → toggle modo debug")
    print("  /limpar      → limpa a tela")
    print("  /rag <arq>   → indexa arquivo no RAG")
    print("=" * 55)
    print()

    while True:
        try:
            entrada = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not entrada:
            continue

        cmd = entrada.lower()

        # ── Comandos especiais ──
        if cmd in ("/sair", "/exit", "/quit"):
            break

        elif cmd == "/stats":
            brain.stats()
            continue

        elif cmd.startswith("/historico"):
            n = 10
            partes = entrada.split()
            if len(partes) > 1 and partes[1].isdigit():
                n = int(partes[1])
            brain.historico(n)
            continue

        elif cmd == "/nova":
            brain.nova_sessao()
            continue

        elif cmd == "/debug":
            debug_mode = not debug_mode
            estado = "ativado ✓" if debug_mode else "desativado"
            print(f"🔧 Debug {estado}\n")
            continue

        elif cmd == "/limpar":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        elif cmd.startswith("/rag"):
            partes = entrada.split(maxsplit=1)
            if len(partes) < 2:
                print("Uso: /rag <caminho_do_arquivo>\n")
            else:
                caminho_rag = partes[1].strip()
                if not os.path.exists(caminho_rag):
                    print(f"❌ Arquivo não encontrado: {caminho_rag}\n")
                else:
                    try:
                        with open(caminho_rag, encoding="utf-8", errors="replace") as f:
                            conteudo_rag = f.read()
                        brain.indexar_documento(caminho_rag, conteudo_rag)
                    except Exception as e:
                        print(f"❌ Erro ao indexar: {e}\n")
            continue

        # ── Processa mensagem ──
        if debug_mode:
            meta = brain.chat(entrada, retornar_meta=True)
            print(
                f"[debug] intenção={meta['intencao']} "
                f"confiança={meta['confianca']:.2f} "
                f"linguagem={meta['linguagem']}"
            )
            print(f"\nParadoxoX: {meta['resposta']}\n")
        else:
            resposta = brain.chat(entrada)
            print(f"\nParadoxoX: {resposta}\n")

    brain.fechar()


if __name__ == "__main__":
    main()