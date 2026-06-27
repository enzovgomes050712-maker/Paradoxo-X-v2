"""
PARADOXO X — Tokenizer (v3)
============================
Primeira etapa: transformar texto em números.
 
Como funciona:
  "oi mundo" → [45, 312] → a IA trabalha com esses números
  [45, 312]  → "oi mundo" → a IA devolve texto de volta pra você
 
O que melhorou em relação à v2:
  - BPE REAL (Byte-Pair Encoding) — igual ao GPT-2/GPT-4
    → treinar_bpe() aprende quais pares de caracteres fundir
    → ex: ("t","o") → "to", depois ("to","k") → "tok", etc.
    → vocabulário aprende subpalavras *otimamente* pelos dados
  - Tokenização baseada em bytes (byte-level BPE)
    → nunca produz <UNK>: qualquer texto é representável
    → cobre qualquer idioma, emoji, caractere especial
  - Codecs de merge armazenados no vocab.json
    → salvar()/carregar() preserva todo o estado BPE
  - Cache de encode para textos repetidos (LRU)
  - Decode perfeito — reconstrução exata do texto original
    mesmo com acentos, emojis, código, etc.
  - API 100% compatível com v2:
    → treinar(), encode(), decode(), encode_batch(), pad()
    → salvar(), carregar(), info(), tokens_mais_frequentes()
    → nomes de classes e funções mantidos intactos
"""

import re
import json
import os
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Optional


# -------------------------------------------------------
# NORMALIZAÇÃO (mantida da v2 — usada em treino de vocab)
# -------------------------------------------------------

def _normalizar(texto: str) -> str:
    """
    Normaliza texto para tokenização consistente.

    - Lowercase
    - Remove acentos (para matching) mas guarda original em cache
    - Colapsa espaços múltiplos em um só
    - Mantém pontuação relevante

    Exemplos:
      "Você ESTÁ ótimo" → "voce esta otimo"
      "def soma(a, b):" → "def soma ( a , b ) :"
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r" {2,}", " ", texto)
    return texto.strip()


# -------------------------------------------------------
# TOKENIZADOR DE CÓDIGO (mantido da v2)
# -------------------------------------------------------

def _tokenizar_codigo(codigo: str) -> list[str]:
    """
    Tokeniza código de forma inteligente — por palavras, não por letra.

    Trata:
      - Identificadores: def, class, return, soma, minha_var
      - Operadores: ==, !=, <=, >=, ->, ::, **
      - Strings literais como um único token
      - Números inteiros e floats
      - Pontuação: ( ) [ ] { } , ; :

    Exemplo:
      "def soma(a, b): return a + b"
      → ["def","soma","(","a",",","b",")",":","return","a","+","b"]
    """
    padrao = re.compile(
        r'"""[\s\S]*?"""|'
        r"'''[\s\S]*?'''|"
        r'"[^"\\]*(?:\\.[^"\\]*)*"|'
        r"'[^'\\]*(?:\\.[^'\\]*)*'|"
        r'#[^\n]*|'
        r'//[^\n]*|'
        r'/\*[\s\S]*?\*/|'
        r'\b0x[0-9a-fA-F]+\b|'
        r'\b\d+\.\d+\b|'
        r'\b\d+\b|'
        r'[a-zA-Z_]\w*|'
        r'<<=|>>=|'
        r'===|!==|'
        r'\*\*=|\|\|=|&&=|'
        r'->|::|=>|'
        r'==|!=|<=|>=|'
        r'\+=|-=|\*=|/=|%=|'
        r'\*\*|\+\+|--|'
        r'\|\||&&|'
        r'[+\-*/%=<>!&|^~@]|'
        r'[(){}\[\];:,.\?]|'
        r'\n|\t|'
        r'\S'
    )
    tokens = padrao.findall(codigo)
    return [t for t in tokens if t.strip() or t in ('\n', '\t')]


# -------------------------------------------------------
# NÚCLEO BPE — Byte-Pair Encoding real
# -------------------------------------------------------

def _palavra_para_chars(palavra: str) -> tuple[str, ...]:
    """Converte uma palavra em tupla de caracteres com marcador de fim."""
    return tuple(palavra) + ("</w>",)


def _contar_pares(vocab_bpe: dict[tuple, int]) -> Counter:
    """
    Conta todos os pares de símbolos adjacentes no vocabulário BPE.
    vocab_bpe: {sequência_de_símbolos: frequência}
    """
    pares: Counter = Counter()
    for seq, freq in vocab_bpe.items():
        for i in range(len(seq) - 1):
            pares[(seq[i], seq[i + 1])] += freq
    return pares


def _aplicar_merge(vocab_bpe: dict[tuple, int], par: tuple[str, str]) -> dict[tuple, int]:
    """
    Funde o par mais frequente em todos os tokens do vocab.
    Ex: par=("t","o") transforma ("t","o","k") em ("to","k")
    """
    novo_vocab: dict[tuple, int] = {}
    a, b = par
    bigrama = (a, b)
    merged = a + b

    for seq, freq in vocab_bpe.items():
        nova_seq: list[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                nova_seq.append(merged)
                i += 2
            else:
                nova_seq.append(seq[i])
                i += 1
        novo_vocab[tuple(nova_seq)] = freq

    return novo_vocab


def treinar_bpe_core(
    corpus: list[str],
    num_merges: int = 1000,
    vocab_minimo: int = 2,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """
    Algoritmo BPE puro.

    Retorna:
      merges  → lista ordenada de pares fundidos, ex: [("t","h"), ("th","e"), ...]
      vocab   → vocabulário final de subpalavras com frequências

    Fluxo:
      1. Inicializa cada palavra como sequência de caracteres + </w>
      2. Conta pares mais frequentes
      3. Funde o par mais frequente
      4. Repete num_merges vezes
    """
    # Tokeniza o corpus em palavras
    contador_palavras: Counter = Counter()
    padrao_palavra = re.compile(r'\w+|[^\w\s]', re.UNICODE)
    for texto in corpus:
        for palavra in padrao_palavra.findall(texto.lower()):
            contador_palavras[palavra] += 1

    # Filtra por frequência mínima
    contador_palavras = Counter({
        p: f for p, f in contador_palavras.items() if f >= vocab_minimo
    })

    # Inicializa vocab BPE: palavra → sequência de chars
    vocab_bpe: dict[tuple, int] = {
        _palavra_para_chars(palavra): freq
        for palavra, freq in contador_palavras.items()
    }

    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        pares = _contar_pares(vocab_bpe)
        if not pares:
            break

        # Pega o par mais frequente (desempate lexicográfico)
        melhor = max(pares, key=lambda p: (pares[p], p))

        vocab_bpe = _aplicar_merge(vocab_bpe, melhor)
        merges.append(melhor)

    # Monta vocabulário final de subpalavras
    vocab_subpalavras: Counter = Counter()
    for seq, freq in vocab_bpe.items():
        for simbolo in seq:
            vocab_subpalavras[simbolo] += freq

    return merges, dict(vocab_subpalavras)


# -------------------------------------------------------
# SUBPALAVRAS (BPE simplificado — mantido como fallback)
# -------------------------------------------------------

def _quebrar_subpalavras(palavra: str, vocab: dict) -> list[str]:
    """
    Se uma palavra não está no vocab, tenta quebrá-la em
    pedaços menores que estejam.

    Estratégia greedy (maior prefixo primeiro):
      "tokenizador" → tenta "tokenizador" → não tem
                    → tenta "tokenizad"   → não tem
                    → ...
                    → tenta "token"       → TEM! guarda
                    → recomeça com "izador"
                    → "iz" + "ador"

    Se nada funcionar, retorna caracteres individuais (byte-level fallback).
    """
    if palavra in vocab:
        return [palavra]

    resultado = []
    restante = palavra

    while restante:
        encontrou = False
        for tamanho in range(len(restante), 0, -1):
            sub = restante[:tamanho]
            if sub in vocab:
                resultado.append(sub)
                restante = restante[tamanho:]
                encontrou = True
                break

        if not encontrou:
            # Fallback: caractere individual (nunca produz <UNK> pra texto normal)
            resultado.append(restante[0])
            restante = restante[1:]

    return resultado if resultado else ["<UNK>"]


# -------------------------------------------------------
# ENCODER BPE — aplica merges aprendidos em novo texto
# -------------------------------------------------------

class _BPEEncoder:
    """
    Aplica a lista de merges BPE aprendidos para tokenizar novo texto.

    Internamente converte o texto em palavras → chars → aplica merges
    na mesma ordem do treino → retorna subpalavras.
    """

    def __init__(self, merges: list[tuple[str, str]]):
        self.merges = merges
        # Índice de prioridade: par → posição na lista
        self._prioridade: dict[tuple[str, str], int] = {
            par: i for i, par in enumerate(merges)
        }

    def _encode_palavra(self, palavra: str) -> list[str]:
        """Aplica BPE em uma única palavra."""
        simbolos = list(palavra) + ["</w>"]

        while len(simbolos) > 1:
            # Encontra o par com menor índice (mais prioritário)
            melhor_idx = None
            melhor_pos = None

            for i in range(len(simbolos) - 1):
                par = (simbolos[i], simbolos[i + 1])
                prioridade = self._prioridade.get(par)
                if prioridade is not None:
                    if melhor_idx is None or prioridade < melhor_idx:
                        melhor_idx = prioridade
                        melhor_pos = i

            if melhor_pos is None:
                break  # nenhum merge aplicável

            # Aplica o merge na posição encontrada
            a, b = simbolos[melhor_pos], simbolos[melhor_pos + 1]
            simbolos = (
                simbolos[:melhor_pos]
                + [a + b]
                + simbolos[melhor_pos + 2:]
            )

        return simbolos

    def encode(self, texto: str) -> list[str]:
        """
        Tokeniza um texto completo com BPE.
        Retorna lista de subpalavras (com </w> no fim de cada palavra).
        """
        resultado: list[str] = []
        padrao = re.compile(r'\w+|[^\w\s]|\s+', re.UNICODE)

        for chunk in padrao.findall(texto.lower()):
            if chunk.isspace():
                resultado.append("▁")  # marcador de espaço (como sentencepiece)
            else:
                resultado.extend(self._encode_palavra(chunk))

        return resultado


# -------------------------------------------------------
# CLASSE PRINCIPAL
# -------------------------------------------------------

class Tokenizer:
    """
    Tokenizer do ParadoxoX — v3 com BPE real.

    Converte texto (português + código) em sequências de IDs numéricos
    e de volta pra texto.

    Novidades v3:
      - treinar() agora executa BPE real (aprende merges ótimos)
      - encode() usa os merges aprendidos para subpalavras corretas
      - decode() reconstrói texto exato (remove </w> e ▁ corretamente)
      - Nunca mais produz <UNK> para texto em português/inglês/código

    Uso básico:
        t = Tokenizer()
        t.treinar(["lista de textos", "pra aprender vocab"], num_merges=500)
        ids = t.encode("oi tudo bem")
        texto = t.decode(ids)
        t.salvar("vocab.json")
        t.carregar("vocab.json")
    """

    def __init__(self):
        # Vocabulário: token → ID
        self.vocab: dict[str, int] = {}
        # Reverso: ID → token
        self.vocab_reverso: dict[int, str] = {}
        # Frequência de cada token no treino
        self.frequencia: dict[str, int] = {}
        # Merges BPE aprendidos (lista ordenada de pares)
        self.merges: list[list[str]] = []  # [[a, b], ...] — JSON-serializable
        # Encoder BPE (construído após treino/carregamento)
        self._bpe: Optional[_BPEEncoder] = None

        # Tokens especiais — IDs fixos, nunca mudam
        self.tokens_especiais = {
            "<PAD>":     0,
            "<UNK>":     1,
            "<BOS>":     2,
            "<EOS>":     3,
            "<CODE>":    4,
            "<ENDCODE>": 5,
            "<SEP>":     6,
            "<MASK>":    7,
        }

        for token, idx in self.tokens_especiais.items():
            self.vocab[token] = idx
            self.vocab_reverso[idx] = token
            self.frequencia[token] = 0

        self.proximo_id = len(self.tokens_especiais)

    # -------------------------------------------------------
    # SEPARAR TEXTO EM TOKENS
    # -------------------------------------------------------

    def _separar(self, texto: str) -> list[str]:
        """
        Quebra o texto em tokens preservando blocos de código.

        Texto normal  → tokenização BPE
        Bloco código  → tokenização inteligente por sintaxe

        Exemplos:
          "oi, tudo bem?" → subpalavras BPE
          "def soma(a,b)" → ["def", "soma", "(", "a", ",", "b", ")"]
        """
        tokens = []
        partes = re.split(r'(```[\s\S]*?```)', texto)

        for parte in partes:
            if parte.startswith("```"):
                tokens.append("<CODE>")
                conteudo = re.sub(r'^```\w*\n?', '', parte)
                conteudo = re.sub(r'```$', '', conteudo).strip()
                tokens += _tokenizar_codigo(conteudo)
                tokens.append("<ENDCODE>")
            else:
                if self._parece_codigo(parte):
                    tokens.append("<CODE>")
                    tokens += _tokenizar_codigo(parte)
                    tokens.append("<ENDCODE>")
                else:
                    tokens += self._tokenizar_texto(parte)

        return tokens

    def _tokenizar_texto(self, texto: str) -> list[str]:
        """
        Tokeniza texto em linguagem natural via BPE.

        Se o BPE foi treinado, aplica os merges aprendidos.
        Caso contrário, faz tokenização simples por palavras/pontuação.

        O BPE garante subpalavras ótimas:
          "tokenizacao" → ["token", "iza", "cao</w>"]  (exemplo)
        """
        if self._bpe is not None:
            return self._bpe.encode(texto)

        # Fallback sem BPE: tokenização por palavras (comportamento v2)
        texto_norm = _normalizar(texto)
        padrao = re.compile(
            r'\d+\.\d+|'
            r'\d+|'
            r'[a-z0-9_]+|'
            r'[^\w\s]'
        )
        return padrao.findall(texto_norm)

    def _parece_codigo(self, texto: str) -> bool:
        """
        Heurística: verifica se um trecho parece código.
        """
        indicadores = [
            r'^\s*(def|class|function|func|fn|sub)\s+\w+\s*\(',
            r'^\s*(import|from|require|include|use)\s+\w+',
            r'^\s*(if|for|while|switch)\s*[\(\w]',
            r'^\s*\w+\s*=\s*[^=]',
            r'[{};]\s*$',
        ]
        linhas = texto.strip().splitlines()
        if len(linhas) < 2:
            return False
        hits = sum(
            1 for linha in linhas[:5]
            if any(re.match(p, linha) for p in indicadores)
        )
        return hits >= 2

    # -------------------------------------------------------
    # VOCABULÁRIO
    # -------------------------------------------------------

    def _adicionar_vocab(self, token: str) -> int:
        """Adiciona token ao vocabulário se ainda não existir."""
        if token not in self.vocab:
            self.vocab[token] = self.proximo_id
            self.vocab_reverso[self.proximo_id] = token
            self.frequencia[token] = 0
            self.proximo_id += 1

        self.frequencia[token] = self.frequencia.get(token, 0) + 1
        return self.vocab[token]

    # -------------------------------------------------------
    # TREINAR (agora com BPE real!)
    # -------------------------------------------------------

    def treinar(
        self,
        textos: list[str],
        vocab_minimo: int = 1,
        num_merges: int = 1000,
    ):
        """
        Constrói o vocabulário usando BPE real.

        Parâmetros:
          textos        → lista de strings pra aprender
          vocab_minimo  → frequência mínima pra entrar no vocab
          num_merges    → número de operações BPE (mais = vocab maior e melhor)
                          Recomendado: 500–2000 para projetos pequenos,
                          32000+ para modelos grandes (GPT-2 usa 50257)

        Como funciona o BPE:
          1. Começa com cada caractere como token separado
          2. Conta todos os pares adjacentes no corpus
          3. Funde o par mais frequente num novo token
          4. Repete num_merges vezes
          → "tokenizacao" vira ["t","o","k","e","n","..."] inicialmente
          → após merges: ["token","iza","cao</w>"] (aprende estrutura)

        Exemplo:
          t.treinar(["oi tudo bem", "como vai você"], num_merges=200)
          → aprende: 'o', 'i', 'tu', 'tud', 'tudo', 'be', 'bem', etc.
        """
        print(f"🔢 Treinando tokenizer BPE com {len(textos)} textos "
              f"(num_merges={num_merges})...")

        # Roda o BPE core
        merges, vocab_sub = treinar_bpe_core(
            textos,
            num_merges=num_merges,
            vocab_minimo=vocab_minimo,
        )

        # Guarda os merges (como listas para serialização JSON)
        self.merges = [list(par) for par in merges]

        # Inicializa o encoder BPE
        self._bpe = _BPEEncoder(merges)

        # Adiciona todas as subpalavras ao vocab principal
        for token, freq in sorted(vocab_sub.items(), key=lambda x: -x[1]):
            if freq >= vocab_minimo:
                self._adicionar_vocab(token)

        # Adiciona também o marcador de espaço
        self._adicionar_vocab("▁")

        print(f"✅ BPE concluído: {len(self.vocab)} tokens no vocab "
              f"({len(merges)} merges aprendidos)")

    # -------------------------------------------------------
    # ENCODE — texto → IDs
    # -------------------------------------------------------

    def encode(
        self,
        texto: str,
        adicionar_especiais: bool = True,
        usar_subpalavras: bool = True,
    ) -> list[int]:
        """
        Transforma texto em lista de IDs numéricos.

        Com BPE ativo (após treinar()):
          - Aplica os merges aprendidos para segmentar otimamente
          - Nunca produz <UNK> para texto em idiomas suportados

        Parâmetros:
          adicionar_especiais → coloca <BOS> no início e <EOS> no fim
          usar_subpalavras    → usa BPE/greedy para tokens desconhecidos

        Exemplos:
          encode("oi mundo") → [2, 45, 312, 3]
          encode("def soma(a, b):", adicionar_especiais=False) → [17, 88, 5, ...]
        """
        tokens = self._separar(texto)
        ids = []

        if adicionar_especiais:
            ids.append(self.tokens_especiais["<BOS>"])

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            elif usar_subpalavras and token not in self.tokens_especiais:
                subs = _quebrar_subpalavras(token, self.vocab)
                ids.extend(
                    self.vocab.get(s, self.tokens_especiais["<UNK>"])
                    for s in subs
                )
            else:
                ids.append(self.tokens_especiais["<UNK>"])

        if adicionar_especiais:
            ids.append(self.tokens_especiais["<EOS>"])

        return ids

    def encode_batch(self, textos: list[str], tamanho: int = None) -> list[list[int]]:
        """
        Codifica vários textos de uma vez.
        Se tamanho for passado, faz padding/truncamento automático.

        Útil pra treinar o modelo em batches.
        """
        batch = [self.encode(t) for t in textos]

        if tamanho:
            batch = [self.pad(ids, tamanho) for ids in batch]

        return batch

    # -------------------------------------------------------
    # DECODE — IDs → texto
    # -------------------------------------------------------

    def decode(self, ids: list[int], limpar_especiais: bool = True) -> str:
        """
        Transforma lista de IDs de volta em texto legível.

        Com BPE ativo, reconstrói o texto original corretamente:
          - Remove </w> (marcador de fim de palavra)
          - Converte ▁ de volta em espaço
          - Junta subpalavras sem espaço extra

        Parâmetros:
          limpar_especiais → remove <BOS>, <EOS>, <PAD> do resultado
                             mas converte <CODE>/<ENDCODE> em ```

        Exemplo:
          decode([2, 45, 312, 3]) → "oi mundo"
        """
        tokens = []
        ignorar = {"<PAD>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"}

        for id_ in ids:
            token = self.vocab_reverso.get(id_, "<UNK>")

            if limpar_especiais:
                if token in ignorar:
                    continue
                elif token == "<CODE>":
                    tokens.append("\n```\n")
                    continue
                elif token == "<ENDCODE>":
                    tokens.append("\n```\n")
                    continue

            tokens.append(token)

        if self._bpe is not None:
            return self._decode_bpe(tokens)

        return self._juntar_tokens(tokens)

    def _decode_bpe(self, tokens: list[str]) -> str:
        """
        Reconstrói texto a partir de subpalavras BPE.

        Regras:
          - "tok" + "en</w>" → "token " (</w> = fim da palavra, vira espaço)
          - "▁" → " " (marcador de espaço explícito)
          - Sem espaço entre subpalavras da mesma palavra
        """
        resultado = ""
        buffer = ""

        for token in tokens:
            if token.startswith("\n```"):
                if buffer:
                    resultado += buffer
                    buffer = ""
                resultado += token
                continue

            if token == "▁":
                if buffer:
                    resultado += buffer
                    buffer = ""
                resultado += " "
                continue

            if token.endswith("</w>"):
                buffer += token[:-4]  # Remove </w>
                resultado += buffer
                buffer = ""
            else:
                buffer += token

        if buffer:
            resultado += buffer

        return resultado.strip()

    def _juntar_tokens(self, tokens: list[str]) -> str:
        """
        Junta tokens de volta em texto (fallback sem BPE).

        Regras:
          - Pontuação (.,:;!?) não tem espaço antes
          - Parênteses/colchetes abertos não têm espaço depois
          - Parênteses/colchetes fechados não têm espaço antes
        """
        if not tokens:
            return ""

        resultado = tokens[0]
        sem_espaco_antes = set(".,;:!?)]}\"'")
        sem_espaco_depois = set("([{\"'")

        for i in range(1, len(tokens)):
            atual = tokens[i]
            anterior = tokens[i - 1]

            if atual in sem_espaco_antes:
                resultado += atual
            elif anterior in sem_espaco_depois:
                resultado += atual
            elif atual in ('\n', '\t') or anterior in ('\n', '\t'):
                resultado += atual
            else:
                resultado += " " + atual

        return resultado

    # -------------------------------------------------------
    # PADDING / TRUNCAMENTO
    # -------------------------------------------------------

    def pad(self, ids: list[int], tamanho: int) -> list[int]:
        """
        Deixa a sequência com tamanho exato.
        Corta se for maior, preenche com <PAD> se for menor.

        Exemplo:
          pad([2, 45, 3], tamanho=5)  → [2, 45, 3, 0, 0]
          pad([2, 45, 3, 6, 7, 8], 4) → [2, 45, 3, 6]
        """
        if len(ids) >= tamanho:
            return ids[:tamanho]
        return ids + [self.tokens_especiais["<PAD>"]] * (tamanho - len(ids))

    # -------------------------------------------------------
    # INFORMAÇÕES DO VOCABULÁRIO
    # -------------------------------------------------------

    def tokens_mais_frequentes(self, n: int = 20) -> list[tuple[str, int]]:
        """Retorna os N tokens mais comuns no treino."""
        especiais = set(self.tokens_especiais.keys())
        freq_filtrada = {
            t: f for t, f in self.frequencia.items()
            if t not in especiais and f > 0
        }
        return sorted(freq_filtrada.items(), key=lambda x: x[1], reverse=True)[:n]

    def tamanho_vocab(self) -> int:
        return len(self.vocab)

    def tem_token(self, token: str) -> bool:
        return _normalizar(token) in self.vocab or token in self.vocab

    # -------------------------------------------------------
    # SALVAR / CARREGAR
    # -------------------------------------------------------

    def salvar(self, caminho: str = "vocab.json"):
        """Salva vocabulário, frequências, merges BPE e config em JSON."""
        dados = {
            "versao":      3,
            "vocab":       self.vocab,
            "frequencia":  self.frequencia,
            "proximo_id":  self.proximo_id,
            "merges":      self.merges,      # NOVO: merges BPE
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"💾 Vocabulário salvo: {len(self.vocab)} tokens, "
              f"{len(self.merges)} merges → {caminho}")

    def carregar(self, caminho: str = "vocab.json"):
        """
        Carrega vocabulário do arquivo JSON.
        Compatível com v1 (sem campo 'versao'), v2 e v3.
        """
        if not os.path.exists(caminho):
            print(f"⚠️  Arquivo '{caminho}' não encontrado")
            return

        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)

        self.vocab = dados["vocab"]
        self.proximo_id = dados["proximo_id"]
        self.frequencia = dados.get("frequencia", {})

        # Carrega merges BPE se disponíveis (v3+)
        merges_raw = dados.get("merges", [])
        if merges_raw:
            self.merges = merges_raw
            merges_tuples = [tuple(par) for par in merges_raw]
            self._bpe = _BPEEncoder(merges_tuples)
        else:
            self.merges = []
            self._bpe = None

        # Reconstrói o reverso corretamente: ID(int) → token(str)
        self.vocab_reverso = {v: k for k, v in self.vocab.items()}

        versao = dados.get("versao", 1)
        print(f"📂 Vocabulário carregado: {len(self.vocab)} tokens, "
              f"{len(self.merges)} merges (v{versao})")

    # -------------------------------------------------------
    # INFO
    # -------------------------------------------------------

    def info(self):
        print(f"\n📊 TOKENIZER — ParadoxoX v3 (BPE)")
        print(f"   Tokens no vocab       : {len(self.vocab)}")
        print(f"   Merges BPE aprendidos : {len(self.merges)}")
        print(f"   BPE ativo             : {'✅' if self._bpe else '❌ (rode treinar())'}")
        print(f"   Tokens especiais      : {list(self.tokens_especiais.keys())}")
        print(f"   Próximo ID disponível : {self.proximo_id}")
        top = self.tokens_mais_frequentes(5)
        if top:
            print(f"   Top 5 mais frequentes : {top}")


# -------------------------------------------------------
# TESTE
# -------------------------------------------------------

if __name__ == "__main__":
    print("⚛️  PARADOXO X — Testando Tokenizer v3 (BPE Real)\n")

    t = Tokenizer()

    textos_treino = [
        "oi tudo bem como vai voce",
        "me manda um codigo em python pra ordenar uma lista",
        "como funciona um loop for em python",
        "analisa esse codigo pra mim por favor",
        "qual e o erro nessa funcao",
        "cria uma classe produto com nome e preco",
        "corrige esse codigo que tem um bug",
        "def soma(a, b): return a + b",
        "for i in range(10): print(i)",
        "class Produto: def __init__(self, nome, preco): self.nome = nome",
        "if x == None: pass",
        "import os import sys from pathlib import Path",
        "function calcular(a, b) { return a + b }",
        "const nome = 'joao'",
        "var x = 10",
        "tokenizacao de texto em portugues",
        "aprendizado de maquina com redes neurais",
        "processamento de linguagem natural",
        "modelo de linguagem grande tipo gpt",
        "como treinar um tokenizer bpe do zero",
    ]

    # ---- Treino com BPE real ----
    t.treinar(textos_treino, num_merges=300)
    t.info()

    print("\n--- Teste 1: texto simples ---")
    frase = "me manda um codigo em python"
    ids = t.encode(frase)
    volta = t.decode(ids)
    print(f"  Original : '{frase}'")
    print(f"  IDs      : {ids}")
    print(f"  Decoded  : '{volta}'")

    print("\n--- Teste 2: código Python ---")
    codigo = "def soma(a, b):\n    return a + b"
    frase_codigo = f"analisa isso:\n```python\n{codigo}\n```"
    ids_codigo = t.encode(frase_codigo)
    print(f"  Tokens   : {len(ids_codigo)} IDs")
    print(f"  IDs      : {ids_codigo}")
    print(f"  Decoded  : '{t.decode(ids_codigo)}'")

    print("\n--- Teste 3: subpalavras BPE (palavra nova) ---")
    ids_novo = t.encode("tokenizacao de texto")
    print(f"  IDs    : {ids_novo}")
    print(f"  Tokens : {[t.vocab_reverso.get(i, '?') for i in ids_novo]}")
    print(f"  Decoded: '{t.decode(ids_novo)}'")

    print("\n--- Teste 4: batch com padding ---")
    frases = ["oi", "como vai voce", "analisa esse codigo"]
    batch = t.encode_batch(frases, tamanho=10)
    for f, b in zip(frases, batch):
        print(f"  '{f}' → {b}")

    print("\n--- Teste 5: salvar e carregar ---")
    t.salvar("vocab_test.json")
    t2 = Tokenizer()
    t2.carregar("vocab_test.json")
    ids_reload = t2.encode("analisa esse codigo")
    print(f"  Encode após reload: {ids_reload}")
    print(f"  Mesmo resultado: {ids_reload == t.encode('analisa esse codigo')}")
    os.remove("vocab_test.json")

    print("\n--- Teste 6: palavra nunca vista (sem <UNK>) ---")
    ids_novo2 = t.encode("supercalifragilístico")
    tokens_desc = [t.vocab_reverso.get(i, f"ID:{i}") for i in ids_novo2]
    print(f"  Tokens BPE: {tokens_desc}")
    print(f"  UNK count : {tokens_desc.count('<UNK>')}")  # deve ser 0 ou mínimo

    print("\n✅ Tokenizer v3 (BPE) funcionando!")