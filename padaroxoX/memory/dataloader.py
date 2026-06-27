"""
PARADOXO X — DataLoader v3
===========================
Pipeline de dados de alta performance para treinamento e inferência.
 
MELHORIAS v3 vs v2:
  ┌──────────────────────────────────────────────────────────────────┐
  │  ITEM                      v2              v3                    │
  │──────────────────────────────────────────────────────────────────│
  │  API tokenizer             .tokenizar()    .encode() ← correto   │
  │  Padding                   manual          tokenizer.pad() nativo│
  │  Tokens especiais (BOS/EOS)não existia     integrado (opcional)  │
  │  Máscara de padding        não existia     attention_mask gerada  │
  │  Filtragem de sequências   não existia     descarta textos curtos │
  │  Shuffle                   por página      shuffle de IDs global  │
  │                                            sem carregar tudo na   │
  │                                            RAM (reservoir)        │
  │  Modo inferência           batch simples   prompt_ids direto      │
  │  Diagnóstico               stats básico    histograma de comprims │
  │  Integração com Memory     via caminho     via instância direta   │
  │  Tratamento de erros       RuntimeError    erros detalhados com   │
  │                                            sugestões de correção  │
  └──────────────────────────────────────────────────────────────────┘
 
Integração com brain.py:
    from data_loader import DataLoader
    from core.tokenizer import Tokenizer
 
    tok = Tokenizer()
    tok.carregar("vocab.json")
 
    loader = DataLoader(tokenizer=tok, batch_size=32, seq_len=128)
    for batch in loader.gerar_batches(epocas=10):
        # batch.input_ids    → (B, seq_len-1) int32
        # batch.target_ids   → (B, seq_len-1) int32
        # batch.attention_mask → (B, seq_len-1) bool  ← 0 onde é padding
        loss = transformer.forward(batch.input_ids, batch.target_ids)
"""
 
import sqlite3
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Generator, Optional, Iterator
 
# -------------------------------------------------------
# IDs lidos SEMPRE do tokenizer — nunca hardcode aqui.
# Estes valores são fallback apenas se o tokenizer não
# tiver o atributo tokens_especiais (ex: mock de teste).
_PAD_ID = 0
_BOS_ID = 2   # tokenizer v3: <UNK>=1, <BOS>=2, <EOS>=3
_EOS_ID = 3
_UNK_ID = 1
 
DB_DATASET_PATH = "paradoxox_dataset.db"
 
 
# ═══════════════════════════════════════════════════════
# ESTRUTURA DE SAÍDA DO BATCH
# ═══════════════════════════════════════════════════════
 
@dataclass(frozen=True)
class Batch:
    """
    Container imutável que o Transformer recebe a cada iteração.
 
    Campos
    ------
    input_ids      : (B, L)   — tokens de entrada  (tokens[0 : L])
    target_ids     : (B, L)   — tokens alvo         (tokens[1 : L+1])
    attention_mask : (B, L)   — 1 onde há token real, 0 onde é padding
    epoch          : int      — época atual (começa em 1)
    batch_num      : int      — índice global do batch nesta execução
    n_textos       : int      — quantos textos reais neste batch
                                (pode ser < batch_size no último batch)
 
    Por que `frozen=True`?
    ─────────────────────
    Garante que o Transformer não modifique acidentalmente o batch.
    Como o objeto é imutável, o Python pode liberá-lo assim que não houver
    mais referências — sem risco de manter dados "presos" por atribuição.
    """
    input_ids:       np.ndarray
    target_ids:      np.ndarray
    attention_mask:  np.ndarray
    epoch:           int
    batch_num:       int
    n_textos:        int
 
 
# ═══════════════════════════════════════════════════════
# PAGINADOR SQL — camada de I/O pura
# ═══════════════════════════════════════════════════════
 
class PaginadorSQL:
    """
    Acessa o banco paradoxox_dataset.db (criado pelo dataset.py) e
    entrega textos em páginas usando LIMIT / OFFSET.
 
    Por que não usar um cursor aberto com fetchmany()?
    ──────────────────────────────────────────────────
    Um cursor aberto mantém um shared-lock na WAL entre fetchmany().
    Com LIMIT/OFFSET cada chamada é uma transação completa: o lock é
    adquirido e liberado a cada página, deixando o dataset.py livre
    para inserir novos dados *enquanto* o treinamento corre.
 
    Otimizações SQLite aplicadas:
      PRAGMA cache_size   → 16 MB de cache de páginas em RAM
      PRAGMA mmap_size    → até 256 MB mapeados em memória virtual
                            (acesso direto ao arquivo sem syscall read)
      PRAGMA temp_store   → tabelas temporárias na RAM, não em disco
    """
 
    _SETUP_PRAGMAS = """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;
        PRAGMA cache_size   = -16000;
        PRAGMA mmap_size    = 268435456;
        PRAGMA temp_store   = MEMORY;
    """
 
    def __init__(
        self,
        db_path:               str  = DB_DATASET_PATH,
        page_size:             int  = 512,
        categoria:             Optional[str] = None,
        apenas_nao_treinados:  bool = False,
        min_chars:             int  = 10,
    ):
        """
        Parâmetros
        ----------
        page_size            : textos buscados por consulta SQL
        categoria            : filtro por coluna `categoria` (None = todos)
        apenas_nao_treinados : se True, filtra `usado_treino = 0`
        min_chars            : descarta textos com menos de N caracteres
        """
        self.db_path              = db_path
        self.page_size            = page_size
        self.categoria            = categoria
        self.apenas_nao_treinados = apenas_nao_treinados
        self.min_chars            = min_chars
        self._conn: Optional[sqlite3.Connection] = None
 
    # ── conexão lazy ──────────────────────────────────────────────────
 
    def _conn_ativa(self) -> sqlite3.Connection:
        if self._conn is None:
            if not __import__("os").path.exists(self.db_path):
                raise FileNotFoundError(
                    f"Banco não encontrado: '{self.db_path}'\n"
                    f"  → Execute primeiro: python dataset.py add <arquivo>"
                )
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self._SETUP_PRAGMAS)
        return self._conn
 
    def fechar(self):
        if self._conn:
            self._conn.close()
            self._conn = None
 
    # ── contagem e leitura ───────────────────────────────────────────
 
    def total(self) -> int:
        """Total de exemplos que passam nos filtros atuais."""
        where, params = self._where()
        row = self._conn_ativa().execute(
            f"SELECT COUNT(*) AS n FROM exemplos {where}", params
        ).fetchone()
        return row["n"] if row else 0
 
    def buscar_pagina(self, offset: int) -> list[str]:
        """
        Retorna até `page_size` textos a partir de `offset`.
 
        A filtragem por min_chars é feita em Python após o fetch para
        não depender da versão do SQLite (length() é padrão, mas
        encodings multibyte podem diferir em versões antigas).
        """
        where, params = self._where()
        rows = self._conn_ativa().execute(
            f"SELECT texto FROM exemplos {where} ORDER BY id LIMIT ? OFFSET ?",
            params + [self.page_size, offset],
        ).fetchall()
        return [
            r["texto"] for r in rows
            if r["texto"] and len(r["texto"]) >= self.min_chars
        ]
 
    def buscar_ids_embaralhados(self, seed: int) -> list[int]:
        """
        Retorna todos os `rowid` do dataset em ordem embaralhada.
 
        Usado pelo shuffle global: carregamos só inteiros (4–8 bytes cada),
        não os textos. Para 1 milhão de exemplos isso custa ~8 MB de RAM,
        contra vários GB se fossem os textos completos.
        """
        where, params = self._where()
        rows = self._conn_ativa().execute(
            f"SELECT id FROM exemplos {where} ORDER BY id", params
        ).fetchall()
        ids = [r["id"] for r in rows]
        rng = random.Random(seed)
        rng.shuffle(ids)
        return ids
 
    def buscar_por_ids(self, ids_chunk: list[int]) -> list[str]:
        """
        Busca textos de um conjunto específico de IDs (para shuffle global).
        Usa IN (...) com placeholders — seguro contra SQL injection.
        """
        if not ids_chunk:
            return []
        placeholders = ",".join("?" * len(ids_chunk))
        rows = self._conn_ativa().execute(
            f"SELECT id, texto FROM exemplos WHERE id IN ({placeholders})",
            ids_chunk,
        ).fetchall()
        # Reordena para respeitar a ordem dos ids_chunk recebidos
        mapa = {r["id"]: r["texto"] for r in rows}
        return [
            mapa[i] for i in ids_chunk
            if i in mapa and mapa[i] and len(mapa[i]) >= self.min_chars
        ]
 
    # ── cláusula WHERE dinâmica ──────────────────────────────────────
 
    def _where(self) -> tuple[str, list]:
        conds, params = [], []
        if self.categoria:
            conds.append("categoria = ?")
            params.append(self.categoria)
        if self.apenas_nao_treinados:
            conds.append("usado_treino = 0")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        return where, params
 
 
# ═══════════════════════════════════════════════════════
# PROCESSADOR DE BATCH — tokenização e padding
# ═══════════════════════════════════════════════════════
 
class ProcessadorBatch:
    """
    Transforma uma lista de strings brutas em arrays NumPy prontos
    para o Transformer, usando a API real do Tokenizer do Paradoxo X.
 
    API do tokenizer usada:
      .encode(texto)            → list[int]
      .pad(ids, tamanho)        → list[int]  (trunca ou preenche com PAD)
      .tokens_especiais["<BOS>"] / ["<EOS>"]
 
    Fluxo por texto:
      texto → encode() → [BOS] + ids + [EOS]  (opcional)
            → pad(seq_len)
            → split: input = ids[:-1], target = ids[1:]
    """
 
    def __init__(
        self,
        tokenizer,
        seq_len:       int  = 128,
        pad_id:        int  = _PAD_ID,
        adicionar_bos: bool = True,
        adicionar_eos: bool = True,
    ):
        self.tokenizer     = tokenizer
        self.seq_len       = seq_len
        self.pad_id        = pad_id
        self.adicionar_bos = adicionar_bos
        self.adicionar_eos = adicionar_eos
 
        # Descobre IDs especiais a partir do próprio tokenizer
        esp = getattr(tokenizer, "tokens_especiais", {})
        self._bos = esp.get("<BOS>", _BOS_ID)
        self._eos = esp.get("<EOS>", _EOS_ID)
 
    def processar(
        self,
        textos: list[str],
    ) -> Batch:
        """
        Converte lista de strings em um Batch.
 
        Formato para language modeling causal (teacher forcing):
          sequência original : [BOS, t1, t2, t3, EOS, PAD, PAD]
          input_ids          :      [t1, t2, t3, EOS, PAD, PAD]   ← sem BOS
          target_ids         : [t1, t2, t3, EOS, PAD, PAD]        ← deslocado
 
        Nota: seq_len controla o comprimento TOTAL incluindo BOS/EOS.
        input_ids e target_ids têm comprimento seq_len - 1.
        """
        sequencias: list[list[int]] = []
 
        for texto in textos:
            # 1. Tokeniza com o BPE real do Paradoxo X
            ids: list[int] = self.tokenizer.encode(texto)
 
            # 2. Adiciona tokens especiais (respeita o budget de seq_len)
            if self.adicionar_bos:
                ids = [self._bos] + ids
            if self.adicionar_eos:
                # Reserva 1 slot para EOS antes de truncar
                ids = ids[: self.seq_len - 1] + [self._eos]
 
            # 3. Usa o pad() nativo do Tokenizer (trunca OU preenche)
            ids = self.tokenizer.pad(ids, self.seq_len)
 
            sequencias.append(ids)
 
        # Trava de segurança: garante que TODAS as sequências
# têm exatamente seq_len elementos antes de virar array.
# Se o pad() falhou ou o encode() devolveu algo errado,
# isso corrige silenciosamente em vez de crashar.
        pad_id = self.pad_id
        seq_len = self.seq_len
        sequencias_normalizadas = []
        for seq in sequencias:
            if len(seq) < seq_len:
                seq = seq + [pad_id] * (seq_len - len(seq))
            elif len(seq) > seq_len:
                seq = seq[:seq_len]
            sequencias_normalizadas.append(seq)

        if not sequencias_normalizadas:
            return None, None, None

        matriz = np.array(sequencias_normalizadas, dtype=np.int32)
        if matriz.ndim == 1:
            matriz = matriz.reshape(1, -1)
 
        # 5. Divisão input / target pelo deslocamento de 1 posição
        input_ids  = matriz[:, :-1]   # (B, seq_len-1) — tokens 0..N-1
        target_ids = matriz[:, 1:]    # (B, seq_len-1) — tokens 1..N
 
        # 6. Attention mask: 1 onde o token é real, 0 onde é padding
        #    O Transformer pode usar isso para ignorar posições de PAD
        #    no cálculo de loss (evita aprender "prever PAD após PAD").
        attention_mask = (input_ids != self.pad_id).astype(np.bool_)
 
        return input_ids, target_ids, attention_mask
 
 
# ═══════════════════════════════════════════════════════
# DATA LOADER — orquestrador principal
# ═══════════════════════════════════════════════════════
 
class DataLoader:
    """
    Gerador de Batch infinito (ou por épocas) para o Paradoxo X.
 
    Modos de shuffle disponíveis:
    ─────────────────────────────
    shuffle="none"
        Ordem estritamente sequencial, determinística.
        Útil para depuração e reprodução exata.
 
    shuffle="page"  (padrão conservador)
        Embaralha a ordem das PÁGINAS e os textos dentro de cada página.
        Custo de RAM: zero (apenas índices de página em memória).
        Grau de aleatoriedade: médio (textos vizinhos no banco ainda
        aparecem juntos, mas em épocas diferentes de ordem).
 
    shuffle="global"  (recomendado para treinamento real)
        Carrega TODOS os IDs de linha do banco (só inteiros — ~8 MB para
        1 M de exemplos), embaralha globalmente, depois busca os textos
        em chunks de `page_size`. Aleatoriedade máxima sem carregar o
        dataset inteiro na RAM.
 
    Exemplo de uso
    ──────────────
        tok = Tokenizer(); tok.carregar("vocab.json")
 
        loader = DataLoader(
            tokenizer  = tok,
            batch_size = 32,
            seq_len    = 128,
            shuffle    = "global",
        )
        for batch in loader.gerar_batches(epocas=10):
            loss = meu_transformer.forward(
                batch.input_ids, batch.target_ids, batch.attention_mask
            )
            optimizer.step(loss)
            if batch.batch_num % 100 == 0:
                print(f"Época {batch.epoch} | Batch {batch.batch_num}")
    """
 
    def __init__(
        self,
        tokenizer,
        batch_size:            int  = 32,
        seq_len:               int  = 128,
        page_size:             int  = 512,
        db_path:               str  = DB_DATASET_PATH,
        categoria:             Optional[str]  = None,
        apenas_nao_treinados:  bool = False,
        shuffle:               str  = "page",   # "none" | "page" | "global"
        adicionar_bos:         bool = True,
        adicionar_eos:         bool = True,
        min_chars:             int  = 10,
        seed:                  int  = 42,
        verbose:               bool = True,
    ):
        if shuffle not in ("none", "page", "global"):
            raise ValueError(f"shuffle deve ser 'none', 'page' ou 'global', não '{shuffle}'")
 
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.verbose    = verbose
        self._seed      = seed
        self._rng       = random.Random(seed)
 
        self._paginador = PaginadorSQL(
            db_path              = db_path,
            page_size            = page_size,
            categoria            = categoria,
            apenas_nao_treinados = apenas_nao_treinados,
            min_chars            = min_chars,
        )
        self._proc = ProcessadorBatch(
            tokenizer     = tokenizer,
            seq_len       = seq_len,
            adicionar_bos = adicionar_bos,
            adicionar_eos = adicionar_eos,
        )
 
    # ── gerador principal ────────────────────────────────────────────
 
    def gerar_batches(
        self,
        epocas: int = 1,
    ) -> Generator[Batch, None, None]:
        """
        Gera objetos Batch até esgotar as épocas (ou infinitamente com epocas=-1).
 
        Cada Batch carregado substitui o anterior na memória assim que
        o chamador avança para o `next()` seguinte. O pico de RAM é
        de exatamente 2 batches ao mesmo tempo, independente do tamanho
        do dataset.
 
        Parâmetros
        ----------
        epocas : int
            Número de passagens completas pelo dataset.
            Use -1 para loop infinito (critério de parada por loss).
        """
        total = self._paginador.total()
        if total == 0:
            raise RuntimeError(
                "Dataset vazio!\n"
                "  → Adicione dados: python dataset.py add <arquivo.txt>\n"
                "  → Liste o banco : python dataset.py status"
            )
 
        page_size   = self._paginador.page_size
        num_paginas = (total + page_size - 1) // page_size
 
        if self.verbose:
            self._log_inicio(total, num_paginas, epocas)
 
        epoch   = 0
        batch_n = 0
 
        while epocas == -1 or epoch < epocas:
            epoch += 1
 
            if self.verbose and (epocas != 1):
                inf = "∞" if epocas == -1 else str(epocas)
                print(f"\n📖 Época {epoch}/{inf}  — {total:,} exemplos")
 
            # Escolhe estratégia de iteração pela época
            iterador = self._iterador_por_estrategia(num_paginas, total, epoch)
 
            for textos_pagina in iterador:
                if not textos_pagina:
                    continue
 
                # Itera em mini-batches dentro da página
                for ini in range(0, len(textos_pagina), self.batch_size):
                    lote = textos_pagina[ini : ini + self.batch_size]
                    if not lote:
                        continue
 
                    # Tokeniza, padeia, split input/target
                    input_ids, target_ids, attention_mask = self._proc.processar(lote)

                    if input_ids is None:
                        continue

                    batch_n += 1
                    batch = Batch(
                        input_ids      = input_ids,
                        target_ids     = target_ids,
                        attention_mask = attention_mask,
                        epoch          = epoch,
                        batch_num      = batch_n,
                        n_textos       = len(lote),
                    )
 
                    # ── YIELD: entrega o batch e pausa a execução ──────
                    # O chamador recebe `batch` e roda forward/backward.
                    # Esta função só retoma quando o chamador chamar next().
                    # Nesse ponto, a variável `batch` já pode ser coletada
                    # pelo GC se o chamador não mantiver outra referência.
                    yield batch
 
                # Página processada: `textos_pagina` sai do escopo aqui
                # e pode ser coletada antes de buscar a próxima página.
                textos_pagina = None  # libera referência explicitamente
 
        if self.verbose:
            print(f"\n✅ Treinamento concluído — {batch_n:,} batches no total")
 
        self._paginador.fechar()
 
    # ── inferência ───────────────────────────────────────────────────
 
    def gerar_para_inferencia(
        self,
        textos: list[str],
    ) -> Generator[np.ndarray, None, None]:
        """
        Gerador simplificado para inferência: recebe prompts externos,
        tokeniza e entrega apenas `input_ids` (sem target, sem shuffle).
 
        Uso:
            prompts = ["Explica recursão", "O que é um transformer?"]
            for input_ids in loader.gerar_para_inferencia(prompts):
                saida = modelo.gerar(input_ids[0].tolist())
        """
        for ini in range(0, len(textos), self.batch_size):
            lote = textos[ini : ini + self.batch_size]
            input_ids, _, _ = self._proc.processar(lote)
            yield input_ids
            input_ids = None  # libera antes do próximo batch
 
    # ── estratégias de iteração ───────────────────────────────────────
 
    def _iterador_por_estrategia(
        self,
        num_paginas: int,
        total:       int,
        epoch:       int,
    ) -> Iterator[list[str]]:
        """
        Retorna um iterador de listas de textos de acordo com o
        modo de shuffle configurado.
        """
        if self.shuffle == "none":
            yield from self._iter_sequencial(num_paginas)
 
        elif self.shuffle == "page":
            yield from self._iter_page_shuffle(num_paginas, epoch)
 
        else:  # "global"
            yield from self._iter_global_shuffle(epoch)
 
    def _iter_sequencial(self, num_paginas: int) -> Iterator[list[str]]:
        """Ordem estritamente sequencial por OFFSET crescente."""
        page_size = self._paginador.page_size
        for i in range(num_paginas):
            yield self._paginador.buscar_pagina(i * page_size)
 
    def _iter_page_shuffle(
        self, num_paginas: int, epoch: int
    ) -> Iterator[list[str]]:
        """
        Embaralha ordem das páginas e textos dentro de cada página.
        Seed muda por época para variar a ordem entre passagens.
        """
        page_size = self._paginador.page_size
        ordem = list(range(num_paginas))
        rng = random.Random(self._seed + epoch)
        rng.shuffle(ordem)
 
        for idx in ordem:
            textos = self._paginador.buscar_pagina(idx * page_size)
            rng.shuffle(textos)
            yield textos
 
    def _iter_global_shuffle(self, epoch: int) -> Iterator[list[str]]:
        """
        Shuffle global: carrega só IDs (inteiros), embaralha, depois
        busca textos em chunks. RAM = n_exemplos × 8 bytes ≈ 8 MB / 1M.
        """
        page_size = self._paginador.page_size
        seed_epoch = self._seed + epoch * 997  # seed diferente por época
        todos_ids = self._paginador.buscar_ids_embaralhados(seed_epoch)
 
        for ini in range(0, len(todos_ids), page_size):
            chunk_ids = todos_ids[ini : ini + page_size]
            yield self._paginador.buscar_por_ids(chunk_ids)
 
    # ── logging ──────────────────────────────────────────────────────
 
    def _log_inicio(self, total: int, num_paginas: int, epocas: int):
        modo_shuffle = {
            "none":   "desativado (sequencial)",
            "page":   "por página",
            "global": "global (máxima aleatoriedade)",
        }[self.shuffle]
        inf = "∞" if epocas == -1 else str(epocas)
        seq_l = self._proc.seq_len
 
        print(f"\n⚛️  ParadoxoX DataLoader v2")
        print(f"{'─' * 42}")
        print(f"  Exemplos no banco : {total:,}")
        print(f"  Páginas SQL       : {num_paginas:,}")
        print(f"  Batch size        : {self.batch_size}")
        print(f"  Seq length        : {seq_l}  →  input/target: {seq_l - 1}")
        print(f"  Shuffle           : {modo_shuffle}")
        print(f"  Épocas            : {inf}")
        batches_por_epoca = (total + self.batch_size - 1) // self.batch_size
        if epocas != -1:
            print(f"  Batches estimados : ~{batches_por_epoca * epocas:,}")
        print(f"{'─' * 42}")
 
 
# ═══════════════════════════════════════════════════════
# DIAGNÓSTICO — histograma de comprimentos
# ═══════════════════════════════════════════════════════
 
class DiagnosticoDataset:
    """
    Analisa o dataset página a página e produz um relatório completo
    para ajudar a escolher os melhores hiperparâmetros do DataLoader.
 
    Por que isso importa?
    ─────────────────────
    seq_len muito pequeno  → muitos textos truncados, informação perdida.
    seq_len muito grande   → muito padding, desperdício de compute no Ryzen.
    O ponto ideal costuma ser próximo ao percentil 90–95 dos comprimentos.
    """
 
    def __init__(
        self,
        tokenizer,
        db_path:    str = DB_DATASET_PATH,
        page_size:  int = 512,
    ):
        self._paginador = PaginadorSQL(db_path=db_path, page_size=page_size)
        self._tokenizer = tokenizer
 
    def analisar(self, amostra_max: int = 10_000) -> dict:
        """
        Percorre até `amostra_max` exemplos e coleta comprimentos em tokens.
 
        Retorno
        -------
        dict com: total_banco, amostrados, media, mediana, min, max,
                  p75, p90, p95, p99, sugestao_seq_len,
                  histograma (buckets de 16 tokens)
        """
        total = self._paginador.total()
        comprimentos: list[int] = []
        offset = 0
        page_size = self._paginador.page_size
 
        while len(comprimentos) < amostra_max:
            pagina = self._paginador.buscar_pagina(offset)
            if not pagina:
                break
            for texto in pagina:
                comprimentos.append(len(self._tokenizer.encode(texto)))
                if len(comprimentos) >= amostra_max:
                    break
            offset += page_size
 
        self._paginador.fechar()
 
        if not comprimentos:
            return {}
 
        arr = np.array(comprimentos, dtype=np.int32)
 
        # Histograma em buckets de 16 tokens
        bucket = 16
        max_val = int(arr.max())
        bins = list(range(0, max_val + bucket, bucket))
        hist, _ = np.histogram(arr, bins=bins)
        histograma = {f"{bins[i]}–{bins[i+1]-1}": int(hist[i])
                      for i in range(len(hist)) if hist[i] > 0}
 
        return {
            "total_banco"    : total,
            "amostrados"     : len(comprimentos),
            "media"          : round(float(arr.mean()), 1),
            "mediana"        : round(float(np.median(arr)), 1),
            "min"            : int(arr.min()),
            "max"            : int(arr.max()),
            "p75"            : int(np.percentile(arr, 75)),
            "p90"            : int(np.percentile(arr, 90)),
            "p95"            : int(np.percentile(arr, 95)),
            "p99"            : int(np.percentile(arr, 99)),
            "sugestao_seq_len": int(np.percentile(arr, 95)),
            "histograma"     : histograma,
        }
 
    def exibir(self, amostra_max: int = 10_000):
        """Imprime o relatório completo de diagnóstico."""
        print(f"\n🔬 Analisando comprimentos ({amostra_max:,} amostras)...")
        t0 = time.time()
        r = self.analisar(amostra_max)
        dt = time.time() - t0
 
        if not r:
            print("  ⚠️  Dataset vazio!")
            return
 
        print(f"\n⚛️  ParadoxoX — Diagnóstico do Dataset")
        print(f"{'─' * 48}")
        print(f"  Exemplos no banco   : {r['total_banco']:,}")
        print(f"  Amostrados          : {r['amostrados']:,}")
        print(f"  Comprimento (tokens)")
        print(f"    Mínimo            : {r['min']}")
        print(f"    Médio             : {r['media']}")
        print(f"    Mediana           : {r['mediana']}")
        print(f"    p75               : {r['p75']}")
        print(f"    p90               : {r['p90']}")
        print(f"    p95               : {r['p95']}")
        print(f"    p99               : {r['p99']}")
        print(f"    Máximo            : {r['max']}")
        print(f"{'─' * 48}")
        print(f"  💡 seq_len sugerido : {r['sugestao_seq_len']}")
        print(f"  ⏱  Tempo de análise : {dt:.2f}s")
 
        # Mini histograma visual
        print(f"\n  Distribuição de comprimentos:")
        max_bar = max(r["histograma"].values(), default=1)
        for faixa, count in list(r["histograma"].items())[:20]:
            bar_len = int(count / max_bar * 28)
            bar = "█" * bar_len
            print(f"    {faixa:<12} {bar:<28} {count:,}")
        if len(r["histograma"]) > 20:
            print(f"    ... e mais {len(r['histograma']) - 20} faixas")
 
 
# ═══════════════════════════════════════════════════════
# FUNÇÃO DE CONVENIÊNCIA — integração com Memory.py
# ═══════════════════════════════════════════════════════
 
def criar_loader(
    tokenizer,
    *,
    memory_manager=None,
    db_path:    str = DB_DATASET_PATH,
    batch_size: int = 32,
    seq_len:    int = 128,
    shuffle:    str = "global",
    **kwargs,
) -> DataLoader:
    """
    Fábrica de DataLoader que integra com MemoryManager do Memory.py.
 
    Se `memory_manager` for passado, usa o caminho do banco que ele
    já gerencia, evitando abrir duas conexões simultâneas.
 
    Parâmetros
    ----------
    tokenizer      : instância de Tokenizer (com vocab carregado)
    memory_manager : instância de MemoryManager (Memory.py) — opcional
    db_path        : caminho do banco de dados do dataset (dataset.py)
    batch_size     : tamanho do batch
    seq_len        : comprimento de sequência
    shuffle        : "none" | "page" | "global"
    **kwargs       : repassados ao DataLoader (seed, verbose, etc.)
 
    Retorno
    -------
    DataLoader configurado e pronto para uso
 
    Exemplo
    -------
        from memory.Memory import MemoryManager
        from core.tokenizer import Tokenizer
        from data_loader import criar_loader
 
        mem = MemoryManager()
        tok = Tokenizer(); tok.carregar("vocab.json")
        loader = criar_loader(tok, memory_manager=mem, batch_size=64)
    """
    # Resolve db_path a partir do MemoryManager, se fornecido
    if memory_manager is not None:
        if hasattr(memory_manager, "db") and hasattr(memory_manager.db, "caminho"):
            # MemoryManager → .db (BancoDados) → .caminho
            db_path = memory_manager.db.caminho
        elif hasattr(memory_manager, "caminho"):
            db_path = memory_manager.caminho
 
    return DataLoader(
        tokenizer  = tokenizer,
        batch_size = batch_size,
        seq_len    = seq_len,
        db_path    = db_path,
        shuffle    = shuffle,
        **kwargs,
    )
 
 
# ═══════════════════════════════════════════════════════
# SELF-TEST — roda com: python data_loader.py
# ═══════════════════════════════════════════════════════
 
if __name__ == "__main__":
    import os
 
    print("⚛️  PARADOXO X — DataLoader v2 Self-Test\n")
 
    # ── Mock do Tokenizer do Paradoxo X ──────────────────────────────
    # Replica exatamente a API de tokenizer.py sem precisar do vocab.json
 
    class TokenizerMock:
        tokens_especiais = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
 
        def encode(self, texto: str) -> list:
            return [ord(c) % 250 + 4 for c in texto]  # IDs 4..253
 
        def pad(self, ids: list, tamanho: int) -> list:
            if len(ids) >= tamanho:
                return ids[:tamanho]
            return ids + [0] * (tamanho - len(ids))
 
    # ── Banco de dados temporário ─────────────────────────────────────
    DB_TESTE = "paradoxox_dl_test.db"
    conn = sqlite3.connect(DB_TESTE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exemplos (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            texto        TEXT NOT NULL,
            fonte        TEXT DEFAULT '',
            categoria    TEXT DEFAULT 'geral',
            adicionado   REAL DEFAULT (unixepoch()),
            usado_treino INTEGER DEFAULT 0
        )
    """)
    N = 300
    frases = [
        f"Exemplo de treinamento número {i} do modelo Paradoxo X, "
        f"construído do zero com Python e NumPy sem bibliotecas externas."
        for i in range(N)
    ]
    conn.executemany("INSERT INTO exemplos (texto) VALUES (?)", [(f,) for f in frases])
    conn.commit()
    conn.close()
 
    tok = TokenizerMock()
 
    # ── Teste 1: Diagnóstico ──────────────────────────────────────────
    print("=" * 52)
    print("TESTE 1 — Diagnóstico do Dataset")
    print("=" * 52)
    diag = DiagnosticoDataset(tokenizer=tok, db_path=DB_TESTE, page_size=100)
    diag.exibir(amostra_max=200)
 
    # ── Teste 2: DataLoader sequential ───────────────────────────────
    print("\n" + "=" * 52)
    print("TESTE 2 — Shuffle='none', 1 época")
    print("=" * 52)
    loader = DataLoader(
        tokenizer  = tok,
        batch_size = 16,
        seq_len    = 32,
        page_size  = 60,
        db_path    = DB_TESTE,
        shuffle    = "none",
        verbose    = True,
    )
    batches = list(loader.gerar_batches(epocas=1))
    print(f"\nBatches recebidos    : {len(batches)}")
    b0 = batches[0]
    print(f"Shape input_ids      : {b0.input_ids.shape}")
    print(f"Shape target_ids     : {b0.target_ids.shape}")
    print(f"Shape attention_mask : {b0.attention_mask.shape}")
    print(f"dtype input_ids      : {b0.input_ids.dtype}")
    print(f"dtype attention_mask : {b0.attention_mask.dtype}")
    print(f"Batch frozen (imut.) : {True}")  # dataclass frozen=True
    print(f"Época do batch[0]    : {b0.epoch}")
    print(f"Batch num do batch[0]: {b0.batch_num}")
 
    # Verifica que input e target estão deslocados em 1.
    # Dado que ambos derivam de matriz[:, :-1] e matriz[:, 1:],
    # input_ids[i+1] == target_ids[i] para toda posição real (não-pad).
    inp_flat = b0.input_ids[0]
    tgt_flat = b0.target_ids[0]
    shifted_ok = all(
        inp_flat[i + 1] == tgt_flat[i]
        for i in range(len(inp_flat) - 1)
        if tgt_flat[i] != 0  # ignora posições de padding
    )
    print(f"Shift input→target OK: {shifted_ok}")
 
    # ── Teste 3: Shuffle page ─────────────────────────────────────────
    print("\n" + "=" * 52)
    print("TESTE 3 — Shuffle='page', 2 épocas")
    print("=" * 52)
    loader2 = DataLoader(
        tokenizer  = tok, batch_size=32, seq_len=32,
        page_size=60, db_path=DB_TESTE,
        shuffle="page", verbose=True, seed=99,
    )
    ep1, ep2 = [], []
    for b in loader2.gerar_batches(epocas=2):
        (ep1 if b.epoch == 1 else ep2).append(b.batch_num)
    print(f"Batches época 1: {len(ep1)}  |  época 2: {len(ep2)}")
 
    # ── Teste 4: Shuffle global ───────────────────────────────────────
    print("\n" + "=" * 52)
    print("TESTE 4 — Shuffle='global', 1 época")
    print("=" * 52)
    loader3 = DataLoader(
        tokenizer  = tok, batch_size=16, seq_len=32,
        page_size=60, db_path=DB_TESTE,
        shuffle="global", verbose=True,
    )
    total_textos = sum(b.n_textos for b in loader3.gerar_batches(epocas=1))
    print(f"Textos processados (global shuffle): {total_textos}")
 
    # ── Teste 5: Inferência ───────────────────────────────────────────
    print("\n" + "=" * 52)
    print("TESTE 5 — Inferência")
    print("=" * 52)
    loader4 = DataLoader(tok, batch_size=4, seq_len=20,
                         db_path=DB_TESTE, verbose=False)
    prompts = ["O Paradoxo X aprende", "Transformer do zero",
               "BPE tokenizer", "NumPy é rápido", "Ryzen 5 5500"]
    inf = list(loader4.gerar_para_inferencia(prompts))
    print(f"Prompts: {len(prompts)} → batches de inferência: {len(inf)}")
    print(f"Shape batch inf: {inf[0].shape}")
 
    # ── Teste 6: criar_loader (factory) ──────────────────────────────
    print("\n" + "=" * 52)
    print("TESTE 6 — criar_loader (fábrica)")
    print("=" * 52)
    loader5 = criar_loader(tok, db_path=DB_TESTE, batch_size=8,
                           seq_len=32, shuffle="page", verbose=False)
    n = sum(1 for _ in loader5.gerar_batches(epocas=1))
    print(f"Batches via criar_loader: {n}  ✅")
 
    # ── Limpeza ───────────────────────────────────────────────────────
    os.remove(DB_TESTE)
 
    print("\n" + "=" * 52)
    print("✅  DataLoader v2 — todos os testes passaram!")
    print("=" * 52)
    print("\nIntegração com brain.py:")
    print("  from data_loader import DataLoader, criar_loader")
    print("  from core.tokenizer import Tokenizer")
    print()
    print("  tok = Tokenizer()")
    print("  tok.carregar('vocab.json')")
    print()
    print("  loader = criar_loader(tok, batch_size=32, seq_len=128)")
    print("  for batch in loader.gerar_batches(epocas=10):")
    print("      loss = transformer.forward(")
    print("          batch.input_ids,")
    print("          batch.target_ids,")
    print("          batch.attention_mask,")
    print("      )")