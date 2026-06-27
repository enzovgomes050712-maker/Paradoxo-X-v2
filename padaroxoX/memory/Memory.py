"""
PARADOXO X — Memory System
===========================
Sistema de memória completo do ParadoxoX.

Guarda tudo que importa entre sessões:
  1. HISTÓRICO      → cada mensagem da conversa (usuário + IA)
  2. CONTEXTO       → arquivos analisados, projetos abertos, erros vistos
  3. PREFERÊNCIAS   → nome do usuário, linguagem favorita, estilo, etc.

Por que SQLite?
  - Já vem no Python (zero instalação)
  - Um arquivo só (paradoxox_memory.db)
  - Suporta queries complexas (busca por data, por tipo, etc.)
  - ACID: dados nunca corrompem mesmo se o programa fechar do nada
  - Até 281 TB de dados — não vai faltar espaço tão cedo

Estrutura do banco:
  ┌─────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
  │   historico     │   │  contexto_codigo     │   │   preferencias      │
  │─────────────────│   │──────────────────────│   │─────────────────────│
  │ id              │   │ id                   │   │ chave               │
  │ sessao_id       │   │ sessao_id            │   │ valor               │
  │ papel (user/ai) │   │ tipo                 │   │ tipo_valor          │
  │ conteudo        │   │ nome_arquivo         │   │ atualizado_em       │
  │ timestamp       │   │ linguagem            │   └─────────────────────┘
  │ tokens_aprox    │   │ conteudo             │
  │ metadados       │   │ score_qualidade      │
  └─────────────────┘   │ problemas_json       │
                        │ timestamp            │
                        └──────────────────────┘
"""

import sqlite3
import json
import time
import hashlib
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any


# -------------------------------------------------------
# ESTRUTURAS DE DADOS
# -------------------------------------------------------

@dataclass
class Mensagem:
    """Uma mensagem no histórico de conversa."""
    papel: str          # "usuario" ou "ia"
    conteudo: str
    timestamp: float = field(default_factory=time.time)
    sessao_id: str = ""
    tokens_aprox: int = 0
    metadados: dict = field(default_factory=dict)

    def __post_init__(self):
        # Estima tokens (regra simples: ~4 chars por token)
        if not self.tokens_aprox:
            self.tokens_aprox = max(1, len(self.conteudo) // 4)

    def resumo(self, max_chars: int = 80) -> str:
        icone = "👤" if self.papel == "usuario" else "🤖"
        texto = self.conteudo[:max_chars]
        if len(self.conteudo) > max_chars:
            texto += "..."
        hora = datetime.fromtimestamp(self.timestamp).strftime("%H:%M")
        return f"{icone} [{hora}] {texto}"


@dataclass
class ContextoCodigo:
    """Contexto de um arquivo/projeto analisado."""
    tipo: str               # "arquivo", "projeto", "snippet", "erro"
    nome_arquivo: str = ""
    linguagem: str = ""
    conteudo: str = ""      # código ou resumo
    score_qualidade: float = 0.0
    problemas: list = field(default_factory=list)
    sessao_id: str = ""
    timestamp: float = field(default_factory=time.time)
    hash_conteudo: str = ""

    def __post_init__(self):
        if self.conteudo and not self.hash_conteudo:
            self.hash_conteudo = hashlib.md5(
                self.conteudo.encode("utf-8", errors="ignore")
            ).hexdigest()[:12]


@dataclass
class Preferencia:
    """Uma preferência do usuário."""
    chave: str
    valor: Any
    tipo_valor: str = "str"   # "str", "int", "float", "bool", "json"

    def valor_tipado(self) -> Any:
        """Retorna o valor com o tipo correto."""
        if self.tipo_valor == "int":
            return int(self.valor)
        elif self.tipo_valor == "float":
            return float(self.valor)
        elif self.tipo_valor == "bool":
            return str(self.valor).lower() in ("true", "1", "sim", "yes")
        elif self.tipo_valor == "json":
            return json.loads(self.valor) if isinstance(self.valor, str) else self.valor
        return str(self.valor)


# -------------------------------------------------------
# GERENCIADOR DE SESSÃO
# -------------------------------------------------------

class GerenciadorSessao:
    """
    Gera e rastreia IDs de sessão.

    Uma sessão = uma conversa contínua.
    Quando o usuário abre o programa, cria uma sessão nova.
    As sessões antigas ficam salvas e podem ser retomadas.
    """

    def __init__(self):
        self._sessao_atual: str = ""

    def nova_sessao(self) -> str:
        """Cria uma nova sessão com ID único baseado no timestamp."""
        ts = time.time()
        base = f"sess_{datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')}"
        # Adiciona sufixo hash pra garantir unicidade
        sufixo = hashlib.md5(str(ts).encode()).hexdigest()[:6]
        self._sessao_atual = f"{base}_{sufixo}"
        return self._sessao_atual

    def sessao_atual(self) -> str:
        if not self._sessao_atual:
            return self.nova_sessao()
        return self._sessao_atual

    def definir_sessao(self, sessao_id: str):
        self._sessao_atual = sessao_id

    @property
    def id(self) -> str:
        return self.sessao_atual()


# -------------------------------------------------------
# BANCO DE DADOS
# -------------------------------------------------------

class BancoDados:
    """
    Camada de acesso ao SQLite.
    Cuida de criar tabelas, conexão, e queries.
    """

    def __init__(self, caminho: str = "paradoxox_memory.db"):
        self.caminho = caminho
        self._conn: Optional[sqlite3.Connection] = None
        self._criar_tabelas()

    def _conectar(self) -> sqlite3.Connection:
        """Retorna conexão ativa (cria se não existe)."""
        if not self._conn:
            self._conn = sqlite3.connect(
                self.caminho,
                check_same_thread=False,   # permite uso em threads diferentes
                timeout=30
            )
            self._conn.row_factory = sqlite3.Row   # resultados como dicionários
            # Otimizações de performance
            self._conn.execute("PRAGMA journal_mode=WAL")   # writes mais rápidos
            self._conn.execute("PRAGMA synchronous=NORMAL") # mais rápido, ainda seguro
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _criar_tabelas(self):
        """Cria todas as tabelas se não existirem."""
        conn = self._conectar()
        conn.executescript("""
            -- Histórico de conversa
            CREATE TABLE IF NOT EXISTS historico (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                sessao_id     TEXT    NOT NULL,
                papel         TEXT    NOT NULL CHECK(papel IN ('usuario', 'ia')),
                conteudo      TEXT    NOT NULL,
                timestamp     REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
                tokens_aprox  INTEGER DEFAULT 0,
                metadados     TEXT    DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_hist_sessao
                ON historico(sessao_id, timestamp);

            -- Contexto de código analisado
            CREATE TABLE IF NOT EXISTS contexto_codigo (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                sessao_id       TEXT    NOT NULL,
                tipo            TEXT    NOT NULL,
                nome_arquivo    TEXT    DEFAULT '',
                linguagem       TEXT    DEFAULT '',
                conteudo        TEXT    DEFAULT '',
                hash_conteudo   TEXT    DEFAULT '',
                score_qualidade REAL    DEFAULT 0.0,
                problemas_json  TEXT    DEFAULT '[]',
                timestamp       REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec'))
            );
            CREATE INDEX IF NOT EXISTS idx_ctx_sessao
                ON contexto_codigo(sessao_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_ctx_arquivo
                ON contexto_codigo(nome_arquivo);

            -- Preferências do usuário (chave-valor persistente)
            CREATE TABLE IF NOT EXISTS preferencias (
                chave           TEXT    PRIMARY KEY,
                valor           TEXT    NOT NULL,
                tipo_valor      TEXT    DEFAULT 'str',
                atualizado_em   REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec'))
            );

            -- Sessões (metadados de cada conversa)
            CREATE TABLE IF NOT EXISTS sessoes (
                sessao_id       TEXT    PRIMARY KEY,
                iniciada_em     REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
                encerrada_em    REAL,
                total_mensagens INTEGER DEFAULT 0,
                resumo          TEXT    DEFAULT ''
            );
        """)
        conn.commit()

    def executar(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conectar().execute(sql, params)

    def executar_many(self, sql: str, params_list: list):
        conn = self._conectar()
        conn.executemany(sql, params_list)
        conn.commit()

    def commit(self):
        self._conectar().commit()

    def fechar(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# -------------------------------------------------------
# REPOSITÓRIOS (acesso a cada tabela)
# -------------------------------------------------------

class RepositorioHistorico:
    """CRUD para o histórico de conversa."""

    def __init__(self, db: BancoDados):
        self.db = db

    def salvar(self, msg: Mensagem) -> int:
        cur = self.db.executar(
            """INSERT INTO historico
               (sessao_id, papel, conteudo, timestamp, tokens_aprox, metadados)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                msg.sessao_id,
                msg.papel,
                msg.conteudo,
                msg.timestamp,
                msg.tokens_aprox,
                json.dumps(msg.metadados, ensure_ascii=False),
            )
        )
        self.db.commit()

        # Atualiza contador na tabela de sessões
        self.db.executar(
            """INSERT INTO sessoes (sessao_id, total_mensagens)
               VALUES (?, 1)
               ON CONFLICT(sessao_id) DO UPDATE SET
               total_mensagens = total_mensagens + 1""",
            (msg.sessao_id,)
        )
        self.db.commit()
        return cur.lastrowid

    def buscar_sessao(
        self,
        sessao_id: str,
        limite: int = 50,
        offset: int = 0
    ) -> list[Mensagem]:
        """Retorna as últimas N mensagens de uma sessão."""
        rows = self.db.executar(
            """SELECT * FROM historico
               WHERE sessao_id = ?
               ORDER BY timestamp DESC
               LIMIT ? OFFSET ?""",
            (sessao_id, limite, offset)
        ).fetchall()

        return [self._row_para_msg(r) for r in reversed(rows)]

    def buscar_recente(self, sessao_id: str, n: int = 10) -> list[Mensagem]:
        """Atalho para pegar as N mensagens mais recentes."""
        return self.buscar_sessao(sessao_id, limite=n)

    def buscar_por_termo(self, termo: str, sessao_id: str = None) -> list[Mensagem]:
        """Busca mensagens que contêm um termo (busca simples)."""
        if sessao_id:
            rows = self.db.executar(
                """SELECT * FROM historico
                   WHERE sessao_id = ? AND conteudo LIKE ?
                   ORDER BY timestamp DESC LIMIT 20""",
                (sessao_id, f"%{termo}%")
            ).fetchall()
        else:
            rows = self.db.executar(
                """SELECT * FROM historico
                   WHERE conteudo LIKE ?
                   ORDER BY timestamp DESC LIMIT 20""",
                (f"%{termo}%",)
            ).fetchall()
        return [self._row_para_msg(r) for r in rows]

    def total_tokens_sessao(self, sessao_id: str) -> int:
        """Total de tokens usados na sessão (estimativa)."""
        row = self.db.executar(
            "SELECT SUM(tokens_aprox) FROM historico WHERE sessao_id = ?",
            (sessao_id,)
        ).fetchone()
        return row[0] or 0

    def listar_sessoes(self, limite: int = 20) -> list[dict]:
        """Lista as sessões mais recentes com metadados."""
        rows = self.db.executar(
            """SELECT s.sessao_id, s.iniciada_em, s.total_mensagens, s.resumo,
                      h.conteudo as primeira_msg
               FROM sessoes s
               LEFT JOIN historico h ON h.sessao_id = s.sessao_id
                   AND h.id = (SELECT MIN(id) FROM historico WHERE sessao_id = s.sessao_id)
               ORDER BY s.iniciada_em DESC
               LIMIT ?""",
            (limite,)
        ).fetchall()
        return [dict(r) for r in rows]

    def _row_para_msg(self, row) -> Mensagem:
        return Mensagem(
            papel=row["papel"],
            conteudo=row["conteudo"],
            timestamp=row["timestamp"],
            sessao_id=row["sessao_id"],
            tokens_aprox=row["tokens_aprox"],
            metadados=json.loads(row["metadados"] or "{}"),
        )


class RepositorioContexto:
    """CRUD para o contexto de código."""

    def __init__(self, db: BancoDados):
        self.db = db

    def salvar(self, ctx: ContextoCodigo) -> int:
        cur = self.db.executar(
            """INSERT INTO contexto_codigo
               (sessao_id, tipo, nome_arquivo, linguagem, conteudo,
                hash_conteudo, score_qualidade, problemas_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ctx.sessao_id,
                ctx.tipo,
                ctx.nome_arquivo,
                ctx.linguagem,
                ctx.conteudo,
                ctx.hash_conteudo,
                ctx.score_qualidade,
                json.dumps(ctx.problemas, ensure_ascii=False),
                ctx.timestamp,
            )
        )
        self.db.commit()
        return cur.lastrowid

    def buscar_sessao(self, sessao_id: str, tipo: str = None) -> list[ContextoCodigo]:
        """Retorna contextos de uma sessão, opcionalmente filtrado por tipo."""
        if tipo:
            rows = self.db.executar(
                """SELECT * FROM contexto_codigo
                   WHERE sessao_id = ? AND tipo = ?
                   ORDER BY timestamp DESC""",
                (sessao_id, tipo)
            ).fetchall()
        else:
            rows = self.db.executar(
                """SELECT * FROM contexto_codigo
                   WHERE sessao_id = ?
                   ORDER BY timestamp DESC""",
                (sessao_id,)
            ).fetchall()
        return [self._row_para_ctx(r) for r in rows]

    def buscar_por_arquivo(self, nome_arquivo: str) -> list[ContextoCodigo]:
        """Retorna todos os contextos de um arquivo específico (histórico de análises)."""
        rows = self.db.executar(
            """SELECT * FROM contexto_codigo
               WHERE nome_arquivo = ?
               ORDER BY timestamp DESC""",
            (nome_arquivo,)
        ).fetchall()
        return [self._row_para_ctx(r) for r in rows]

    def ja_analisado(self, hash_conteudo: str) -> bool:
        """Verifica se esse exato código já foi analisado antes (evita reanalise)."""
        row = self.db.executar(
            "SELECT id FROM contexto_codigo WHERE hash_conteudo = ? LIMIT 1",
            (hash_conteudo,)
        ).fetchone()
        return row is not None

    def buscar_linguagens_usadas(self, sessao_id: str = None) -> list[dict]:
        """Retorna quais linguagens o usuário mais usa."""
        if sessao_id:
            rows = self.db.executar(
                """SELECT linguagem, COUNT(*) as total
                   FROM contexto_codigo
                   WHERE sessao_id = ? AND linguagem != ''
                   GROUP BY linguagem ORDER BY total DESC""",
                (sessao_id,)
            ).fetchall()
        else:
            rows = self.db.executar(
                """SELECT linguagem, COUNT(*) as total
                   FROM contexto_codigo
                   WHERE linguagem != ''
                   GROUP BY linguagem ORDER BY total DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def _row_para_ctx(self, row) -> ContextoCodigo:
        _raw = row["problemas_json"]
        return ContextoCodigo(
            tipo=row["tipo"],
            nome_arquivo=row["nome_arquivo"],
            linguagem=row["linguagem"],
            conteudo=row["conteudo"],
            hash_conteudo=row["hash_conteudo"],
            score_qualidade=row["score_qualidade"],
            problemas=json.loads(_raw) if _raw and _raw.strip() else [],
            sessao_id=row["sessao_id"],
            timestamp=row["timestamp"],
        )


class RepositorioPreferencias:
    """CRUD para preferências do usuário."""

    # Preferências padrão do ParadoxoX
    PADROES = {
        "nome_usuario":         ("Usuário",   "str"),
        "linguagem_favorita":   ("python",    "str"),
        "estilo_resposta":      ("tecnico",   "str"),   # "tecnico", "simples", "detalhado"
        "mostrar_emojis":       ("true",      "bool"),
        "max_historico":        ("50",        "int"),   # mensagens por sessão
        "tema":                 ("escuro",    "str"),
        "idioma":               ("pt-br",     "str"),
        "salvar_codigos":       ("true",      "bool"),  # salvar código analisado
        "nivel_detalhe_relat":  ("completo",  "str"),   # "resumo", "completo"
        "auto_corrigir":        ("false",     "bool"),  # corrigir automaticamente
        "linguagens_ativas":    ('["python","javascript","typescript"]', "json"),
    }

    def __init__(self, db: BancoDados):
        self.db = db
        self._cache: dict[str, Any] = {}
        self._inicializar_padroes()

    def _inicializar_padroes(self):
        """Insere preferências padrão se não existirem."""
        for chave, (valor, tipo) in self.PADROES.items():
            self.db.executar(
                """INSERT OR IGNORE INTO preferencias (chave, valor, tipo_valor)
                   VALUES (?, ?, ?)""",
                (chave, valor, tipo)
            )
        self.db.commit()

    def get(self, chave: str, padrao: Any = None) -> Any:
        """Lê uma preferência com cache em memória."""
        if chave in self._cache:
            return self._cache[chave]

        row = self.db.executar(
            "SELECT valor, tipo_valor FROM preferencias WHERE chave = ?",
            (chave,)
        ).fetchone()

        if not row:
            return padrao

        pref = Preferencia(chave=chave, valor=row["valor"], tipo_valor=row["tipo_valor"])
        valor = pref.valor_tipado()
        self._cache[chave] = valor
        return valor

    def set(self, chave: str, valor: Any, tipo: str = None):
        """Define ou atualiza uma preferência."""
        # Detecta tipo automaticamente se não informado
        if tipo is None:
            if isinstance(valor, bool):
                tipo = "bool"
                valor = str(valor).lower()
            elif isinstance(valor, int):
                tipo = "int"
            elif isinstance(valor, float):
                tipo = "float"
            elif isinstance(valor, (list, dict)):
                tipo = "json"
                valor = json.dumps(valor, ensure_ascii=False)
            else:
                tipo = "str"

        self.db.executar(
            """INSERT INTO preferencias (chave, valor, tipo_valor, atualizado_em)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(chave) DO UPDATE SET
               valor = excluded.valor,
               tipo_valor = excluded.tipo_valor,
               atualizado_em = excluded.atualizado_em""",
            (chave, str(valor), tipo, time.time())
        )
        self.db.commit()
        self._cache[chave] = Preferencia(chave=chave, valor=str(valor), tipo_valor=tipo).valor_tipado()

    def get_todas(self) -> dict[str, Any]:
        """Retorna todas as preferências como dicionário."""
        rows = self.db.executar("SELECT chave, valor, tipo_valor FROM preferencias").fetchall()
        return {
            row["chave"]: Preferencia(
                chave=row["chave"],
                valor=row["valor"],
                tipo_valor=row["tipo_valor"]
            ).valor_tipado()
            for row in rows
        }

    def resetar_tudo(self):
        """Reseta todas as preferências para os valores padrão."""
        self.db.executar("DELETE FROM preferencias")
        self._cache.clear()
        self._inicializar_padroes()
        print("⚠️  Preferências resetadas para os valores padrão.")

    # Atalhos para preferências comuns
    @property
    def nome_usuario(self) -> str:
        return self.get("nome_usuario", "Usuário")

    @nome_usuario.setter
    def nome_usuario(self, valor: str):
        self.set("nome_usuario", valor)

    @property
    def linguagem_favorita(self) -> str:
        return self.get("linguagem_favorita", "python")

    @linguagem_favorita.setter
    def linguagem_favorita(self, valor: str):
        self.set("linguagem_favorita", valor.lower())

    @property
    def estilo_resposta(self) -> str:
        return self.get("estilo_resposta", "tecnico")

    @property
    def max_historico(self) -> int:
        return self.get("max_historico", 50)


# -------------------------------------------------------
# EXTRATOR DE PREFERÊNCIAS DA CONVERSA
# -------------------------------------------------------

class ExtractorPreferencias:
    """
    Aprende preferências do usuário automaticamente
    a partir do que ele escreve na conversa.

    Exemplos:
      "me chamo João"          → nome_usuario = "João"
      "prefiro python"          → linguagem_favorita = "python"
      "pode me chamar de Gus"   → nome_usuario = "Gus"
      "odeio javascript"        → linguagem_evitar = "javascript"
    """

    LINGUAGENS = {
        "python", "javascript", "typescript", "java", "go", "rust",
        "c", "cpp", "c++", "csharp", "c#", "ruby", "php", "swift",
        "kotlin", "dart", "lua", "bash", "sql",
    }

    def extrair(self, texto: str) -> dict[str, Any]:
        """
        Analisa uma mensagem e retorna as preferências encontradas.
        Retorna dict vazio se nenhuma for encontrada.
        """
        prefs = {}
        texto_lower = texto.lower()

        # Nome do usuário
        padroes_nome = [
            r'(?:me\s+chamo|meu\s+nome\s+[eé]|pode\s+me\s+chamar\s+de|sou\s+o|sou\s+a)\s+([A-ZÀ-Ú][a-zà-ú]+)',
            r'(?:me\s+chamo|meu\s+nome\s+[eé])\s+(\w+)',
        ]
        for p in padroes_nome:
            m = re.search(p, texto, re.IGNORECASE)
            if m:
                prefs["nome_usuario"] = m.group(1).strip()
                break

        # Linguagem favorita
        padroes_lang = [
            r'(?:prefiro|gosto\s+de|trabalho\s+com|uso\s+muito|minha\s+linguagem\s+[eé])\s+(\w+)',
            r'(?:sou\s+(?:dev|desenvolvedor|programador)\s+(?:de\s+)?(\w+))',
        ]
        for p in padroes_lang:
            m = re.search(p, texto_lower)
            if m:
                lang = m.group(1).lower()
                if lang in self.LINGUAGENS:
                    prefs["linguagem_favorita"] = lang
                    break

        # Detecta linguagem mencionada mesmo sem frase de preferência
        # se o usuário mandar código numa linguagem específica várias vezes
        # (isso é feito pelo MemoryManager, não aqui)

        # Estilo de resposta
        if any(p in texto_lower for p in ["seja simples", "explica simples", "mais simples", "sem termos técnicos"]):
            prefs["estilo_resposta"] = "simples"
        elif any(p in texto_lower for p in ["seja detalhado", "explica tudo", "mais detalhes", "bem detalhado"]):
            prefs["estilo_resposta"] = "detalhado"
        elif any(p in texto_lower for p in ["seja técnico", "pode ser técnico", "fala técnico"]):
            prefs["estilo_resposta"] = "tecnico"

        # Emojis
        if any(p in texto_lower for p in ["sem emoji", "sem emojis", "não usa emoji"]):
            prefs["mostrar_emojis"] = False
        elif any(p in texto_lower for p in ["pode usar emoji", "usa emoji", "com emoji"]):
            prefs["mostrar_emojis"] = True

        return prefs


# -------------------------------------------------------
# GERENCIADOR DE CONTEXTO DE JANELA
# -------------------------------------------------------

class GerenciadorContextoJanela:
    """
    Decide quais mensagens do histórico incluir no contexto
    enviado ao modelo (janela de contexto limitada).

    Estratégia:
      1. Sempre inclui as N mensagens mais recentes
      2. Inclui mensagens relevantes ao que o usuário está perguntando
      3. Inclui um resumo das sessões anteriores se existir

    Quando o modelo tiver treino real, isso alimenta o input dele.
    """

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens

    def montar_contexto(
        self,
        mensagens: list[Mensagem],
        contextos: list[ContextoCodigo],
        prefs: dict,
        pergunta_atual: str = "",
    ) -> dict:
        """
        Monta o dicionário de contexto que vai ser passado ao modelo.

        Retorna:
          {
            "historico_recente": [...],
            "arquivos_relevantes": [...],
            "preferencias": {...},
            "tokens_usados": int,
          }
        """
        tokens_usados = 0
        resultado = {
            "historico_recente": [],
            "arquivos_relevantes": [],
            "preferencias": prefs,
            "tokens_usados": 0,
        }

        # Adiciona mensagens mais recentes primeiro (até o limite)
        for msg in reversed(mensagens):
            if tokens_usados + msg.tokens_aprox > self.max_tokens * 0.7:
                break
            resultado["historico_recente"].insert(0, {
                "papel": msg.papel,
                "conteudo": msg.conteudo,
                "timestamp": msg.timestamp,
            })
            tokens_usados += msg.tokens_aprox

        # Adiciona contextos de código relevantes
        tokens_ctx = int(self.max_tokens * 0.2)
        for ctx in contextos[:3]:  # máx 3 arquivos no contexto
            ctx_tokens = max(1, len(ctx.conteudo) // 4)
            if tokens_usados + ctx_tokens > tokens_ctx:
                # Inclui só o resumo
                resultado["arquivos_relevantes"].append({
                    "arquivo": ctx.nome_arquivo,
                    "linguagem": ctx.linguagem,
                    "score": ctx.score_qualidade,
                    "problemas": len(ctx.problemas),
                    "resumo": ctx.conteudo[:200] + "..." if len(ctx.conteudo) > 200 else ctx.conteudo,
                })
            else:
                resultado["arquivos_relevantes"].append({
                    "arquivo": ctx.nome_arquivo,
                    "linguagem": ctx.linguagem,
                    "score": ctx.score_qualidade,
                    "conteudo": ctx.conteudo,
                })
                tokens_usados += ctx_tokens

        resultado["tokens_usados"] = tokens_usados
        return resultado


# -------------------------------------------------------
# MEMORY MANAGER — interface principal
# -------------------------------------------------------

class MemoryManager:
    """
    Interface única para todo o sistema de memória.

    Uso básico:
        mem = MemoryManager()

        # Salvar mensagem
        mem.salvar_mensagem("usuario", "oi, me chamo João")
        mem.salvar_mensagem("ia", "Oi João! Como posso ajudar?")

        # Salvar análise de código
        mem.salvar_contexto_codigo("meu_script.py", "python", codigo, score=8.5)

        # Ler histórico recente
        msgs = mem.historico_recente(n=10)

        # Preferências
        mem.prefs.nome_usuario = "João"
        print(mem.prefs.linguagem_favorita)

        # Montar contexto pra o modelo
        ctx = mem.contexto_para_modelo("como faço um loop em python?")
    """

    def __init__(self, caminho_db: str = "paradoxox_memory.db"):
        self.db         = BancoDados(caminho_db)
        self.sessao     = GerenciadorSessao()
        self.historico  = RepositorioHistorico(self.db)
        self.contexto   = RepositorioContexto(self.db)
        self.prefs      = RepositorioPreferencias(self.db)
        self.extractor  = ExtractorPreferencias()
        self.janela     = GerenciadorContextoJanela()

        # Inicia nova sessão
        self.sessao.nova_sessao()

        print(f"🧠 Memory iniciada — sessão: {self.sessao.id}")
        print(f"   Olá, {self.prefs.nome_usuario}!")

    # -------------------------------------------------------
    # SALVAR
    # -------------------------------------------------------

    def salvar_mensagem(
        self,
        papel: str,
        conteudo: str,
        metadados: dict = None
    ) -> int:
        """
        Salva uma mensagem no histórico.
        Se for do usuário, tenta extrair preferências automaticamente.
        """
        msg = Mensagem(
            papel=papel,
            conteudo=conteudo,
            sessao_id=self.sessao.id,
            metadados=metadados or {},
        )
        id_ = self.historico.salvar(msg)

        # Aprende preferências do que o usuário escreve
        if papel == "usuario":
            prefs_encontradas = self.extractor.extrair(conteudo)
            for chave, valor in prefs_encontradas.items():
                self.prefs.set(chave, valor)
                print(f"   💡 Preferência aprendida: {chave} = {valor}")

        return id_

    def salvar_contexto_codigo(
        self,
        nome_arquivo: str,
        linguagem: str,
        conteudo: str,
        score: float = 0.0,
        problemas: list = None,
        tipo: str = "arquivo",
    ) -> int:
        """Salva o contexto de um código analisado."""
        if not self.prefs.get("salvar_codigos", True):
            return -1

        ctx = ContextoCodigo(
            tipo=tipo,
            nome_arquivo=nome_arquivo,
            linguagem=linguagem,
            conteudo=conteudo,
            score_qualidade=score,
            problemas=problemas or [],
            sessao_id=self.sessao.id,
        )

        # Atualiza linguagem favorita se essa linguagem aparece muito
        langs = self.contexto.buscar_linguagens_usadas()
        if langs and langs[0]["linguagem"] != self.prefs.linguagem_favorita:
            nova_lang = langs[0]["linguagem"]
            self.prefs.linguagem_favorita = nova_lang
            print(f"   💡 Linguagem favorita atualizada: {nova_lang}")

        return self.contexto.salvar(ctx)

    # -------------------------------------------------------
    # LER
    # -------------------------------------------------------

    def historico_recente(self, n: int = None) -> list[Mensagem]:
        """Retorna as N mensagens mais recentes da sessão atual."""
        n = n or self.prefs.max_historico
        return self.historico.buscar_recente(self.sessao.id, n)

    def historico_completo_sessao(self) -> list[Mensagem]:
        """Retorna todo o histórico da sessão atual."""
        return self.historico.buscar_sessao(self.sessao.id, limite=9999)

    def contextos_sessao(self, tipo: str = None) -> list[ContextoCodigo]:
        """Retorna os contextos de código da sessão atual."""
        return self.contexto.buscar_sessao(self.sessao.id, tipo=tipo)

    def buscar_mensagem(self, termo: str) -> list[Mensagem]:
        """Busca mensagens que contêm um termo."""
        return self.historico.buscar_por_termo(termo, self.sessao.id)

    def historico_arquivo(self, nome_arquivo: str) -> list[ContextoCodigo]:
        """Retorna todas as análises de um arquivo específico."""
        return self.contexto.buscar_por_arquivo(nome_arquivo)

    # -------------------------------------------------------
    # CONTEXTO PARA O MODELO
    # -------------------------------------------------------

    def contexto_para_modelo(self, pergunta: str = "") -> dict:
        """
        Monta o contexto completo para alimentar o modelo.
        Chama isso antes de gerar uma resposta.
        """
        msgs = self.historico_recente()
        ctxs = self.contextos_sessao()
        prefs = self.prefs.get_todas()

        return self.janela.montar_contexto(msgs, ctxs, prefs, pergunta)

    # -------------------------------------------------------
    # SESSÕES
    # -------------------------------------------------------

    def nova_sessao(self) -> str:
        """Inicia uma nova conversa."""
        sid = self.sessao.nova_sessao()
        print(f"🆕 Nova sessão iniciada: {sid}")
        return sid

    def retomar_sessao(self, sessao_id: str):
        """Retoma uma sessão anterior."""
        self.sessao.definir_sessao(sessao_id)
        msgs = self.historico_recente(5)
        print(f"📂 Sessão retomada: {sessao_id}")
        if msgs:
            print(f"   Última mensagem: {msgs[-1].resumo()}")

    def listar_sessoes(self) -> list[dict]:
        """Lista sessões anteriores."""
        return self.historico.listar_sessoes()

    # -------------------------------------------------------
    # ESTATÍSTICAS
    # -------------------------------------------------------

    def stats(self) -> dict:
        """Retorna estatísticas gerais da memória."""
        row = self.db.executar(
            "SELECT COUNT(*) as total FROM historico WHERE sessao_id = ?",
            (self.sessao.id,)
        ).fetchone()
        total_msgs = row["total"] if row else 0

        row2 = self.db.executar(
            "SELECT COUNT(*) as total FROM contexto_codigo WHERE sessao_id = ?",
            (self.sessao.id,)
        ).fetchone()
        total_ctx = row2["total"] if row2 else 0

        tokens = self.historico.total_tokens_sessao(self.sessao.id)
        langs = self.contexto.buscar_linguagens_usadas()

        return {
            "sessao_id": self.sessao.id,
            "mensagens_sessao": total_msgs,
            "contextos_sessao": total_ctx,
            "tokens_estimados": tokens,
            "linguagens_usadas": langs,
            "nome_usuario": self.prefs.nome_usuario,
            "linguagem_favorita": self.prefs.linguagem_favorita,
        }

    def exibir_stats(self):
        """Imprime estatísticas de forma legível."""
        s = self.stats()
        print(f"\n📊 MEMORY STATS — ParadoxoX")
        print(f"   Sessão           : {s['sessao_id']}")
        print(f"   Usuário          : {s['nome_usuario']}")
        print(f"   Mensagens        : {s['mensagens_sessao']}")
        print(f"   Contextos código : {s['contextos_sessao']}")
        print(f"   Tokens est.      : {s['tokens_estimados']:,}")
        print(f"   Lang. favorita   : {s['linguagem_favorita']}")
        if s["linguagens_usadas"]:
            langs = ", ".join(f"{l['linguagem']}({l['total']})" for l in s["linguagens_usadas"][:5])
            print(f"   Linguagens usadas: {langs}")

    # -------------------------------------------------------
    # LIMPEZA
    # -------------------------------------------------------

    def limpar_sessao(self, sessao_id: str = None):
        """Apaga o histórico de uma sessão específica."""
        sid = sessao_id or self.sessao.id
        self.db.executar("DELETE FROM historico WHERE sessao_id = ?", (sid,))
        self.db.executar("DELETE FROM contexto_codigo WHERE sessao_id = ?", (sid,))
        self.db.executar("DELETE FROM sessoes WHERE sessao_id = ?", (sid,))
        self.db.commit()
        print(f"🗑️  Sessão {sid} apagada.")

    def fechar(self):
        """Fecha a conexão com o banco."""
        self.db.fechar()
        print("💤 Memory fechada.")


# -------------------------------------------------------
# TESTE RÁPIDO
# -------------------------------------------------------
if __name__ == "__main__":
    import os

    # Limpa banco de teste se existir
    if os.path.exists("paradoxox_memory_test.db"):
        os.remove("paradoxox_memory_test.db")

    print("⚛️  PARADOXO X — Testando Memory System\n")

    mem = MemoryManager("paradoxox_memory_test.db")

    # --- Teste 1: Salvar histórico ---
    print("\n--- Teste 1: Histórico de conversa ---")
    mem.salvar_mensagem("usuario", "oi, me chamo João e prefiro python")
    mem.salvar_mensagem("ia", "Oi João! Entendido, vou usar Python por padrão.")
    mem.salvar_mensagem("usuario", "analisa esse código pra mim")
    mem.salvar_mensagem("ia", "Claro! Me manda o código.")
    mem.salvar_mensagem("usuario", "def soma(a, b): return a + b")
    mem.salvar_mensagem("ia", "Código simples e limpo! Score 9.5/10.")

    msgs = mem.historico_recente(n=4)
    print(f"Últimas 4 mensagens:")
    for m in msgs:
        print(f"  {m.resumo()}")

    # --- Teste 2: Contexto de código ---
    print("\n--- Teste 2: Contexto de código ---")
    mem.salvar_contexto_codigo(
        nome_arquivo="meu_script.py",
        linguagem="python",
        conteudo="def soma(a, b):\n    return a + b",
        score=9.5,
        problemas=[]
    )
    mem.salvar_contexto_codigo(
        nome_arquivo="app.js",
        linguagem="javascript",
        conteudo="var x = 1\nconsole.log(x)",
        score=6.0,
        problemas=[{"tipo": "aviso", "mensagem": "Use const/let"}]
    )

    ctxs = mem.contextos_sessao()
    print(f"Contextos salvos: {len(ctxs)}")
    for c in ctxs:
        print(f"  📄 {c.nome_arquivo} ({c.linguagem}) — score {c.score_qualidade}")

    # --- Teste 3: Preferências ---
    print("\n--- Teste 3: Preferências ---")
    print(f"Nome    : {mem.prefs.nome_usuario}")           # Aprendido da conversa
    print(f"Linguagem: {mem.prefs.linguagem_favorita}")    # Aprendido da conversa

    mem.prefs.set("tema", "claro")
    mem.prefs.set("auto_corrigir", True)
    print(f"Tema    : {mem.prefs.get('tema')}")
    print(f"Auto-corrigir: {mem.prefs.get('auto_corrigir')}")

    # --- Teste 4: Contexto para o modelo ---
    print("\n--- Teste 4: Contexto para o modelo ---")
    ctx_modelo = mem.contexto_para_modelo("como melhoro esse código?")
    print(f"Mensagens no contexto : {len(ctx_modelo['historico_recente'])}")
    print(f"Arquivos no contexto  : {len(ctx_modelo['arquivos_relevantes'])}")
    print(f"Tokens estimados      : {ctx_modelo['tokens_usados']}")

    # --- Teste 5: Busca ---
    print("\n--- Teste 5: Busca ---")
    encontradas = mem.buscar_mensagem("código")
    print(f"Mensagens com 'código': {len(encontradas)}")
    for m in encontradas:
        print(f"  {m.resumo()}")

    # --- Stats finais ---
    mem.exibir_stats()

    # --- Teste 6: Nova sessão e retomada ---
    print("\n--- Teste 6: Sessões ---")
    sessao_antiga = mem.sessao.id
    mem.nova_sessao()
    mem.salvar_mensagem("usuario", "nova sessão aqui")

    sessoes = mem.listar_sessoes()
    print(f"Total de sessões: {len(sessoes)}")
    for s in sessoes:
        print(f"  📅 {s['sessao_id']} — {s['total_mensagens']} msgs")

    mem.retomar_sessao(sessao_antiga)

    # Limpeza
    mem.fechar()
    os.remove("paradoxox_memory_test.db")
    print("\n✅ Memory System funcionando! Próximo: integrar com o core.")