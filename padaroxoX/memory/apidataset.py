"""
PARADOXO X — API do Dataset v2
================================
Servidor HTTP que recebe dados de qualquer fonte externa
e os armazena diretamente no banco paradoxox_dataset.db.
 
MELHORIAS v2:
  ✅ Fila de escrita serializada (queue.Queue) — zero "database is locked"
  ✅ Conexão por requisição para leituras — não compartilha estado
  ✅ Worker thread dedicado para todos os INSERTs
  ✅ Timeout de fila configurável — nunca trava o servidor
  ✅ Logging estruturado com timestamp e método HTTP
  ✅ Graceful shutdown — fecha conexões ao Ctrl+C
  ✅ Endpoint GET /buscar?q=termo — busca textos no banco
  ✅ Endpoint POST /marcar_treinado — marca exemplos como usados
  ✅ Validações mais detalhadas com mensagens de erro claras
 
Como iniciar:
  python memory/apidataset.py
 
Endpoints:
  POST   /add               → adiciona 1 texto
  POST   /add_batch         → adiciona até 1000 textos de uma vez
  GET    /status            → estatísticas do banco
  GET    /recentes          → últimos N exemplos (padrão 20)
  GET    /buscar?q=termo    → busca textos por termo
  GET    /health            → verifica se está rodando
  DELETE /remover           → remove por ?id=N, ?fonte=X ou ?categoria=X
  POST   /marcar_treinado   → marca exemplos como usados no treino
 
Exemplos curl:
  curl -X POST http://localhost:7799/add \\
       -H "Content-Type: application/json" \\
       -d '{"texto": "o gato comeu o rato"}'
 
  curl -X POST http://localhost:7799/add_batch \\
       -H "Content-Type: application/json" \\
       -d '{"textos": ["frase 1", "frase 2"], "categoria": "geral"}'
 
  curl "http://localhost:7799/recentes?limite=10"
  curl "http://localhost:7799/buscar?q=python"
  curl  http://localhost:7799/status
"""
 
import argparse
import json
import queue
import sqlite3
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
 
# ── Caminhos ──────────────────────────────────────────────────────────
_DIR  = Path(__file__).resolve().parent
_ROOT = _DIR.parent
sys.path.insert(0, str(_ROOT))
 
# ── Configurações ─────────────────────────────────────────────────────
HOST      = "localhost"
PORT      = 7799
DB_PATH   = str(_DIR / "paradoxox_dataset.db")
API_KEY   = ""        # vazio = sem autenticação (só localhost)
MIN_CHARS = 5
MAX_CHARS = 50_000
FILA_TIMEOUT = 15.0   # segundos máximos esperando a fila de escrita
 
 
# ═══════════════════════════════════════════════════════════════════════
# BANCO DE DADOS — funções puras (sem estado global)
# ═══════════════════════════════════════════════════════════════════════
 
def _nova_conexao() -> sqlite3.Connection:
    """
    Abre uma conexão nova, configura WAL e garante que as tabelas existem.
    Cada chamada retorna uma conexão independente — sem compartilhamento.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-8000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exemplos (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            texto        TEXT    NOT NULL,
            fonte        TEXT    DEFAULT '',
            categoria    TEXT    DEFAULT 'geral',
            adicionado   REAL    NOT NULL DEFAULT (unixepoch('now','subsec')),
            usado_treino INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fonte ON exemplos(fonte)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cat   ON exemplos(categoria)")
    conn.commit()
    return conn
 
 
def _inserir(
    conn:      sqlite3.Connection,
    textos:    list[str],
    categoria: str = "geral",
    fonte:     str = "api",
) -> tuple[int, int]:
    """
    Insere textos no banco ignorando duplicatas exatas.
    Retorna (adicionados, duplicatas).
    """
    adicionados = 0
    duplicatas  = 0
 
    for texto in textos:
        texto = texto.strip()
        if not texto or len(texto) < MIN_CHARS:
            continue
        if len(texto) > MAX_CHARS:
            texto = texto[:MAX_CHARS]
 
        existe = conn.execute(
            "SELECT 1 FROM exemplos WHERE texto = ? LIMIT 1", (texto,)
        ).fetchone()
 
        if existe:
            duplicatas += 1
        else:
            conn.execute(
                "INSERT INTO exemplos (texto, fonte, categoria) VALUES (?,?,?)",
                (texto, fonte, categoria),
            )
            adicionados += 1
 
    conn.commit()
    return adicionados, duplicatas
 
 
def _status(conn: sqlite3.Connection) -> dict:
    total     = conn.execute("SELECT COUNT(*) FROM exemplos").fetchone()[0]
    treinados = conn.execute(
        "SELECT COUNT(*) FROM exemplos WHERE usado_treino=1"
    ).fetchone()[0]
    cats = conn.execute(
        "SELECT categoria, COUNT(*) AS n FROM exemplos "
        "GROUP BY categoria ORDER BY n DESC"
    ).fetchall()
    return {
        "total"     : total,
        "treinados" : treinados,
        "pendentes" : total - treinados,
        "categorias": {r["categoria"]: r["n"] for r in cats},
        "db_path"   : DB_PATH,
    }
 
 
# ═══════════════════════════════════════════════════════════════════════
# FILA DE ESCRITA — serializa todos os INSERTs numa thread única
# ═══════════════════════════════════════════════════════════════════════
#
# Por que fila e não lock?
# ─────────────────────────
# SQLite no modo WAL suporta múltiplos leitores simultâneos, mas apenas
# UM escritor por vez. Um threading.Lock funcionaria, mas se duas
# requisições chegarem ao mesmo tempo, uma delas bloquearia a thread
# HTTP inteira enquanto espera o lock — degradando a latência de leituras.
#
# Com uma fila + worker thread dedicado:
#   - A thread HTTP nunca fica bloqueada escrevendo
#   - A thread HTTP apenas enfileira o trabalho e aguarda o Event
#   - O worker processa um INSERT por vez, sem concorrência de escrita
#   - Leituras (GET) abrem sua própria conexão e nunca passam pela fila
 
_fila_escrita: queue.Queue = queue.Queue()
_worker_thread: threading.Thread = None
_worker_conn:   sqlite3.Connection = None
 
 
def _worker_escrita():
    """
    Thread única que processa todos os INSERTs em sequência.
    Roda como daemon — encerra automaticamente quando o processo principal sai.
    """
    global _worker_conn
    _worker_conn = _nova_conexao()
 
    while True:
        try:
            item = _fila_escrita.get(timeout=1.0)
        except queue.Empty:
            continue
 
        if item is None:
            # Sinal de encerramento graceful
            break
 
        textos, categoria, fonte, resultado = item
        try:
            add, dup = _inserir(_worker_conn, textos, categoria, fonte)
            resultado["adicionados"] = add
            resultado["duplicados"]  = dup
            resultado["ok"]          = True
        except Exception as e:
            resultado["ok"]   = False
            resultado["erro"] = str(e)
        finally:
            resultado["pronto"].set()
 
    if _worker_conn:
        _worker_conn.close()
 
 
def _iniciar_worker():
    """Inicia o worker de escrita se ainda não estiver rodando."""
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(
            target=_worker_escrita,
            name="paradoxox-db-writer",
            daemon=True,
        )
        _worker_thread.start()
 
 
def _enfileirar_escrita(
    textos:    list[str],
    categoria: str,
    fonte:     str,
) -> tuple[int, int]:
    """
    Enfileira textos para inserção e aguarda o resultado.
    Thread-safe — pode ser chamada de qualquer handler simultaneamente.
 
    Levanta RuntimeError se a fila não responder dentro de FILA_TIMEOUT.
    """
    resultado = {
        "pronto"     : threading.Event(),
        "ok"         : False,
        "adicionados": 0,
        "duplicados" : 0,
    }
    _fila_escrita.put((textos, categoria, fonte, resultado))
 
    if not resultado["pronto"].wait(timeout=FILA_TIMEOUT):
        raise RuntimeError(
            f"Timeout na fila de escrita após {FILA_TIMEOUT}s. "
            "O banco pode estar travado."
        )
    if not resultado["ok"]:
        raise RuntimeError(resultado.get("erro", "Erro desconhecido na escrita"))
 
    return resultado["adicionados"], resultado["duplicados"]
 
 
# ═══════════════════════════════════════════════════════════════════════
# HANDLER HTTP
# ═══════════════════════════════════════════════════════════════════════
 
class DatasetHandler(BaseHTTPRequestHandler):
    """
    Handler HTTP do ParadoxoX Dataset API.
 
    Regra fundamental:
      Escritas  → sempre via _enfileirar_escrita() (thread-safe)
      Leituras  → sempre via _nova_conexao() por requisição (sem estado)
    """
 
    # ── Logging limpo ─────────────────────────────────────────────────
 
    def log_message(self, fmt, *args):
        agora = datetime.now().strftime("%H:%M:%S")
        status = args[1] if len(args) > 1 else "?"
        print(f"  [{agora}] {self.command:<7} {self.path:<32} → {status}")
 
    # ── Autenticação ──────────────────────────────────────────────────
 
    def _autenticado(self) -> bool:
        if not API_KEY:
            return True
        return self.headers.get("X-API-Key", "") == API_KEY
 
    # ── Helpers ───────────────────────────────────────────────────────
 
    def _responder(self, status: int, dados: dict):
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)
 
    def _ler_body(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n == 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, ValueError):
            return None
 
    def _params(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
 
    def _rota(self) -> str:
        return urlparse(self.path).path
 
    def _conn_leitura(self) -> sqlite3.Connection:
        """Abre uma conexão nova só para esta requisição de leitura."""
        return _nova_conexao()
 
    # ── GET ───────────────────────────────────────────────────────────
 
    def do_GET(self):
        if not self._autenticado():
            self._responder(401, {"erro": "API key inválida"})
            return
 
        rota = self._rota()
 
        # GET /health
        if rota == "/health":
            self._responder(200, {
                "status" : "ok",
                "hora"   : datetime.now().isoformat(),
                "db"     : DB_PATH,
                "fila"   : _fila_escrita.qsize(),
            })
 
        # GET /status
        elif rota == "/status":
            conn = self._conn_leitura()
            try:
                self._responder(200, _status(conn))
            finally:
                conn.close()
 
        # GET /recentes?limite=20
        elif rota == "/recentes":
            limite = min(int(self._params().get("limite", 20)), 500)
            conn   = self._conn_leitura()
            try:
                rows = conn.execute(
                    "SELECT id, texto, categoria, fonte, adicionado "
                    "FROM exemplos ORDER BY id DESC LIMIT ?",
                    (limite,)
                ).fetchall()
                dados = [
                    {
                        "id"        : r["id"],
                        "texto"     : r["texto"][:150] + ("…" if len(r["texto"]) > 150 else ""),
                        "categoria" : r["categoria"],
                        "fonte"     : r["fonte"],
                        "adicionado": datetime.fromtimestamp(r["adicionado"]).isoformat(),
                    }
                    for r in rows
                ]
                self._responder(200, {"total": len(dados), "exemplos": dados})
            finally:
                conn.close()
 
        # GET /buscar?q=termo&limite=20
        elif rota == "/buscar":
            params = self._params()
            termo  = params.get("q", "").strip()
            if not termo:
                self._responder(400, {"erro": "Parâmetro ?q=termo é obrigatório"})
                return
            limite = min(int(params.get("limite", 20)), 200)
            conn   = self._conn_leitura()
            try:
                rows = conn.execute(
                    "SELECT id, texto, categoria, fonte FROM exemplos "
                    "WHERE texto LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{termo}%", limite)
                ).fetchall()
                self._responder(200, {
                    "termo"    : termo,
                    "total"    : len(rows),
                    "exemplos" : [dict(r) for r in rows],
                })
            finally:
                conn.close()
 
        else:
            self._responder(404, {"erro": f"Rota GET '{rota}' não existe"})
 
    # ── POST ──────────────────────────────────────────────────────────
 
    def do_POST(self):
        if not self._autenticado():
            self._responder(401, {"erro": "API key inválida"})
            return
 
        rota = self._rota()
        body = self._ler_body()
 
        if body is None:
            self._responder(400, {"erro": "JSON inválido no corpo da requisição"})
            return
 
        # POST /add — adiciona 1 texto
        if rota == "/add":
            texto = str(body.get("texto", "")).strip()
 
            if not texto:
                self._responder(400, {"erro": "Campo 'texto' é obrigatório"})
                return
            if len(texto) < MIN_CHARS:
                self._responder(400, {
                    "erro": f"Texto muito curto (mínimo {MIN_CHARS} caracteres, recebeu {len(texto)})"
                })
                return
 
            categoria = str(body.get("categoria", "geral")).strip() or "geral"
            fonte     = str(body.get("fonte",     "api"  )).strip() or "api"
 
            try:
                add, dup = _enfileirar_escrita([texto], categoria, fonte)
            except RuntimeError as e:
                self._responder(503, {"erro": str(e)})
                return
 
            if add > 0:
                self._responder(201, {
                    "status"   : "adicionado",
                    "categoria": categoria,
                    "fonte"    : fonte,
                    "chars"    : len(texto),
                })
            else:
                self._responder(200, {
                    "status": "duplicado",
                    "info"  : "Esse texto já existe no banco",
                })
 
        # POST /add_batch — adiciona vários textos
        elif rota == "/add_batch":
            textos = body.get("textos", [])
 
            if not isinstance(textos, list) or not textos:
                self._responder(400, {
                    "erro": "Campo 'textos' deve ser uma lista não-vazia"
                })
                return
            if len(textos) > 1000:
                self._responder(400, {
                    "erro": f"Máximo 1000 textos por chamada (recebeu {len(textos)}). "
                            "Divida em chunks menores."
                })
                return
 
            categoria = str(body.get("categoria", "geral")).strip() or "geral"
            fonte     = str(body.get("fonte", "api_batch")).strip() or "api_batch"
 
            try:
                add, dup = _enfileirar_escrita(
                    [str(t) for t in textos], categoria, fonte
                )
            except RuntimeError as e:
                self._responder(503, {"erro": str(e)})
                return
 
            self._responder(201, {
                "status"     : "ok",
                "adicionados": add,
                "duplicados" : dup,
                "recusados"  : len(textos) - add - dup,
                "categoria"  : categoria,
            })
 
        # POST /marcar_treinado — marca exemplos como usados no treino
        # Body: {"ids": [1, 2, 3]}  ou  {"categoria": "wikipedia"}
        elif rota == "/marcar_treinado":
            ids_lista  = body.get("ids", [])
            categoria  = body.get("categoria", "")
 
            if ids_lista:
                resultado = {"marcados": threading.Event(), "n": 0}
 
                def _marcar_ids():
                    conn = _nova_conexao()
                    try:
                        placeholders = ",".join("?" * len(ids_lista))
                        cur = conn.execute(
                            f"UPDATE exemplos SET usado_treino=1 WHERE id IN ({placeholders})",
                            ids_lista,
                        )
                        conn.commit()
                        resultado["n"] = cur.rowcount
                    finally:
                        conn.close()
                        resultado["marcados"].set()
 
                t = threading.Thread(target=_marcar_ids, daemon=True)
                t.start()
                resultado["marcados"].wait(timeout=10.0)
                self._responder(200, {"marcados": resultado["n"]})
 
            elif categoria:
                resultado = {"marcados": threading.Event(), "n": 0}
 
                def _marcar_cat():
                    conn = _nova_conexao()
                    try:
                        cur = conn.execute(
                            "UPDATE exemplos SET usado_treino=1 WHERE categoria=?",
                            (categoria,)
                        )
                        conn.commit()
                        resultado["n"] = cur.rowcount
                    finally:
                        conn.close()
                        resultado["marcados"].set()
 
                t = threading.Thread(target=_marcar_cat, daemon=True)
                t.start()
                resultado["marcados"].wait(timeout=10.0)
                self._responder(200, {"marcados": resultado["n"]})
 
            else:
                self._responder(400, {
                    "erro": "Informe 'ids' (lista) ou 'categoria' (string)"
                })
 
        else:
            self._responder(404, {"erro": f"Rota POST '{rota}' não existe"})
 
    # ── DELETE ────────────────────────────────────────────────────────
 
    def do_DELETE(self):
        if not self._autenticado():
            self._responder(401, {"erro": "API key inválida"})
            return
 
        rota   = self._rota()
        params = self._params()
 
        if rota == "/remover":
            resultado = {"ev": threading.Event(), "n": 0, "ok": True, "erro": ""}
 
            def _remover():
                conn = _nova_conexao()
                try:
                    if "id" in params:
                        cur = conn.execute(
                            "DELETE FROM exemplos WHERE id = ?",
                            (int(params["id"]),)
                        )
                    elif "fonte" in params:
                        cur = conn.execute(
                            "DELETE FROM exemplos WHERE fonte = ?",
                            (params["fonte"],)
                        )
                    elif "categoria" in params:
                        cur = conn.execute(
                            "DELETE FROM exemplos WHERE categoria = ?",
                            (params["categoria"],)
                        )
                    else:
                        resultado["ok"]   = False
                        resultado["erro"] = "Informe ?id=N, ?fonte=X ou ?categoria=X"
                        return
                    conn.commit()
                    resultado["n"] = cur.rowcount
                except Exception as e:
                    resultado["ok"]   = False
                    resultado["erro"] = str(e)
                finally:
                    conn.close()
                    resultado["ev"].set()
 
            t = threading.Thread(target=_remover, daemon=True)
            t.start()
            resultado["ev"].wait(timeout=10.0)
 
            if not resultado["ok"]:
                self._responder(400, {"erro": resultado["erro"]})
            else:
                self._responder(200, {"removidos": resultado["n"]})
 
        else:
            self._responder(404, {"erro": f"Rota DELETE '{rota}' não existe"})
 
 
# ═══════════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO DO SERVIDOR
# ═══════════════════════════════════════════════════════════════════════
 
def iniciar(host: str = HOST, porta: int = PORT):
    """Inicia o servidor e o worker de escrita. Roda até Ctrl+C."""
 
    # Garante banco e inicia worker de escrita antes de aceitar requisições
    conn_init = _nova_conexao()
    stats     = _status(conn_init)
    conn_init.close()
 
    _iniciar_worker()
 
    print(f"\n⚛️  ParadoxoX — API do Dataset v2")
    print(f"{'─' * 46}")
    print(f"  Endereço  : http://{host}:{porta}")
    print(f"  Banco     : {DB_PATH}")
    print(f"  Exemplos  : {stats['total']:,}  ({stats['pendentes']:,} pendentes)")
    print(f"  Auth      : {'✅ API Key ativa' if API_KEY else '⚠️  Sem auth (localhost only)'}")
    print(f"  Escrita   : fila serializada (zero database locked)")
    print(f"{'─' * 46}")
    print(f"  Endpoints:")
    print(f"    POST   /add               → 1 texto")
    print(f"    POST   /add_batch         → até 1000 textos")
    print(f"    POST   /marcar_treinado   → marca como usado")
    print(f"    GET    /status            → estatísticas")
    print(f"    GET    /recentes          → últimos exemplos")
    print(f"    GET    /buscar?q=termo    → busca no banco")
    print(f"    GET    /health            → status da API")
    print(f"    DELETE /remover           → por id/fonte/categoria")
    print(f"\n  Ctrl+C para encerrar\n")
    print(f"{'─' * 46}")
 
    servidor = HTTPServer((host, porta), DatasetHandler)
 
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\n⛔  Encerrando servidor...")
        # Sinaliza o worker para encerrar graciosamente
        _fila_escrita.put(None)
        if _worker_thread:
            _worker_thread.join(timeout=3.0)
        servidor.server_close()
        print(f"⛔  Servidor encerrado.")
 
 
# ═══════════════════════════════════════════════════════════════════════
# CLIENTE PYTHON
# ═══════════════════════════════════════════════════════════════════════
 
class DatasetAPIClient:
    """
    Cliente Python para enviar dados à API sem usar curl.
 
    Uso:
        from memory.apidataset import DatasetAPIClient
 
        cliente = DatasetAPIClient()
        cliente.add("o gato comeu o rato")
        cliente.add("def soma(a,b): return a+b", categoria="codigo")
        cliente.add_batch(["frase 1", "frase 2", "frase 3"])
        print(cliente.status())
        print(cliente.buscar("python"))
    """
 
    def __init__(self, host: str = HOST, porta: int = PORT, api_key: str = ""):
        self.base = f"http://{host}:{porta}"
        self._key = api_key
 
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._key:
            h["X-API-Key"] = self._key
        return h
 
    def _post(self, rota: str, dados: dict) -> dict:
        from urllib.request import Request, urlopen
        from urllib.error import URLError
        body = json.dumps(dados).encode("utf-8")
        req  = Request(f"{self.base}{rota}", data=body,
                       headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except URLError:
            print(f"⚠️  API offline em {self.base}")
            print(f"   Inicie: python memory/apidataset.py")
            return {"erro": "conexão recusada"}
 
    def _get(self, rota: str) -> dict:
        from urllib.request import Request, urlopen
        from urllib.error import URLError
        req = Request(f"{self.base}{rota}", headers=self._headers())
        try:
            with urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except URLError:
            return {"erro": "conexão recusada"}
 
    def add(self, texto: str, categoria: str = "geral", fonte: str = "script") -> dict:
        return self._post("/add", {"texto": texto, "categoria": categoria, "fonte": fonte})
 
    def add_batch(self, textos: list[str], categoria: str = "geral",
                  fonte: str = "script_batch") -> dict:
        resultado = {"adicionados": 0, "duplicados": 0}
        for i in range(0, len(textos), 1000):
            r = self._post("/add_batch", {
                "textos": textos[i:i+1000],
                "categoria": categoria,
                "fonte": fonte,
            })
            resultado["adicionados"] += r.get("adicionados", 0)
            resultado["duplicados"]  += r.get("duplicados", 0)
        return resultado
 
    def status(self) -> dict:
        return self._get("/status")
 
    def recentes(self, limite: int = 20) -> list:
        return self._get(f"/recentes?limite={limite}").get("exemplos", [])
 
    def buscar(self, termo: str, limite: int = 20) -> list:
        from urllib.parse import quote
        return self._get(f"/buscar?q={quote(termo)}&limite={limite}").get("exemplos", [])
 
    def health(self) -> bool:
        return self._get("/health").get("status") == "ok"
 
    def marcar_treinado(self, ids: list[int] = None, categoria: str = "") -> int:
        body = {}
        if ids:
            body["ids"] = ids
        elif categoria:
            body["categoria"] = categoria
        return self._post("/marcar_treinado", body).get("marcados", 0)
 
 
# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="⚛️  ParadoxoX — API do Dataset v2")
    parser.add_argument("--host",  default=HOST,  help=f"Host (padrão: {HOST})")
    parser.add_argument("--porta", default=PORT,  type=int, help=f"Porta (padrão: {PORT})")
    parser.add_argument("--key",   default="",    help="API Key (opcional)")
    args = parser.parse_args()
 
    if args.key:
        API_KEY = args.key
 
    iniciar(args.host, args.porta)