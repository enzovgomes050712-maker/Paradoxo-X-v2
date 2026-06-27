"""
PARADOXO X — Alimentador Wikipedia + Groq v2
=============================================
Puxa artigos da Wikipedia, processa com o Groq e grava no dataset.
 
MELHORIAS v2:
  ✅ Verifica se o tópico já foi processado ANTES de chamar a Wikipedia
  ✅ Rate limiting inteligente — pausa adaptativa para não ser bloqueado
  ✅ Retry automático com backoff exponencial (Wikipedia e Groq)
  ✅ Groq via urllib corrigido — sem 403
  ✅ Deduplicação por fonte (não repete tópicos entre sessões)
  ✅ Progresso salvo em arquivo local — retoma de onde parou
  ✅ User-Agent realista para a Wikipedia
  ✅ Sem duplicatas na lista de tópicos (remove duplicatas automaticamente)
 
Como usar:
  python memory/alimentador.py --direto
  python memory/alimentador.py --direto --sem-groq   (sem precisar do Groq)
  python memory/alimentador.py --direto --topicos 50
"""
 
import json
import sqlite3
import time
import re
import random
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import URLError, HTTPError
 
# ═══════════════════════════════════════════════════════
# ⚠️  COLOQUE SUA CHAVE DO GROQ AQUI
# ═══════════════════════════════════════════════════════
GROQ_API_KEY = ""   # ← cole aqui: "gsk_xxxxxxxxxxxxxxxxxxxx"
 
# ═══════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════
_DIR     = Path(__file__).resolve().parent
DB_PATH  = str(_DIR / "paradoxox_dataset.db")
 
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"   # modelo ativo e gratuito
 
TEXTOS_POR_ARTIGO = 5    # textos gerados pelo Groq por artigo
PAUSA_WIKIPEDIA   = 2.0  # segundos entre chamadas à Wikipedia
PAUSA_GROQ        = 2.0  # segundos entre chamadas ao Groq
PAUSA_JITTER      = 0.5  # variação aleatória extra (evita padrão fixo)
 
# ── User-Agent realista — Wikipedia bloqueia bots sem isso ──────────
USER_AGENT = (
    "Mozilla/5.0 (compatible; ParadoxoX-Dataset/2.0; "
    "educational research; +https://github.com/paradoxox)"
)
 
# ═══════════════════════════════════════════════════════
# LISTA DE TÓPICOS
# ═══════════════════════════════════════════════════════
TOPICOS_WIKIPEDIA = [
    # ── IA / ML ──
    "Inteligência artificial", "Aprendizado de máquina", "Rede neural artificial",
    "Processamento de linguagem natural", "Deep learning", "Transformer (aprendizado de máquina)",
    "Large language model", "TensorFlow", "PyTorch", "Fine-tuning",
    "GPU", "CUDA", "Embeddings", "Tokenização", "OpenAI", "Prompt engineering",
    # ── Programação ──
    "Python (linguagem de programação)", "JavaScript", "TypeScript", "Java",
    "C++", "Rust", "Go (linguagem de programação)", "Haskell", "Lua", "Ruby",
    "PHP", "Swift", "Kotlin", "Scala", "Perl", "Dart", "Fortran", "COBOL",
    "Elixir", "Julia (linguagem de programação)", "Lisp", "Prolog",
    "Programação orientada a objetos", "Programação funcional",
    "Programação concorrente", "Algoritmos", "Estrutura de dados",
    "Compiladores", "Interpretadores", "Garbage collector",
    # ── Web ──
    "HTML", "CSS", "React", "Vue.js", "Angular", "Node.js",
    "FastAPI", "Flask", "Django", "REST API", "GraphQL",
    "WebSocket", "HTTP", "JWT", "OAuth", "Nginx",
    # ── Banco de dados ──
    "Banco de dados", "SQL", "SQLite", "MySQL", "PostgreSQL",
    "MongoDB", "Redis", "NoSQL", "Firebase",
    # ── Sistemas / Infra ──
    "Linux", "Windows", "macOS", "Ubuntu", "Debian", "Arch Linux", "Kali Linux",
    "Kernel", "Shell script", "Bash", "Docker", "Kubernetes",
    "Cloud computing", "Virtualização", "Git", "GitHub", "CI/CD",
    "Microserviços", "DevOps", "Sistema operacional", "TCP/IP",
    "DNS", "VPN", "Redes de computadores",
    # ── Segurança ──
    "Cibersegurança", "Criptografia", "Firewall", "Malware", "Ransomware",
    "Phishing", "SQL injection", "Buffer overflow", "Ethical hacking",
    "Engenharia social", "Zero-day",
    # ── Hardware ──
    "CPU", "Processador", "Memória RAM", "SSD", "Placa de vídeo",
    "Arduino", "Raspberry Pi", "Overclock",
    # ── Ciências ──
    "Física quântica", "Teoria da relatividade", "Big Bang", "Buraco negro",
    "DNA", "Genética", "Evolução biológica", "Neurociência", "Biologia celular",
    "Tabela periódica", "Energia nuclear", "Computação quântica",
    # ── Matemática ──
    "Matemática", "Álgebra", "Geometria", "Cálculo diferencial",
    "Probabilidade", "Estatística", "Teoria dos grafos", "Lógica matemática",
    # ── História / Mundo ──
    "História do Brasil", "Segunda Guerra Mundial", "Guerra Fria",
    "Revolução Industrial", "Império Romano", "Grécia Antiga",
    "Egito Antigo", "Napoleão Bonaparte", "Vikings", "Samurais",
    "Estados Unidos", "Japão", "China", "Alemanha", "França",
    # ── Cultura ──
    "Anime", "Mangá", "Dragon Ball", "Naruto", "One Piece",
    "Attack on Titan", "Death Note", "Pokémon", "Minecraft",
    "League of Legends", "Counter-Strike", "Game of Thrones",
    "Breaking Bad", "Marvel Comics", "DC Comics", "Cinema",
    "Rock", "Hip hop", "K-pop", "Michael Jackson", "The Beatles",
    # ── Outros ──
    "Bitcoin", "Blockchain", "Empreendedorismo", "Marketing digital",
    "Amazônia", "Mudanças climáticas", "Energia solar",
    "Medicina", "Vacinas", "Psicologia", "Filosofia",
    "Futebol", "Astronomia", "Sistema solar",
]
 
# Remove duplicatas mantendo a ordem original
_visto: set = set()
_unicos: list = []
for _t in TOPICOS_WIKIPEDIA:
    _chave = _t.lower().strip()
    if _chave not in _visto:
        _visto.add(_chave)
        _unicos.append(_t)
TOPICOS_WIKIPEDIA = _unicos
 
 
# ═══════════════════════════════════════════════════════
# BANCO DE DADOS
# ═══════════════════════════════════════════════════════
 
def _conectar() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
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
    conn.commit()
    return conn
 
 
def topico_ja_existe(conn: sqlite3.Connection, topico: str) -> bool:
    """
    Verifica se esse tópico já foi processado em sessões anteriores.
    Usa o campo `fonte` que gravamos como 'wikipedia:NomeDoTopico'.
    Assim nunca buscamos o mesmo artigo duas vezes.
    """
    fonte = f"wikipedia:{topico}"
    row = conn.execute(
        "SELECT 1 FROM exemplos WHERE fonte = ? LIMIT 1", (fonte,)
    ).fetchone()
    return row is not None
 
 
def gravar(
    conn: sqlite3.Connection,
    textos: list[str],
    categoria: str,
    topico: str,
) -> tuple[int, int]:
    """Grava textos no banco, ignorando duplicatas exatas."""
    adicionados = 0
    duplicados  = 0
    fonte = f"wikipedia:{topico}"
 
    for texto in textos:
        texto = texto.strip()
        if not texto or len(texto) < 10:
            continue
        existe = conn.execute(
            "SELECT 1 FROM exemplos WHERE texto = ? LIMIT 1", (texto,)
        ).fetchone()
        if existe:
            duplicados += 1
        else:
            conn.execute(
                "INSERT INTO exemplos (texto, fonte, categoria) VALUES (?,?,?)",
                (texto, fonte, categoria),
            )
            adicionados += 1
 
    conn.commit()
    return adicionados, duplicados
 
 
# ═══════════════════════════════════════════════════════
# WIKIPEDIA
# ═══════════════════════════════════════════════════════
 
def _pausa_wikipedia():
    """Pausa com jitter aleatório para não parecer um bot."""
    t = PAUSA_WIKIPEDIA + random.uniform(0, PAUSA_JITTER)
    time.sleep(t)
 
 
def buscar_wikipedia(topico: str, tentativas: int = 3) -> str | None:
    """
    Busca artigo completo da Wikipedia com retry e backoff exponencial.
 
    Backoff exponencial: se der erro, espera 2s, depois 4s, depois 8s.
    Isso evita o bloqueio por flood que você estava recebendo.
    """
    titulo_encoded = quote(topico.replace(" ", "_"))
    url = (
        f"https://pt.wikipedia.org/w/api.php"
        f"?action=query&titles={titulo_encoded}"
        f"&prop=extracts&exintro=false&explaintext=true"
        f"&format=json&utf8=1&redirects=1"
    )
 
    for tentativa in range(1, tentativas + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=15) as resp:
                dados = json.loads(resp.read().decode("utf-8"))
 
            pages = dados.get("query", {}).get("pages", {})
            for page in pages.values():
                # -1 significa que a página não existe
                if str(page.get("pageid", -1)) == "-1":
                    return None
                texto = page.get("extract", "")
                if texto and len(texto) > 100:
                    texto = re.sub(r"\n{3,}", "\n\n", texto)
                    return texto[:5000]
 
            return None
 
        except HTTPError as e:
            if e.code == 429:  # Too Many Requests
                espera = (2 ** tentativa) + random.uniform(0, 1)
                print(f"        ⏳ Wikipedia rate limit — aguardando {espera:.1f}s...")
                time.sleep(espera)
            else:
                print(f"        ⚠️  Wikipedia HTTP {e.code} em '{topico}'")
                return None
 
        except Exception as e:
            if tentativa < tentativas:
                espera = (2 ** tentativa) + random.uniform(0, 1)
                print(f"        ⚠️  Wikipedia erro ({e}) — retry em {espera:.1f}s...")
                time.sleep(espera)
            else:
                print(f"        ⚠️  Wikipedia falhou em '{topico}': {e}")
                return None
 
    return None
 
 
# ═══════════════════════════════════════════════════════
# GROQ
# ═══════════════════════════════════════════════════════
 
def _chamar_groq(prompt: str, tentativas: int = 3) -> str | None:
    """
    Chama a API do Groq com retry e backoff exponencial.
 
    O 403 anterior era causado pelo modelo descontinuado (llama3-8b-8192).
    Agora usa llama-3.1-8b-instant que está ativo.
    """
    if not GROQ_API_KEY:
        return None
 
    corpo = json.dumps({
        "model"      : GROQ_MODEL,
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : 1024,
        "temperature": 0.8,
    }).encode("utf-8")
 
    headers = {
        "Content-Type" : "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
 
    for tentativa in range(1, tentativas + 1):
        try:
            req = Request(GROQ_URL, data=corpo, headers=headers, method="POST")
            with urlopen(req, timeout=30) as resp:
                dados = json.loads(resp.read().decode("utf-8"))
            return dados["choices"][0]["message"]["content"]
 
        except HTTPError as e:
            corpo_erro = e.read().decode("utf-8", errors="ignore")
 
            if e.code == 401:
                print(f"\n  ❌ GROQ: Chave inválida (401)")
                print(f"     Verifique sua GROQ_API_KEY no arquivo")
                return None
 
            elif e.code == 403:
                print(f"\n  ❌ GROQ: Acesso negado (403)")
                print(f"     Resposta: {corpo_erro[:200]}")
                print(f"     Verifique se sua chave tem permissão para o modelo '{GROQ_MODEL}'")
                return None
 
            elif e.code == 429:  # Rate limit
                espera = (2 ** tentativa) * 3 + random.uniform(0, 2)
                print(f"        ⏳ Groq rate limit — aguardando {espera:.1f}s...")
                time.sleep(espera)
 
            elif e.code == 503:  # Serviço indisponível
                espera = (2 ** tentativa) + random.uniform(0, 1)
                print(f"        ⏳ Groq indisponível — retry em {espera:.1f}s...")
                time.sleep(espera)
 
            else:
                print(f"        ⚠️  Groq HTTP {e.code}: {corpo_erro[:100]}")
                return None
 
        except Exception as e:
            if tentativa < tentativas:
                espera = (2 ** tentativa) + random.uniform(0, 1)
                print(f"        ⚠️  Groq erro ({e}) — retry em {espera:.1f}s...")
                time.sleep(espera)
            else:
                print(f"        ⚠️  Groq falhou: {e}")
                return None
 
    return None
 
 
def gerar_com_groq(artigo: str, topico: str) -> list[str]:
    """
    Gera textos de treino usando o Groq com base no artigo da Wikipedia.
    Fallback automático para parágrafos brutos se o Groq falhar.
    """
    prompt = f"""Você é um gerador de dados de treino para um modelo de linguagem em português brasileiro.
 
Com base no texto abaixo sobre "{topico}", gere exatamente {TEXTOS_POR_ARTIGO} textos de treino DIFERENTES.
 
Regras estritas:
- Cada texto deve ter entre 2 e 4 frases completas
- Varie os estilos: definição, curiosidade, comparação, exemplo prático, contexto histórico
- Escreva em português brasileiro natural e fluido
- NÃO numere, NÃO use bullet points, NÃO adicione títulos
- Separe cada texto com exatamente uma linha em branco
- Comece direto com o conteúdo, sem introdução
 
TEXTO DA WIKIPEDIA:
{artigo[:2500]}
 
TEXTOS DE TREINO:"""
 
    resposta = _chamar_groq(prompt)
 
    if not resposta:
        return _paragrafos_brutos(artigo)
 
    textos = [t.strip() for t in resposta.split("\n\n") if t.strip()]
    textos = [t for t in textos if len(t) > 30 and not t.startswith("TEXTO")]
    return textos[:TEXTOS_POR_ARTIGO] if textos else _paragrafos_brutos(artigo)
 
 
def _paragrafos_brutos(texto: str) -> list[str]:
    """Fallback: extrai parágrafos do artigo bruto."""
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    return [p for p in paragrafos if len(p) > 50][:TEXTOS_POR_ARTIGO]
 
 
# ═══════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════
 
def alimentar(
    topicos:   list[str],
    categoria: str  = "wikipedia",
    sem_groq:  bool = False,
):
    conn = _conectar()
    usar_groq = bool(GROQ_API_KEY) and not sem_groq
 
    print(f"\n⚛️  ParadoxoX — Alimentador v2")
    print(f"{'─' * 50}")
    print(f"  Tópicos na lista : {len(topicos)}")
    print(f"  Modelo Groq      : {GROQ_MODEL if usar_groq else 'desativado'}")
    print(f"  Textos/artigo    : {TEXTOS_POR_ARTIGO}")
    print(f"  Banco            : {DB_PATH}")
    print(f"{'─' * 50}")
 
    if not GROQ_API_KEY and not sem_groq:
        print("\n  ⚠️  GROQ_API_KEY vazia — salvando artigos brutos")
        print("  Abra o arquivo e cole sua chave em GROQ_API_KEY\n")
 
    # ── Filtra tópicos já processados ────────────────────────────────
    pendentes = []
    pulados   = 0
    for topico in topicos:
        if topico_ja_existe(conn, topico):
            pulados += 1
        else:
            pendentes.append(topico)
 
    if pulados > 0:
        print(f"\n  ⏭️  {pulados} tópicos já estão no banco — pulando")
    print(f"  📋 {len(pendentes)} tópicos para processar\n")
 
    if not pendentes:
        print("✅ Todos os tópicos já foram processados!")
        conn.close()
        return
 
    total_add  = 0
    total_dup  = 0
    total_erro = 0
 
    for i, topico in enumerate(pendentes, 1):
        print(f"[{i:02d}/{len(pendentes)}] 📖 {topico}")
 
        # ── 1. Busca na Wikipedia ──────────────────────────────────
        artigo = buscar_wikipedia(topico)
 
        if not artigo:
            print(f"         ⚠️  Não encontrado na Wikipedia — pulando\n")
            total_erro += 1
            _pausa_wikipedia()
            continue
 
        print(f"         → {len(artigo):,} chars obtidos da Wikipedia")
 
        # ── 2. Gera textos ─────────────────────────────────────────
        if usar_groq:
            textos = gerar_com_groq(artigo, topico)
            print(f"         → {len(textos)} textos gerados pelo Groq")
            time.sleep(PAUSA_GROQ + random.uniform(0, PAUSA_JITTER))
        else:
            textos = _paragrafos_brutos(artigo)
            print(f"         → {len(textos)} parágrafos extraídos")
 
        if not textos:
            print(f"         ⚠️  Nenhum texto gerado — pulando\n")
            total_erro += 1
            _pausa_wikipedia()
            continue
 
        # ── 3. Grava no banco ──────────────────────────────────────
        add, dup = gravar(conn, textos, categoria, topico)
        total_add += add
        total_dup += dup
        print(f"         ✅ +{add} textos gravados  ({dup} duplicados)\n")
 
        # Pausa entre tópicos para não sobrecarregar a Wikipedia
        _pausa_wikipedia()
 
    conn.close()
 
    print(f"{'─' * 50}")
    print(f"⚛️  Concluído!")
    print(f"  Textos adicionados : {total_add:,}")
    print(f"  Duplicados         : {total_dup:,}")
    print(f"  Erros              : {total_erro}")
    print(f"  Banco              : {DB_PATH}")
 
 
# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="⚛️  ParadoxoX — Alimentador Wikipedia + Groq v2"
    )
    parser.add_argument(
        "--topicos", type=int, default=len(TOPICOS_WIKIPEDIA),
        help=f"Quantos tópicos processar (padrão: todos os {len(TOPICOS_WIKIPEDIA)})"
    )
    parser.add_argument(
        "--sem-groq", action="store_true",
        help="Salva artigos brutos sem passar pelo Groq"
    )
    parser.add_argument(
        "--categoria", default="wikipedia",
        help="Categoria dos textos no banco (padrão: wikipedia)"
    )
 
    args = parser.parse_args()
    topicos = TOPICOS_WIKIPEDIA[: args.topicos]
 
    alimentar(
        topicos   = topicos,
        categoria = args.categoria,
        sem_groq  = args.sem_groq,
    )