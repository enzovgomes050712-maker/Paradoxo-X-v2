"""
PARADOXO X — Dataset Manager
==============================
Adiciona dados de treino ao tokenizer e à memória
SEM precisar editar nenhum código.

Como usar:
  python dataset_manager.py --help
  python dataset_manager.py add texto.txt
  python dataset_manager.py add dados.json
  python dataset_manager.py add pasta/com/arquivos/
  python dataset_manager.py add "frase direto na linha de comando"
  python dataset_manager.py list
  python dataset_manager.py treinar
  python dataset_manager.py status

Formatos suportados:
  .txt   → cada linha é um exemplo de treino
  .json  → lista de strings  ["frase1", "frase2", ...]
           OU lista de dicts [{"texto": "..."}, {"input": "...", "output": "..."}]
  .csv   → primeira coluna de texto usada (ou coluna "texto"/"text")
  .md    → parágrafos separados por linha em branco
  pasta/ → processa todos os .txt/.json/.csv/.md dentro dela
"""

import argparse
import json
import os
import csv
import sys
import sqlite3
import time
from pathlib import Path
from datetime import datetime


# -------------------------------------------------------
# BANCO DE DADOS DO DATASET
# -------------------------------------------------------

DB_PATH = "paradoxox_dataset.db"

def _conectar() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS exemplos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            texto       TEXT    NOT NULL,
            fonte       TEXT    DEFAULT '',
            categoria   TEXT    DEFAULT 'geral',
            adicionado  REAL    NOT NULL DEFAULT (unixepoch('now','subsec')),
            usado_treino INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historico_treino (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            data        REAL    NOT NULL,
            total_textos INTEGER,
            vocab_size   INTEGER,
            num_merges   INTEGER,
            arquivo_vocab TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fonte ON exemplos(fonte)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cat ON exemplos(categoria)")
    conn.commit()
    return conn


# -------------------------------------------------------
# LEITURA DE ARQUIVOS
# -------------------------------------------------------

def _ler_txt(caminho: Path) -> list[str]:
    """Lê .txt — cada linha não-vazia é um exemplo."""
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        linhas = f.readlines()
    return [l.strip() for l in linhas if l.strip()]


def _ler_json(caminho: Path) -> list[str]:
    """
    Lê .json. Suporta:
      - ["frase1", "frase2"]
      - [{"texto": "..."}, ...]
      - [{"input": "...", "output": "..."}, ...]   (pares pergunta/resposta)
      - {"textos": ["frase1", "frase2"]}
    """
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)

    textos = []

    if isinstance(dados, list):
        for item in dados:
            if isinstance(item, str):
                textos.append(item.strip())
            elif isinstance(item, dict):
                # Tenta chaves comuns
                for chave in ("texto", "text", "content", "conteudo", "message", "mensagem"):
                    if chave in item:
                        textos.append(str(item[chave]).strip())
                        break
                # Par input/output — junta os dois
                if "input" in item and "output" in item:
                    textos.append(str(item["input"]).strip())
                    textos.append(str(item["output"]).strip())
                elif "pergunta" in item and "resposta" in item:
                    textos.append(str(item["pergunta"]).strip())
                    textos.append(str(item["resposta"]).strip())

    elif isinstance(dados, dict):
        # {"textos": [...]}
        for chave in ("textos", "texts", "data", "dados", "examples", "exemplos"):
            if chave in dados and isinstance(dados[chave], list):
                textos.extend(_ler_json_lista(dados[chave]))
                break

    return [t for t in textos if t]


def _ler_json_lista(lista: list) -> list[str]:
    textos = []
    for item in lista:
        if isinstance(item, str):
            textos.append(item.strip())
        elif isinstance(item, dict):
            for chave in ("texto", "text", "content", "conteudo"):
                if chave in item:
                    textos.append(str(item[chave]).strip())
                    break
    return textos


def _ler_csv(caminho: Path) -> list[str]:
    """
    Lê .csv. Usa coluna chamada 'texto', 'text', 'content' ou
    a primeira coluna se nenhuma dessas existir.
    """
    textos = []
    with open(caminho, encoding="utf-8", errors="ignore", newline="") as f:
        leitor = csv.DictReader(f)
        if leitor.fieldnames is None:
            return textos

        # Descobre qual coluna usar
        colunas_alvo = ("texto", "text", "content", "conteudo", "message", "frase")
        coluna = None
        for c in colunas_alvo:
            for fn in leitor.fieldnames:
                if fn.lower() == c:
                    coluna = fn
                    break
            if coluna:
                break

        # Usa primeira coluna se não encontrou
        if not coluna and leitor.fieldnames:
            coluna = leitor.fieldnames[0]

        if coluna:
            for row in leitor:
                val = row.get(coluna, "").strip()
                if val:
                    textos.append(val)

    return textos


def _ler_md(caminho: Path) -> list[str]:
    """
    Lê .md — usa parágrafos (blocos separados por linha em branco).
    Ignora linhas de header (#, ##) como exemplos isolados.
    """
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        conteudo = f.read()

    paragrafos = []
    bloco_atual = []

    for linha in conteudo.splitlines():
        if linha.strip():
            bloco_atual.append(linha.strip())
        else:
            if bloco_atual:
                paragrafo = " ".join(bloco_atual)
                if len(paragrafo) > 10:  # ignora linhas muito curtas
                    paragrafos.append(paragrafo)
                bloco_atual = []

    if bloco_atual:
        paragrafo = " ".join(bloco_atual)
        if len(paragrafo) > 10:
            paragrafos.append(paragrafo)

    return paragrafos


def _ler_arquivo(caminho: Path) -> list[str]:
    """Redireciona para o leitor correto pelo tipo de arquivo."""
    ext = caminho.suffix.lower()
    if ext == ".txt":
        return _ler_txt(caminho)
    elif ext == ".json":
        return _ler_json(caminho)
    elif ext == ".csv":
        return _ler_csv(caminho)
    elif ext in (".md", ".markdown"):
        return _ler_md(caminho)
    else:
        # Tenta como texto simples
        try:
            return _ler_txt(caminho)
        except Exception:
            return []


# -------------------------------------------------------
# COMANDOS
# -------------------------------------------------------

def cmd_add(args):
    """Adiciona exemplos de treino ao dataset."""
    conn = _conectar()
    fonte = args.fonte
    categoria = args.categoria or "geral"
    total_adicionados = 0
    total_duplicados = 0

    entradas = args.entrada  # lista de caminhos ou strings

    todos_textos: list[tuple[str, str]] = []  # (texto, nome_fonte)

    for entrada in entradas:
        caminho = Path(entrada)

        if caminho.is_dir():
            # Processa todos os arquivos suportados na pasta
            extensoes = {".txt", ".json", ".csv", ".md", ".markdown"}
            arquivos = [
                p for p in caminho.rglob("*")
                if p.is_file() and p.suffix.lower() in extensoes
            ]
            print(f"📁 Pasta '{entrada}': {len(arquivos)} arquivo(s) encontrado(s)")
            for arq in sorted(arquivos):
                textos = _ler_arquivo(arq)
                for t in textos:
                    todos_textos.append((t, str(arq)))
                print(f"   ✔ {arq.name}: {len(textos)} exemplo(s)")

        elif caminho.is_file():
            textos = _ler_arquivo(caminho)
            for t in textos:
                todos_textos.append((t, str(caminho)))
            print(f"📄 Arquivo '{caminho.name}': {len(textos)} exemplo(s) lido(s)")

        else:
            # Trata como texto direto
            texto = entrada.strip()
            if texto:
                todos_textos.append((texto, "linha_de_comando"))
                print(f"📝 Texto direto adicionado: '{texto[:60]}...' " if len(texto) > 60 else f"📝 Texto direto: '{texto}'")

    if not todos_textos:
        print("⚠️  Nenhum texto encontrado para adicionar.")
        conn.close()
        return

    # Insere no banco em lote — muito mais rápido que um por um
    # INSERT OR IGNORE ignora duplicatas automaticamente pelo índice único
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_texto_unico ON exemplos(texto)")
    conn.commit()

    dados = [
        (texto, fonte or nome_fonte, categoria)
        for texto, nome_fonte in todos_textos
    ]

    BATCH = 5000
    print(f"💾 Inserindo {len(dados):,} exemplos em lotes de {BATCH}...")

    for i in range(0, len(dados), BATCH):
        lote = dados[i:i + BATCH]
        try:
            cursor = conn.executemany(
                "INSERT OR IGNORE INTO exemplos (texto, fonte, categoria) VALUES (?, ?, ?)",
                lote
            )
            conn.commit()
            total_adicionados += cursor.rowcount if cursor.rowcount >= 0 else len(lote)
        except Exception as e:
            print(f"   ⚠️  Erro no lote {i//BATCH + 1}: {e}")

        progresso = min(i + BATCH, len(dados))
        print(f"   ✔ {progresso:,}/{len(dados):,}", end="\r")

    total_duplicados = max(0, len(dados) - total_adicionados)
    conn.close()

    print(f"\n✅ Dataset atualizado:")
    print(f"   ➕ Adicionados  : {total_adicionados}")
    print(f"   🔁 Duplicados   : {total_duplicados} (ignorados)")
    print(f"   💡 Use 'python dataset_manager.py treinar' para retreinar o tokenizer.")


def cmd_list(args):
    """Lista os exemplos no dataset."""
    conn = _conectar()
    limite = args.limite or 20
    categoria = args.categoria

    if categoria:
        rows = conn.execute(
            "SELECT * FROM exemplos WHERE categoria = ? ORDER BY adicionado DESC LIMIT ?",
            (categoria, limite)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM exemplos ORDER BY adicionado DESC LIMIT ?",
            (limite,)
        ).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM exemplos").fetchone()[0]
    cats = conn.execute(
        "SELECT categoria, COUNT(*) as n FROM exemplos GROUP BY categoria ORDER BY n DESC"
    ).fetchall()

    conn.close()

    print(f"\n📊 DATASET — {total} exemplo(s) total\n")
    print(f"{'ID':<6} {'FONTE':<25} {'CAT':<12} {'TREINO':<8} {'TEXTO'}")
    print("─" * 80)
    for r in rows:
        texto_curto = r["texto"][:45] + "..." if len(r["texto"]) > 45 else r["texto"]
        fonte_curta = Path(r["fonte"]).name[:23] if r["fonte"] else "—"
        usado = "✅" if r["usado_treino"] else "⬜"
        print(f"{r['id']:<6} {fonte_curta:<25} {r['categoria']:<12} {usado:<8} {texto_curto}")

    print(f"\nCategorias: " + ", ".join(f"{r['categoria']}({r['n']})" for r in cats))


def cmd_treinar(args):
    """Treina (ou retreina) o tokenizer com todos os exemplos do dataset."""
    conn = _conectar()

    # Carrega todos os textos
    rows = conn.execute("SELECT id, texto FROM exemplos").fetchall()
    if not rows:
        print("⚠️  Dataset vazio! Adicione exemplos primeiro com: python dataset_manager.py add arquivo.txt")
        conn.close()
        return

    textos = [r["texto"] for r in rows]
    ids_rows = [r["id"] for r in rows]
    num_merges = args.num_merges or 1000

    print(f"🔢 Treinando tokenizer com {len(textos)} exemplo(s) (num_merges={num_merges})...")

    # Importa o Tokenizer do projeto (COM A CORREÇÃO DE CAMINHO ABSOLUTO)
    try:
        from pathlib import Path
        
        # Pega o caminho absoluto da raiz do projeto (padaroxox)
        raiz_do_projeto = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(raiz_do_projeto))
        
        # Importa o Tokenizer de dentro da pasta 'core'
        from core.tokenizer import Tokenizer
        
    except ImportError as e:
        # Se der erro, ele vai te dizer exatamente ONDE procurou
        print(f"❌ Erro ao importar Tokenizer: {e}")
        print(f"💡 O Python procurou na raiz: {Path(__file__).resolve().parent.parent}")
        conn.close()
        return

    t = Tokenizer()
    t.treinar(textos, num_merges=num_merges)

    vocab_path = args.saida or "vocab.json"
    t.salvar(vocab_path)

    # Marca todos como usados no treino
    conn.executemany(
        "UPDATE exemplos SET usado_treino = 1 WHERE id = ?",
        [(i,) for i in ids_rows]
    )

    # Registra o treino no histórico
    conn.execute(
        """INSERT INTO historico_treino (data, total_textos, vocab_size, num_merges, arquivo_vocab)
           VALUES (?, ?, ?, ?, ?)""",
        (time.time(), len(textos), t.tamanho_vocab(), len(t.merges), vocab_path)
    )
    conn.commit()
    conn.close()

    print(f"\n🎉 Treino concluído!")
    print(f"   Exemplos usados : {len(textos)}")
    print(f"   Tamanho do vocab: {t.tamanho_vocab()} tokens")
    print(f"   Merges BPE      : {len(t.merges)}")
    print(f"   Vocab salvo em  : {vocab_path}")


def cmd_remover(args):
    """Remove exemplos do dataset por ID ou fonte."""
    conn = _conectar()

    if args.id:
        conn.execute("DELETE FROM exemplos WHERE id = ?", (args.id,))
        conn.commit()
        print(f"🗑️  Exemplo #{args.id} removido.")

    elif args.fonte:
        cur = conn.execute("DELETE FROM exemplos WHERE fonte LIKE ?", (f"%{args.fonte}%",))
        conn.commit()
        print(f"🗑️  {cur.rowcount} exemplo(s) removido(s) da fonte '{args.fonte}'.")

    elif args.categoria:
        cur = conn.execute("DELETE FROM exemplos WHERE categoria = ?", (args.categoria,))
        conn.commit()
        print(f"🗑️  {cur.rowcount} exemplo(s) da categoria '{args.categoria}' removidos.")

    elif args.tudo:
        confirm = input("⚠️  Apagar TUDO? Digite 'sim' para confirmar: ")
        if confirm.strip().lower() == "sim":
            conn.execute("DELETE FROM exemplos")
            conn.commit()
            print("🗑️  Dataset limpo.")
        else:
            print("Cancelado.")

    conn.close()


def cmd_status(args):
    """Mostra estatísticas do dataset e histórico de treinos."""
    conn = _conectar()

    total = conn.execute("SELECT COUNT(*) FROM exemplos").fetchone()[0]
    treinados = conn.execute("SELECT COUNT(*) FROM exemplos WHERE usado_treino=1").fetchone()[0]
    pendentes = total - treinados

    cats = conn.execute(
        "SELECT categoria, COUNT(*) as n FROM exemplos GROUP BY categoria ORDER BY n DESC"
    ).fetchall()

    fontes = conn.execute(
        "SELECT fonte, COUNT(*) as n FROM exemplos GROUP BY fonte ORDER BY n DESC LIMIT 10"
    ).fetchall()

    treinos = conn.execute(
        "SELECT * FROM historico_treino ORDER BY data DESC LIMIT 5"
    ).fetchall()

    conn.close()

    print(f"\n⚛️  PARADOXO X — Dataset Status")
    print(f"{'─'*45}")
    print(f"  Total de exemplos   : {total:,}")
    print(f"  Já treinados        : {treinados:,}")
    print(f"  Pendentes de treino : {pendentes:,}")

    if cats:
        print(f"\n  Categorias:")
        for r in cats:
            bar = "█" * min(20, int(r["n"] / max(1, total) * 20))
            print(f"    {r['categoria']:<18} {r['n']:>5}  {bar}")

    if fontes:
        print(f"\n  Top fontes:")
        for r in fontes:
            nome = Path(r["fonte"]).name if r["fonte"] else "—"
            print(f"    {nome[:35]:<36} {r['n']:>5}")

    if treinos:
        print(f"\n  Histórico de treinos:")
        for r in treinos:
            data = datetime.fromtimestamp(r["data"]).strftime("%d/%m/%Y %H:%M")
            print(f"    {data}  vocab={r['vocab_size']:,}  merges={r['num_merges']}  exemplos={r['total_textos']:,}")
    else:
        print(f"\n  ⚠️  Nenhum treino realizado ainda.")
        print(f"     Execute: python dataset_manager.py treinar")

    if pendentes > 0:
        print(f"\n  💡 {pendentes} exemplo(s) novo(s) aguardando treino.")
        print(f"     Execute: python dataset_manager.py treinar")


def cmd_exportar(args):
    """Exporta o dataset para um arquivo JSON ou TXT."""
    conn = _conectar()
    categoria = args.categoria
    saida = Path(args.saida)

    if categoria:
        rows = conn.execute(
            "SELECT texto, fonte, categoria FROM exemplos WHERE categoria = ?",
            (categoria,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT texto, fonte, categoria FROM exemplos"
        ).fetchall()

    conn.close()

    if saida.suffix == ".json":
        dados = [{"texto": r["texto"], "fonte": r["fonte"], "categoria": r["categoria"]} for r in rows]
        with open(saida, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    else:
        with open(saida, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(r["texto"] + "\n")

    print(f"📤 {len(rows)} exemplo(s) exportado(s) para '{saida}'")


# -------------------------------------------------------
# CLI — ARGPARSE
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="dataset_manager",
        description="⚛️  ParadoxoX — Gerenciador de Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Adicionar de arquivo
  python dataset_manager.py add meu_texto.txt
  python dataset_manager.py add dados.json --categoria codigo
  python dataset_manager.py add pasta_com_dados/

  # Adicionar texto direto
  python dataset_manager.py add "como fazer um loop em python"

  # Listar exemplos
  python dataset_manager.py list
  python dataset_manager.py list --limite 50 --categoria codigo

  # Treinar o tokenizer
  python dataset_manager.py treinar
  python dataset_manager.py treinar --num-merges 2000 --saida vocab_v2.json

  # Ver estatísticas
  python dataset_manager.py status

  # Remover
  python dataset_manager.py remover --id 42
  python dataset_manager.py remover --fonte meu_texto.txt
  python dataset_manager.py remover --tudo

  # Exportar
  python dataset_manager.py exportar --saida backup.json
        """
    )

    sub = parser.add_subparsers(dest="comando", required=True)

    # --- add ---
    p_add = sub.add_parser("add", help="Adiciona exemplos ao dataset")
    p_add.add_argument(
        "entrada", nargs="+",
        help="Arquivo(s), pasta(s) ou texto(s) direto(s)"
    )
    p_add.add_argument("--categoria", "-c", default="geral", help="Categoria dos exemplos (padrão: geral)")
    p_add.add_argument("--fonte", "-f", help="Nome da fonte (padrão: nome do arquivo)")
    p_add.set_defaults(func=cmd_add)

    # --- list ---
    p_list = sub.add_parser("list", help="Lista exemplos no dataset")
    p_list.add_argument("--limite", "-n", type=int, default=20, help="Quantos exemplos mostrar (padrão: 20)")
    p_list.add_argument("--categoria", "-c", help="Filtrar por categoria")
    p_list.set_defaults(func=cmd_list)

    # --- treinar ---
    p_train = sub.add_parser("treinar", help="Treina o tokenizer com o dataset atual")
    p_train.add_argument("--num-merges", "-m", type=int, default=1000, help="Número de merges BPE (padrão: 1000)")
    p_train.add_argument("--saida", "-o", default="vocab.json", help="Arquivo de saída do vocab (padrão: vocab.json)")
    p_train.set_defaults(func=cmd_treinar)

    # --- remover ---
    p_rm = sub.add_parser("remover", help="Remove exemplos do dataset")
    g = p_rm.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="Remove por ID")
    g.add_argument("--fonte", help="Remove todos de uma fonte")
    g.add_argument("--categoria", help="Remove todos de uma categoria")
    g.add_argument("--tudo", action="store_true", help="Remove TUDO (pede confirmação)")
    p_rm.set_defaults(func=cmd_remover)

    # --- status ---
    p_status = sub.add_parser("status", help="Mostra estatísticas do dataset")
    p_status.set_defaults(func=cmd_status)

    # --- exportar ---
    p_exp = sub.add_parser("exportar", help="Exporta o dataset para arquivo")
    p_exp.add_argument("--saida", "-o", default="dataset_export.json", help="Arquivo de saída")
    p_exp.add_argument("--categoria", "-c", help="Exportar só uma categoria")
    p_exp.set_defaults(func=cmd_exportar)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()