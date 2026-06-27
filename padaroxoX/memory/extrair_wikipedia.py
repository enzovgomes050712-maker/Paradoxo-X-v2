"""
Extrator de texto limpo do dump XML da Wikipedia em português.
Funciona com Python 3.11 sem dependências externas.

Uso:
    python extrair_wikipedia.py
    python extrair_wikipedia.py --xml outro_caminho.xml
    python extrair_wikipedia.py --limite 50000   (para extrair só 50k artigos)

Saída:
    wikipedia_limpa.txt  → um parágrafo por linha, pronto pro dataset.py
"""

import re
import sys
import argparse
from pathlib import Path
from xml.etree.ElementTree import iterparse


# ── Configurações ──────────────────────────────────────────────────────────────

XML_PADRAO   = "ptwiki-latest-pages-articles.xml"
SAIDA_PADRAO = "wikipedia_limpa.txt"
MINIMO_CHARS = 80   # ignora parágrafos muito curtos
BATCH_LOG    = 1000 # printa progresso a cada N artigos


# ── Limpeza do wikitext ────────────────────────────────────────────────────────

def limpar_wikitext(texto: str) -> list[str]:
    """
    Remove marcações da Wikipedia e retorna lista de parágrafos limpos.
    Não usa nenhuma biblioteca externa — só regex.
    """

    # Remove blocos de template {{ ... }} (aninhados)
    texto = _remover_aninhado(texto, "{{", "}}")

    # Remove blocos de tabela {| ... |}
    texto = _remover_aninhado(texto, "{|", "|}")

    # Remove tags XML/HTML completas com conteúdo
    texto = re.sub(r"<(ref|gallery|math|score|timeline|imagemap|poem|source|syntaxhighlight)[^>]*>.*?</\1>",
                   "", texto, flags=re.DOTALL | re.IGNORECASE)

    # Remove tags HTML restantes (abertura e fechamento)
    texto = re.sub(r"<[^>]+>", "", texto)

    # Remove links com pipe [[Texto|Label]] → mantém Label
    texto = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", texto)

    # Remove links externos [http://... texto] → mantém texto
    texto = re.sub(r"\[https?://[^\s\]]*\s+([^\]]+)\]", r"\1", texto)
    texto = re.sub(r"\[https?://[^\]]*\]", "", texto)

    # Remove negrito/itálico (''' e '')
    texto = re.sub(r"'{2,3}", "", texto)

    # Remove headers (== Título ==)
    texto = re.sub(r"={2,6}[^=]+=+", "", texto)

    # Remove linhas que começam com marcadores de lista ou indent
    texto = re.sub(r"^[*#:;]+\s*", "", texto, flags=re.MULTILINE)

    # Remove categorias, arquivos, imagens
    texto = re.sub(r"\[\[(Categoria|Category|Arquivo|File|Image|Imagem):[^\]]*\]\]",
                   "", texto, flags=re.IGNORECASE)

    # Remove referências soltas
    texto = re.sub(r"<ref[^/]*/?>", "", texto, flags=re.IGNORECASE)

    # Remove entidades HTML (&nbsp; &lt; etc)
    texto = re.sub(r"&[a-zA-Z]+;", " ", texto)
    texto = re.sub(r"&#\d+;", " ", texto)

    # Remove URLs soltas
    texto = re.sub(r"https?://\S+", "", texto)

    # Normaliza espaços múltiplos
    texto = re.sub(r"[ \t]+", " ", texto)

    # Separa em parágrafos (linhas não vazias)
    paragrafos = []
    for linha in texto.splitlines():
        linha = linha.strip()
        # Filtra linhas muito curtas, com muitos símbolos ou que parecem lixo
        if (len(linha) >= MINIMO_CHARS
                and not linha.startswith("|")
                and not linha.startswith("!")
                and _razao_letras(linha) > 0.6):
            paragrafos.append(linha)

    return paragrafos


def _remover_aninhado(texto: str, abre: str, fecha: str) -> str:
    """Remove blocos aninhados como {{ }} e {| |} iterativamente."""
    MAX_ITER = 8
    for _ in range(MAX_ITER):
        novo = re.sub(
            re.escape(abre) + r"[^" + re.escape(abre[0]) + re.escape(fecha[0]) + r"]*?" + re.escape(fecha),
            "",
            texto,
            flags=re.DOTALL
        )
        if novo == texto:
            break
        texto = novo
    return texto


def _razao_letras(texto: str) -> float:
    """Retorna a proporção de letras no texto (filtra lixo)."""
    if not texto:
        return 0.0
    letras = sum(1 for c in texto if c.isalpha())
    return letras / len(texto)


# ── Parser do XML ──────────────────────────────────────────────────────────────

NS = "{http://www.mediawiki.org/xml/export-0.11/}"

def extrair_artigos(caminho_xml: str, limite: int = 0):
    """
    Gera (titulo, texto_limpo) para cada artigo da Wikipedia.
    Usa iterparse para não carregar o XML inteiro na memória.
    """
    contexto = iterparse(caminho_xml, events=("end",))

    titulo_atual = ""
    ns_atual     = ""

    for evento, elem in contexto:
        tag = elem.tag

        if tag == f"{NS}title":
            titulo_atual = elem.text or ""

        elif tag == f"{NS}ns":
            ns_atual = elem.text or ""

        elif tag == f"{NS}text":
            # ns=0 são artigos comuns (ignora categorias, usuários, etc)
            if ns_atual == "0" and elem.text:
                # Ignora redirecionamentos
                if not elem.text.strip().lower().startswith("#redirect"):
                    paragrafos = limpar_wikitext(elem.text)
                    if paragrafos:
                        yield titulo_atual, paragrafos

        # Libera memória — crucial pra arquivos de 11GB
        elem.clear()

        if limite and hasattr(extrair_artigos, "_contador"):
            if extrair_artigos._contador >= limite:
                break


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrai texto limpo do dump XML da Wikipedia")
    parser.add_argument("--xml",    default=XML_PADRAO,   help=f"Caminho do XML (padrão: {XML_PADRAO})")
    parser.add_argument("--saida",  default=SAIDA_PADRAO, help=f"Arquivo de saída (padrão: {SAIDA_PADRAO})")
    parser.add_argument("--limite", type=int, default=0,  help="Máximo de artigos (0 = todos)")
    args = parser.parse_args()

    if not Path(args.xml).exists():
        print(f"❌ Arquivo não encontrado: {args.xml}")
        print(f"   Coloca o script na mesma pasta do XML ou usa --xml caminho/completo.xml")
        sys.exit(1)

    print(f"📖 Lendo: {args.xml}")
    print(f"💾 Salvando em: {args.saida}")
    if args.limite:
        print(f"🔢 Limite: {args.limite} artigos")
    print(f"⏳ Processando... (arquivo grande, pode demorar alguns minutos)\n")

    total_artigos   = 0
    total_paragrafos = 0

    with open(args.saida, "w", encoding="utf-8") as f_saida:
        for titulo, paragrafos in extrair_artigos(args.xml, args.limite):
            total_artigos += 1

            for p in paragrafos:
                f_saida.write(p + "\n")
                total_paragrafos += 1

            # Progresso
            if total_artigos % BATCH_LOG == 0:
                print(f"  ✔ {total_artigos:,} artigos | {total_paragrafos:,} parágrafos", end="\r")

            if args.limite and total_artigos >= args.limite:
                break

    print(f"\n\n✅ Concluído!")
    print(f"   Artigos processados : {total_artigos:,}")
    print(f"   Parágrafos extraídos: {total_paragrafos:,}")
    print(f"   Arquivo gerado      : {args.saida}")
    print(f"\n🚀 Próximo passo:")
    print(f"   python dataset.py add {args.saida} --categoria wikipedia")


if __name__ == "__main__":
    main()