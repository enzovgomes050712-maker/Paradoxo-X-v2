"""
PARADOXO X — Code Engine / Analyzer
=====================================
Analisa QUALQUER linguagem de programação.

O que ele faz:
  1. Detecta automaticamente a linguagem do código
  2. Analisa estrutura: funções, classes, imports, etc.
  3. Detecta problemas: bugs, code smells, más práticas
  4. Calcula métricas: complexidade, tamanho, qualidade
  5. Gera um relatório completo com sugestões

Linguagens suportadas:
  Python, JavaScript, TypeScript, Java, C, C++, C#,
  Go, Rust, Ruby, PHP, Swift, Kotlin, Dart, Lua,
  Shell/Bash, SQL, HTML, CSS, JSON, YAML, e mais.
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional





# -------------------------------------------------------
# ESTRUTURAS DE DADOS
# -------------------------------------------------------

@dataclass
class Problema:
    """Um problema encontrado no código."""
    tipo: str           # "erro", "aviso", "sugestao", "info"
    categoria: str      # "sintaxe", "seguranca", "qualidade", etc.
    linha: int          # linha onde está o problema (-1 = geral)
    mensagem: str       # descrição do problema
    sugestao: str       # como corrigir
    severidade: int     # 1 (leve) a 5 (crítico)

    def __str__(self):
        icone = {"erro": "🔴", "aviso": "🟡", "sugestao": "🔵", "info": "⚪"}.get(self.tipo, "❓")
        linha_str = f"Linha {self.linha}" if self.linha > 0 else "Geral"
        return f"{icone} [{linha_str}] {self.mensagem}\n   → {self.sugestao}"


@dataclass
class MetricasCodigo:
    """Métricas calculadas sobre o código."""
    linguagem: str = "desconhecida"
    total_linhas: int = 0
    linhas_codigo: int = 0          # sem comentários e vazias
    linhas_comentario: int = 0
    linhas_vazias: int = 0
    num_funcoes: int = 0
    num_classes: int = 0
    num_imports: int = 0
    complexidade_estimada: int = 0  # baseado em ifs, loops, etc.
    profundidade_max: int = 0       # nesting máximo
    funcao_maior: int = 0           # linhas da maior função
    cobertura_docs: float = 0.0     # % de funções com docstring/comentário
    score_qualidade: float = 0.0    # 0.0 a 10.0


@dataclass
class ResultadoAnalise:
    """Resultado completo da análise."""
    metricas: MetricasCodigo = field(default_factory=MetricasCodigo)
    problemas: list = field(default_factory=list)
    funcoes: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    imports: list = field(default_factory=list)
    sugestoes_gerais: list = field(default_factory=list)

    def problemas_por_severidade(self, min_sev: int = 1):
        return [p for p in self.problemas if p.severidade >= min_sev]

    def resumo(self) -> str:
        erros    = len([p for p in self.problemas if p.tipo == "erro"])
        avisos   = len([p for p in self.problemas if p.tipo == "aviso"])
        sugest   = len([p for p in self.problemas if p.tipo == "sugestao"])
        return (f"🔴 {erros} erros  🟡 {avisos} avisos  🔵 {sugest} sugestões  "
                f"| Score: {self.metricas.score_qualidade:.1f}/10")


# -------------------------------------------------------
# DETECTOR DE LINGUAGEM
# -------------------------------------------------------

class DetectorLinguagem:
    """
    Detecta automaticamente a linguagem de programação.
    Usa extensão de arquivo (se fornecida) + padrões no código.
    """

    # Extensão → linguagem
    EXTENSOES = {
        ".py": "python", ".pyw": "python",
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".java": "java",
        ".c": "c", ".h": "c",
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin", ".kts": "kotlin",
        ".dart": "dart",
        ".lua": "lua",
        ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".sql": "sql",
        ".html": "html", ".htm": "html",
        ".css": "css", ".scss": "css", ".sass": "css",
        ".json": "json",
        ".yaml": "yaml", ".yml": "yaml",
        ".xml": "xml",
        ".r": "r", ".R": "r",
        ".scala": "scala",
        ".ex": "elixir", ".exs": "elixir",
        ".hs": "haskell",
        ".pl": "perl",
        ".vim": "vimscript",
        ".ps1": "powershell",
        ".tf": "terraform",
        ".md": "markdown",
    }

    # Padrões únicos por linguagem (pra quando não tem extensão)
    PADROES = {
        "python": [
            r"^\s*def\s+\w+\s*\(", r"^\s*class\s+\w+",
            r"^\s*import\s+\w+", r"^\s*from\s+\w+\s+import",
            r"if\s+__name__\s*==\s*['\"]__main__['\"]",
            r"^\s*#.*", r":\s*$", r"print\s*\(",
        ],
        "javascript": [
            r"^\s*const\s+\w+\s*=", r"^\s*let\s+\w+\s*=",
            r"^\s*var\s+\w+\s*=", r"function\s+\w+\s*\(",
            r"=>\s*[{(]", r"console\.log\s*\(",
            r"require\s*\(", r"module\.exports",
        ],
        "typescript": [
            r":\s*(string|number|boolean|any|void|never)\b",
            r"interface\s+\w+", r"type\s+\w+\s*=",
            r"<\w+>", r"as\s+\w+",
        ],
        "java": [
            r"public\s+class\s+\w+", r"public\s+static\s+void\s+main",
            r"System\.out\.print", r"import\s+java\.",
            r"private|protected|public", r"@Override",
        ],
        "c": [
            r"#include\s*[<\"]", r"int\s+main\s*\(",
            r"printf\s*\(", r"scanf\s*\(",
            r"->\s*\w+", r"malloc\s*\(", r"free\s*\(",
        ],
        "cpp": [
            r"#include\s*<(iostream|vector|string|map)",
            r"std::", r"cout\s*<<", r"cin\s*>>",
            r"class\s+\w+\s*[:{]", r"template\s*<",
        ],
        "csharp": [
            r"using\s+System", r"namespace\s+\w+",
            r"Console\.Write", r"public\s+class\s+\w+",
            r"\.NET", r"async\s+Task",
        ],
        "go": [
            r"^package\s+\w+", r"^import\s*\(",
            r"func\s+\w+\s*\(", r"fmt\.Print",
            r":=", r"goroutine", r"chan\s+",
        ],
        "rust": [
            r"fn\s+\w+\s*\(", r"let\s+mut\s+\w+",
            r"println!\s*\(", r"use\s+std::",
            r"impl\s+\w+", r"->\s*Result<",
            r"&str\b", r"Vec<",
        ],
        "ruby": [
            r"^\s*def\s+\w+", r"^\s*end\b",
            r"puts\s+", r"require\s+['\"]",
            r"\.each\s+do", r"attr_accessor",
        ],
        "php": [
            r"<\?php", r"\$\w+\s*=",
            r"echo\s+", r"function\s+\w+\s*\(",
            r"->", r"::",
        ],
        "swift": [
            r"import\s+(UIKit|Foundation|SwiftUI)",
            r"var\s+\w+\s*:", r"let\s+\w+\s*:",
            r"func\s+\w+\s*\(", r"guard\s+let",
            r"@IBOutlet", r"override\s+func",
        ],
        "kotlin": [
            r"fun\s+\w+\s*\(", r"val\s+\w+\s*=",
            r"var\s+\w+\s*:", r"data\s+class",
            r"println\s*\(", r"import\s+kotlin\.",
        ],
        "bash": [
            r"^#!/bin/(bash|sh|zsh)", r"^\s*echo\s+",
            r"\$\{?\w+\}?", r"^\s*if\s+\[",
            r"^\s*for\s+\w+\s+in\s+", r"\|\s*grep",
        ],
        "sql": [
            r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b",
            r"\bINSERT\s+INTO\b", r"\bCREATE\s+TABLE\b",
            r"\bJOIN\b", r"\bGROUP\s+BY\b",
        ],
        "html": [
            r"<!DOCTYPE\s+html", r"<html", r"<head>",
            r"<body>", r"<div", r"<script",
        ],
        "css": [
            r"\{[^}]*:\s*[^}]+\}", r"@media\s+",
            r":\s*hover", r"#\w+\s*\{", r"\.\w+\s*\{",
        ],
    }

    def detectar(self, codigo: str, nome_arquivo: str = "") -> str:
        # Tenta pela extensão primeiro
        if nome_arquivo:
            ext = "." + nome_arquivo.rsplit(".", 1)[-1].lower() if "." in nome_arquivo else ""
            if ext in self.EXTENSOES:
                return self.EXTENSOES[ext]

        # Tenta pelos padrões no código
        scores = {}
        for lang, padroes in self.PADROES.items():
            score = 0
            for padrao in padroes:
                matches = len(re.findall(padrao, codigo, re.MULTILINE | re.IGNORECASE))
                score += matches
            if score > 0:
                scores[lang] = score

        if scores:
            return max(scores, key=scores.get)

        return "desconhecida"

    def info_linguagem(self, lang: str) -> dict:
        """Retorna informações sobre a linguagem para guiar a análise."""
        infos = {
            "python":     {"comentario": "#",  "bloco": ('"""', "'''"), "indentacao": True,  "tipagem": "dinamica"},
            "javascript": {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "dinamica"},
            "typescript": {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "java":       {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "c":          {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "cpp":        {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "csharp":     {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "go":         {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "rust":       {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "ruby":       {"comentario": "#",  "bloco": ("=begin","=end"), "indentacao": False, "tipagem": "dinamica"},
            "php":        {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "dinamica"},
            "swift":      {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "kotlin":     {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "go":         {"comentario": "//", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
            "bash":       {"comentario": "#",  "bloco": None,           "indentacao": False, "tipagem": "nenhuma"},
            "sql":        {"comentario": "--", "bloco": ("/*", "*/"),   "indentacao": False, "tipagem": "estatica"},
        }
        return infos.get(lang, {"comentario": "//", "bloco": ("/*", "*/"), "indentacao": False, "tipagem": "dinamica"})


# -------------------------------------------------------
# ANALISADORES ESPECÍFICOS POR LINGUAGEM
# -------------------------------------------------------





class CodeAnalyzer:
    """
    Ponto de entrada único para análise de qualquer código.

    Uso:
        analyzer = CodeAnalyzer()
        resultado = analyzer.analisar(codigo, nome_arquivo="meu_script.py")
        print(analyzer.relatorio(resultado))
    """

    def __init__(self):
        self.detector = DetectorLinguagem()
        self.universal = AnalisadorUniversal()
        self.py = AnalisadorPython()
        self.js = AnalisadorJS()
        self.generico = AnalisadorGenerico()

    def analisar(self, codigo: str, nome_arquivo: str = "") -> ResultadoAnalise:
        """
        Analisa o código e retorna um ResultadoAnalise completo.

        Parâmetros:
          codigo        → string com o código-fonte
          nome_arquivo  → opcional, ajuda na detecção da linguagem
        """
        resultado = ResultadoAnalise()
        linhas = codigo.splitlines()

        # 1. Detecta a linguagem
        lang = self.detector.detectar(codigo, nome_arquivo)
        info_lang = self.detector.info_linguagem(lang)

        # 2. Métricas básicas
        total, cod, coment, vazias = self.universal.analisar_linhas(linhas, lang, info_lang)
        profundidade = self.universal.calcular_profundidade(linhas, lang)
        complexidade = self.universal.calcular_complexidade(codigo, lang)

        # 3. Extrai estruturas (funções, classes, imports)
        funcoes, classes, imports = [], [], []

        if lang == "python":
            funcoes = self.py.extrair_funcoes(linhas)
            classes = self.py.extrair_classes(linhas)
            imports = self.py.extrair_imports(linhas)
        elif lang in ("javascript", "typescript"):
            funcoes = self.js.extrair_funcoes(linhas)
        else:
            funcoes = self.generico.extrair_funcoes(linhas, lang)

        # Calcula tamanho de cada função
        for k in range(len(funcoes) - 1):
            funcoes[k]["linhas"] = funcoes[k + 1]["linha_inicio"] - funcoes[k]["linha_inicio"] - 1
        if funcoes:
            funcoes[-1]["linhas"] = total - funcoes[-1]["linha_inicio"]

        funcao_maior = max((f["linhas"] for f in funcoes), default=0)
        docs_com = sum(1 for f in funcoes if f.get("tem_docstring", False))
        cobertura_docs = (docs_com / len(funcoes) * 100) if funcoes else 0.0

        # 4. Detecta problemas
        problemas = []

        if lang == "python":
            problemas += self.py.detectar_problemas(linhas, funcoes, imports)
        elif lang in ("javascript", "typescript"):
            problemas += self.js.detectar_problemas(linhas)

        # Problemas genéricos pra qualquer linguagem
        problemas += self.generico.detectar_problemas_gerais(linhas)

        # 5. Calcula score de qualidade (0-10)
        score = 10.0
        erros_criticos = sum(1 for p in problemas if p.severidade >= 4)
        avisos         = sum(1 for p in problemas if p.severidade == 3)
        sugestoes      = sum(1 for p in problemas if p.severidade <= 2)

        score -= erros_criticos * 2.0
        score -= avisos * 0.5
        score -= sugestoes * 0.1
        score -= max(0, (complexidade - 10) * 0.05)
        score -= max(0, (profundidade - 5) * 0.3)
        if funcao_maior > 100:
            score -= 1.0
        score = max(0.0, min(10.0, score))

        # 6. Monta resultado
        resultado.metricas = MetricasCodigo(
            linguagem=lang,
            total_linhas=total,
            linhas_codigo=cod,
            linhas_comentario=coment,
            linhas_vazias=vazias,
            num_funcoes=len(funcoes),
            num_classes=len(classes),
            num_imports=len(imports),
            complexidade_estimada=complexidade,
            profundidade_max=profundidade,
            funcao_maior=funcao_maior,
            cobertura_docs=cobertura_docs,
            score_qualidade=score
        )
        resultado.problemas = sorted(problemas, key=lambda p: (-p.severidade, p.linha))
        resultado.funcoes   = funcoes
        resultado.classes   = classes
        resultado.imports   = imports

        # Sugestões gerais baseadas nas métricas
        if complexidade > 20:
            resultado.sugestoes_gerais.append(
                "⚠️  Complexidade alta — considere dividir em módulos menores"
            )
        if cobertura_docs < 50 and len(funcoes) > 3:
            resultado.sugestoes_gerais.append(
                "📝 Menos de 50% das funções têm documentação — adicione docstrings/comentários"
            )
        if funcao_maior > 80:
            resultado.sugestoes_gerais.append(
                "✂️  Há funções muito longas — idealmente < 50 linhas por função"
            )
        if profundidade > 6:
            resultado.sugestoes_gerais.append(
                "📐 Nesting muito profundo — use early return / guard clauses pra achatar"
            )

        return resultado

    def relatorio(self, resultado: ResultadoAnalise, verbose: bool = True) -> str:
        """
        Gera um relatório legível da análise.

        Parâmetros:
          verbose → se True, mostra todos os problemas; se False, só o resumo
        """
        m = resultado.metricas
        linhas = []

        # Cabeçalho
        linhas.append("=" * 60)
        linhas.append(f"  PARADOXO X — ANÁLISE DE CÓDIGO")
        linhas.append("=" * 60)
        linhas.append(f"  Linguagem detectada : {m.linguagem.upper()}")
        linhas.append(f"  Score de qualidade  : {m.score_qualidade:.1f}/10  {self._barra(m.score_qualidade)}")
        linhas.append("")

        # Métricas
        linhas.append("── MÉTRICAS ──────────────────────────────────────────")
        linhas.append(f"  Total de linhas     : {m.total_linhas}")
        linhas.append(f"  Linhas de código    : {m.linhas_codigo}")
        linhas.append(f"  Linhas de comentário: {m.linhas_comentario}")
        linhas.append(f"  Funções             : {m.num_funcoes}")
        linhas.append(f"  Classes             : {m.num_classes}")
        linhas.append(f"  Imports             : {m.num_imports}")
        linhas.append(f"  Complexidade est.   : {m.complexidade_estimada}")
        linhas.append(f"  Profundidade máx    : {m.profundidade_max}")
        linhas.append(f"  Maior função        : {m.funcao_maior} linhas")
        linhas.append(f"  Cobertura de docs   : {m.cobertura_docs:.0f}%")
        linhas.append("")

        # Resumo de problemas
        linhas.append("── PROBLEMAS ─────────────────────────────────────────")
        linhas.append(f"  {resultado.resumo()}")
        linhas.append("")

        if verbose and resultado.problemas:
            for p in resultado.problemas:
                linhas.append(str(p))
                linhas.append("")

        # Sugestões gerais
        if resultado.sugestoes_gerais:
            linhas.append("── SUGESTÕES GERAIS ──────────────────────────────────")
            for s in resultado.sugestoes_gerais:
                linhas.append(f"  {s}")
            linhas.append("")

        # Funções detectadas
        if resultado.funcoes and verbose:
            linhas.append("── FUNÇÕES DETECTADAS ────────────────────────────────")
            for f in resultado.funcoes:
                docs = "✅" if f.get("tem_docstring") else "❌"
                linhas.append(
                    f"  {docs} {f['nome']}({len(f['params'])} params) "
                    f"— linha {f['linha_inicio']} "
                    f"[{f['linhas']} linhas]"
                )
            linhas.append("")

        linhas.append("=" * 60)
        return "\n".join(linhas)

    def _barra(self, score: float) -> str:
        """Barra visual de score."""
        filled = int(score)
        empty  = 10 - filled
        cor = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
        return f"{'█' * filled}{'░' * empty} {cor}"

    def analisar_arquivo(self, caminho: str) -> ResultadoAnalise:
        """Lê e analisa um arquivo diretamente."""
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            codigo = f.read()
        nome = os.path.basename(caminho) if "os" in dir() else caminho.split("/")[-1]
        return self.analisar(codigo, nome_arquivo=nome)

    def analisar_multiplos(self, arquivos: dict) -> dict:
        """
        Analisa múltiplos arquivos de uma vez.

        Parâmetros:
          arquivos → {"nome.py": "código...", "outro.js": "código..."}

        Retorna:
          {"nome.py": ResultadoAnalise, ...}
        """
        resultados = {}
        for nome, codigo in arquivos.items():
            resultados[nome] = self.analisar(codigo, nome_arquivo=nome)
        return resultados













class AnalisadorUniversal:
    """
    Análises que funcionam em QUALQUER linguagem.
    Baseado em padrões de texto e estrutura geral.
    """

    def analisar_linhas(self, linhas: list[str], lang: str, info: dict) -> tuple:
        """Conta e classifica todas as linhas."""
        total = len(linhas)
        vazias = sum(1 for l in linhas if not l.strip())
        comentarios = 0
        codigo = 0

        comentario_char = info.get("comentario", "//")
        em_bloco = False
        bloco = info.get("bloco")

        for linha in linhas:
            stripped = linha.strip()
            if not stripped:
                continue

            # Detecta comentários em bloco
            if bloco:
                if bloco[0] in stripped:
                    em_bloco = True
                if em_bloco:
                    comentarios += 1
                    if bloco[1] in stripped:
                        em_bloco = False
                    continue

            if stripped.startswith(comentario_char):
                comentarios += 1
            else:
                codigo += 1

        return total, codigo, comentarios, vazias

    def calcular_profundidade(self, linhas: list[str], lang: str) -> int:
        """Calcula o nesting máximo (profundidade de indentação)."""
        max_prof = 0

        if lang in ("python",):
            # Python usa indentação real
            for linha in linhas:
                if linha.strip():
                    espacos = len(linha) - len(linha.lstrip())
                    prof = espacos // 4
                    max_prof = max(max_prof, prof)
        else:
            # Outras linguagens: conta chaves abertas
            profundidade = 0
            for linha in linhas:
                profundidade += linha.count("{") - linha.count("}")
                max_prof = max(max_prof, profundidade)

        return max_prof

    def calcular_complexidade(self, codigo: str, lang: str) -> int:
        """
        Complexidade ciclomática estimada.
        Conta pontos de decisão: if, else, for, while, case, catch, etc.
        """
        # Palavras-chave de decisão universais
        palavras_decisao = [
            r'\bif\b', r'\belse\b', r'\belif\b', r'\bfor\b', r'\bwhile\b',
            r'\bcase\b', r'\bswitch\b', r'\bcatch\b', r'\bexcept\b',
            r'\band\b', r'\bor\b', r'\?\?', r'\?\s', r'&&', r'\|\|',
        ]
        complexidade = 1  # base
        for padrao in palavras_decisao:
            complexidade += len(re.findall(padrao, codigo))
        return complexidade


class AnalisadorPython:
    """Análises específicas para Python."""

    def extrair_funcoes(self, linhas: list[str]) -> list[dict]:
        funcoes = []
        atual = None
        indent_base = 0

        for i, linha in enumerate(linhas, 1):
            match = re.match(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)', linha)
            if match:
                indent = len(match.group(1))
                nome = match.group(2)
                params = [p.strip() for p in match.group(3).split(",") if p.strip()]
                if atual:
                    atual["linhas"] = i - atual["linha_inicio"] - 1
                    funcoes.append(atual)
                atual = {
                    "nome": nome,
                    "linha_inicio": i,
                    "params": params,
                    "num_params": len(params),
                    "indent": indent,
                    "tem_docstring": False,
                    "linhas": 0
                }
                indent_base = indent

            # Detecta docstring logo após def
            if atual and i == atual["linha_inicio"] + 1:
                stripped = linha.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    atual["tem_docstring"] = True

        if atual:
            atual["linhas"] = len(linhas) - atual["linha_inicio"]
            funcoes.append(atual)

        return funcoes

    def extrair_classes(self, linhas: list[str]) -> list[dict]:
        classes = []
        for i, linha in enumerate(linhas, 1):
            match = re.match(r'^\s*class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:', linha)
            if match:
                classes.append({
                    "nome": match.group(1),
                    "heranca": match.group(2) or "",
                    "linha": i
                })
        return classes

    def extrair_imports(self, linhas: list[str]) -> list[dict]:
        imports = []
        for i, linha in enumerate(linhas, 1):
            m1 = re.match(r'^\s*import\s+(.+)', linha)
            m2 = re.match(r'^\s*from\s+(\S+)\s+import\s+(.+)', linha)
            if m1:
                imports.append({"modulo": m1.group(1).strip(), "linha": i, "tipo": "import"})
            elif m2:
                imports.append({"modulo": m2.group(1), "items": m2.group(2).strip(), "linha": i, "tipo": "from"})
        return imports

    def detectar_problemas(self, linhas: list[str], funcoes: list, imports: list) -> list[Problema]:
        problemas = []
        codigo_completo = "\n".join(linhas)

        for i, linha in enumerate(linhas, 1):
            stripped = linha.strip()

            # Bare except
            if re.match(r'^\s*except\s*:', linha):
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=i,
                    mensagem="'except:' sem especificar exceção captura TUDO, incluindo KeyboardInterrupt",
                    sugestao="Use 'except Exception as e:' ou seja específico: 'except ValueError:'",
                    severidade=3
                ))

            # Comparação com None usando ==
            if re.search(r'==\s*None\b|None\s*==', linha) and "!=" not in linha:
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=i,
                    mensagem="Use 'is None' em vez de '== None'",
                    sugestao="Python recomenda: 'if x is None:' — mais correto semanticamente",
                    severidade=2
                ))

            # Comparação com True/False usando ==
            if re.search(r'==\s*True\b|==\s*False\b', linha):
                problemas.append(Problema(
                    tipo="sugestao", categoria="qualidade", linha=i,
                    mensagem="Comparação desnecessária com True/False",
                    sugestao="Use 'if condicao:' em vez de 'if condicao == True:'",
                    severidade=1
                ))

            # print sem logging
            if re.match(r'^\s*print\s*\(', linha) and "debug" not in linha.lower():
                problemas.append(Problema(
                    tipo="info", categoria="qualidade", linha=i,
                    mensagem="print() encontrado — em produção considere usar logging",
                    sugestao="Use 'import logging; logging.debug(...)' para controle melhor de logs",
                    severidade=1
                ))

            # eval/exec perigosos
            if re.match(r'^\s*eval\s*\(|^\s*exec\s*\(', linha):
                problemas.append(Problema(
                    tipo="erro", categoria="seguranca", linha=i,
                    mensagem="eval()/exec() com input externo é uma vulnerabilidade crítica de segurança",
                    sugestao="Evite eval/exec com dados do usuário. Use ast.literal_eval() se precisar avaliar literais",
                    severidade=5
                ))

            # Senha/token hardcoded
            if re.search(r'(password|senha|token|secret|api_key)\s*=\s*["\'][^"\']+["\']', linha, re.IGNORECASE):
                problemas.append(Problema(
                    tipo="erro", categoria="seguranca", linha=i,
                    mensagem="Credencial hardcoded no código! Isso é uma vulnerabilidade grave",
                    sugestao="Use variáveis de ambiente: os.environ.get('MINHA_SENHA') ou um arquivo .env",
                    severidade=5
                ))

            # Linha muito longa
            if len(linha.rstrip()) > 120:
                problemas.append(Problema(
                    tipo="sugestao", categoria="estilo", linha=i,
                    mensagem=f"Linha muito longa ({len(linha.rstrip())} chars) — dificulta leitura",
                    sugestao="Quebre em múltiplas linhas. PEP 8 recomenda máx 79 chars (prático: 100-120)",
                    severidade=1
                ))

            # Função com muitos parâmetros
        for func in funcoes:
            if func["num_params"] > 6:
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=func["linha_inicio"],
                    mensagem=f"Função '{func['nome']}' tem {func['num_params']} parâmetros — muito acoplada",
                    sugestao="Considere agrupar parâmetros num objeto/dataclass ou dividir a função",
                    severidade=3
                ))

            # Função muito grande
            if func["linhas"] > 50:
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=func["linha_inicio"],
                    mensagem=f"Função '{func['nome']}' tem {func['linhas']} linhas — faça mais curta",
                    sugestao="Funções grandes são difíceis de testar e entender. Divida em funções menores",
                    severidade=3
                ))

            # Sem docstring
            if not func["tem_docstring"] and not func["nome"].startswith("_"):
                problemas.append(Problema(
                    tipo="sugestao", categoria="documentacao", linha=func["linha_inicio"],
                    mensagem=f"Função '{func['nome']}' sem docstring",
                    sugestao='Adicione: """O que essa função faz, parâmetros, o que retorna."""',
                    severidade=1
                ))

        # Imports duplicados
        modulos_vistos = {}
        for imp in imports:
            mod = imp["modulo"]
            if mod in modulos_vistos:
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=imp["linha"],
                    mensagem=f"Import duplicado: '{mod}' já foi importado na linha {modulos_vistos[mod]}",
                    sugestao="Remova o import duplicado",
                    severidade=2
                ))
            else:
                modulos_vistos[mod] = imp["linha"]

        return problemas


class AnalisadorJS:
    """Análises específicas para JavaScript/TypeScript."""

    def extrair_funcoes(self, linhas: list[str]) -> list[dict]:
        funcoes = []
        padroes = [
            r'(?:function\s+(\w+)\s*\(([^)]*)\))',               # function nome()
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>',  # const x = () =>
            r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?function',          # const x = function
            r'(\w+)\s*\(([^)]*)\)\s*\{',                          # método em classe
        ]
        for i, linha in enumerate(linhas, 1):
            for padrao in padroes:
                match = re.search(padrao, linha)
                if match:
                    nome = match.group(1) if match.lastindex >= 1 else "anonima"
                    params_str = match.group(2) if match.lastindex >= 2 else ""
                    params = [p.strip() for p in params_str.split(",") if p.strip()]
                    funcoes.append({
                        "nome": nome, "linha_inicio": i,
                        "params": params, "num_params": len(params),
                        "linhas": 0
                    })
                    break
        return funcoes

    def detectar_problemas(self, linhas: list[str]) -> list[Problema]:
        problemas = []
        for i, linha in enumerate(linhas, 1):
            # var em vez de const/let
            if re.match(r'^\s*var\s+', linha):
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=i,
                    mensagem="'var' tem escopo de função e causa bugs sutis",
                    sugestao="Use 'const' (padrão) ou 'let' (se precisar reatribuir)",
                    severidade=2
                ))

            # == em vez de ===
            if re.search(r'[^=!<>]==[^=]', linha) and "===" not in linha:
                problemas.append(Problema(
                    tipo="aviso", categoria="qualidade", linha=i,
                    mensagem="'==' faz coerção de tipo e causa bugs — use '==='",
                    sugestao="Substitua '==' por '===' para comparação estrita",
                    severidade=3
                ))

            # console.log esquecido
            if re.search(r'console\.(log|debug|warn)\s*\(', linha):
                problemas.append(Problema(
                    tipo="info", categoria="qualidade", linha=i,
                    mensagem="console.log encontrado — remover antes de produção",
                    sugestao="Use uma biblioteca de logging ou remova antes de deploy",
                    severidade=1
                ))

            # eval
            if re.search(r'\beval\s*\(', linha):
                problemas.append(Problema(
                    tipo="erro", categoria="seguranca", linha=i,
                    mensagem="eval() é perigoso — executa código arbitrário",
                    sugestao="Evite eval(). Quase sempre há uma alternativa mais segura",
                    severidade=5
                ))

            # Senha hardcoded
            if re.search(r'(password|senha|token|secret|apiKey)\s*[=:]\s*["\'][^"\']+["\']', linha, re.IGNORECASE):
                problemas.append(Problema(
                    tipo="erro", categoria="seguranca", linha=i,
                    mensagem="Credencial hardcoded! Vulnerabilidade grave",
                    sugestao="Use process.env.MINHA_SENHA ou variáveis de ambiente",
                    severidade=5
                ))

        return problemas


class AnalisadorGenerico:
    """
    Análises genéricas para linguagens sem analisador específico.
    Funciona bem o suficiente pra C, Java, Go, Rust, etc.
    """

    def extrair_funcoes(self, linhas: list[str], lang: str) -> list[dict]:
        funcoes = []

        # Padrões de declaração de função por linguagem
        padroes_por_lang = {
            "java":   [r'(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(([^)]*)\)'],
            "c":      [r'\b\w[\w\s\*]+\s+(\w+)\s*\(([^)]*)\)\s*\{'],
            "cpp":    [r'\b\w[\w\s\*:<>]+\s+(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*\{'],
            "go":     [r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)'],
            "rust":   [r'(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)'],
            "ruby":   [r'def\s+(\w+)\s*(?:\(([^)]*)\))?'],
            "php":    [r'function\s+(\w+)\s*\(([^)]*)\)'],
            "swift":  [r'func\s+(\w+)\s*\(([^)]*)\)'],
            "kotlin": [r'fun\s+(\w+)\s*\(([^)]*)\)'],
            "dart":   [r'(?:\w+\s+)?(\w+)\s*\(([^)]*)\)\s*(?:async)?\s*\{'],
        }

        padroes = padroes_por_lang.get(lang, [r'(?:function|func|def|fn|sub)\s+(\w+)\s*\(([^)]*)\)'])

        for i, linha in enumerate(linhas, 1):
            for padrao in padroes:
                match = re.search(padrao, linha)
                if match:
                    nome = match.group(1)
                    params_str = match.group(2) if match.lastindex >= 2 else ""
                    params = [p.strip() for p in params_str.split(",") if p.strip()]
                    if nome not in ("if", "for", "while", "switch", "catch"):
                        funcoes.append({
                            "nome": nome, "linha_inicio": i,
                            "params": params, "num_params": len(params),
                            "linhas": 0
                        })
                    break

        return funcoes

    def detectar_problemas_gerais(self, linhas: list[str]) -> list[Problema]:
        problemas = []
        for i, linha in enumerate(linhas, 1):
            # Magic numbers (números soltos no código)
            if re.search(r'(?<![.\w])[2-9]\d{2,}(?![.\w])', linha):
                if not re.search(r'(//|#|--|/\*)', linha):  # ignora comentários
                    problemas.append(Problema(
                        tipo="sugestao", categoria="qualidade", linha=i,
                        mensagem="Número mágico no código — dificulta manutenção",
                        sugestao="Declare como constante com nome significativo: MAX_TENTATIVAS = 300",
                        severidade=2
                    ))

            # TODO/FIXME/HACK
            if re.search(r'\b(TODO|FIXME|HACK|XXX|BUG)\b', linha, re.IGNORECASE):
                tipo_nota = re.search(r'\b(TODO|FIXME|HACK|XXX|BUG)\b', linha, re.IGNORECASE).group(1).upper()
                problemas.append(Problema(
                    tipo="info", categoria="qualidade", linha=i,
                    mensagem=f"{tipo_nota} encontrado — item pendente no código",
                    sugestao="Resolva ou crie uma issue/ticket para rastrear",
                    severidade=1
                ))

            # Linha muito longa (universal)
            if len(linha.rstrip()) > 150:
                problemas.append(Problema(
                    tipo="sugestao", categoria="estilo", linha=i,
                    mensagem=f"Linha muito longa ({len(linha.rstrip())} chars)",
                    sugestao="Quebre em múltiplas linhas para melhor legibilidade",
                    severidade=1
                ))

            # Senha hardcoded (universal)
            if re.search(r'(password|senha|token|secret|api_key|apikey)\s*[=:]\s*["\'][^"\']{4,}["\']',
                         linha, re.IGNORECASE):
                problemas.append(Problema(
                    tipo="erro", categoria="seguranca", linha=i,
                    mensagem="Possível credencial hardcoded detectada",
                    sugestao="Use variáveis de ambiente ou um gerenciador de segredos",
                    severidade=5
                ))

        return problemas





# -------------------------------------------------------
# TESTE RÁPIDO
# -------------------------------------------------------
if __name__ == "__main__":
    import os

    print("⚛️  PARADOXO X — Testando Code Analyzer Universal\n")

    analyzer = CodeAnalyzer()

    # --- Teste 1: Python ---
    codigo_python = '''
import os
import os  # duplicado!

password = "123456"  # hardcoded!

def funcao_gigante(a, b, c, d, e, f, g):
    x = None
    if x == None:
        print("debug")
    try:
        result = eval(a)
    except:
        pass
    if True == True:
        return 42
    for i in range(999):
        for j in range(999):
            for k in range(999):
                pass

class MinhaClasse:
    def metodo(self):
        """Tem docstring."""
        pass
'''

    resultado = analyzer.analisar(codigo_python, "teste.py")
    print(analyzer.relatorio(resultado))

    # --- Teste 2: JavaScript ---
    codigo_js = '''
var nome = "João"
const senha = "abc123"

function verificar(x) {
    if (x == null) {
        console.log("nulo")
        eval("alert(1)")
    }
}
'''
    resultado_js = analyzer.analisar(codigo_js, "teste.js")
    print(analyzer.relatorio(resultado_js))

    # --- Teste 3: Detecção automática (Go) ---
    codigo_go = '''
package main

import "fmt"

func soma(a int, b int) int {
    return a + b
}

func main() {
    fmt.Println(soma(1, 2))
}
'''
    resultado_go = analyzer.analisar(codigo_go)
    print(f"Linguagem detectada: {resultado_go.metricas.linguagem}")
    print(analyzer.relatorio(resultado_go, verbose=False))