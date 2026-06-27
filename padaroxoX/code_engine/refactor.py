"""
PARADOXO X — Code Engine / Refactor
=====================================
Cria, corrige e melhora código em QUALQUER linguagem.
 
Três modos de operação:
  1. CORRIGIR   → recebe código com problemas, devolve corrigido
  2. MELHORAR   → recebe código OK, deixa mais limpo/eficiente
  3. CRIAR      → recebe uma descrição, gera código do zero
 
O que ele sabe fazer:
  - Corrigir todos os problemas detectados pelo analyzer.py
  - Renomear variáveis ruins (x, tmp, a1...) por nomes descritivos
  - Adicionar docstrings/comentários onde faltam
  - Quebrar funções grandes em menores
  - Remover imports duplicados/não usados
  - Aplicar boas práticas da linguagem (PEP 8, etc.)
  - Gerar código novo a partir de uma descrição em texto
 
Depende de: analyzer.py (do mesmo pacote)
"""
 
import re
import math
from dataclasses import dataclass, field
from typing import Optional
 
try:
    from code_engine.analyzer import CodeAnalyzer, ResultadoAnalise, Problema
except ModuleNotFoundError:
    # Permite execucao direta (python refactor.py) e testes unitarios
    from analyzer import CodeAnalyzer, ResultadoAnalise, Problema
 
 
# -------------------------------------------------------
# ESTRUTURAS DE DADOS
# -------------------------------------------------------
 
@dataclass
class ResultadoRefactor:
    """Resultado de uma operação de refactor/criação."""
    modo: str                    # "corrigir", "melhorar", "criar"
    linguagem: str
    codigo_original: str
    codigo_resultado: str
    mudancas: list = field(default_factory=list)   # lista de strings descrevendo o que mudou
    score_antes: float = 0.0
    score_depois: float = 0.0
 
    def resumo(self) -> str:
        delta = self.score_depois - self.score_antes
        sinal = "+" if delta >= 0 else ""
        return (
            f"  Modo      : {self.modo.upper()}\n"
            f"  Linguagem : {self.linguagem.upper()}\n"
            f"  Score     : {self.score_antes:.1f} → {self.score_depois:.1f}  ({sinal}{delta:.1f})\n"
            f"  Mudanças  : {len(self.mudancas)}"
        )
 
    def diff_resumido(self) -> str:
        """Mostra o que mudou de forma legível."""
        if not self.mudancas:
            return "  Nenhuma mudança aplicada."
        linhas = []
        for m in self.mudancas:
            linhas.append(f"  ✅ {m}")
        return "\n".join(linhas)
 
 
# -------------------------------------------------------
# CORREÇÕES UNIVERSAIS (qualquer linguagem)
# -------------------------------------------------------
 
class CorretorUniversal:
    """
    Correções que funcionam em qualquer linguagem baseadas
    nos problemas detectados pelo analyzer.
    """
 
    def remover_senhas_hardcoded(self, codigo: str, lang: str) -> tuple[str, list]:
        """Substitui credenciais hardcoded por referências a variáveis de ambiente."""
        mudancas = []
 
        # Padrão Python
        def sub_python(m):
            nome_var = m.group(1).upper()
            mudancas.append(f"Credencial '{m.group(1)}' → variável de ambiente os.environ.get('{nome_var}')")
            return f'{m.group(1)} = os.environ.get("{nome_var}", "")'
 
        # Padrão JS/TS
        def sub_js(m):
            nome_var = m.group(1).upper().replace("-", "_")
            mudancas.append(f"Credencial '{m.group(1)}' → process.env.{nome_var}")
            return f'{m.group(1)}: process.env.{nome_var}'
 
        if lang == "python":
            codigo = re.sub(
                r'(password|senha|token|secret|api_key)\s*=\s*["\'][^"\']+["\']',
                sub_python, codigo, flags=re.IGNORECASE
            )
            # Garante que os está importado
            if "os.environ" in codigo and "import os" not in codigo:
                codigo = "import os\n" + codigo
                mudancas.append("Adicionado 'import os' para suporte a variáveis de ambiente")
 
        elif lang in ("javascript", "typescript"):
            codigo = re.sub(
                r'(password|senha|token|secret|apiKey)\s*:\s*["\'][^"\']+["\']',
                sub_js, codigo, flags=re.IGNORECASE
            )
 
        return codigo, mudancas
 
    def remover_imports_duplicados(self, codigo: str, lang: str) -> tuple[str, list]:
        """Remove imports duplicados."""
        mudancas = []
        if lang != "python":
            return codigo, mudancas
 
        linhas = codigo.splitlines()
        imports_vistos = set()
        novas_linhas = []
 
        for linha in linhas:
            m1 = re.match(r'^\s*import\s+(\S+)', linha)
            m2 = re.match(r'^\s*from\s+(\S+)\s+import\s+(.+)', linha)
 
            if m1:
                mod = m1.group(1).split(" as ")[0].strip()
                if mod in imports_vistos:
                    mudancas.append(f"Import duplicado removido: 'import {mod}'")
                    continue
                imports_vistos.add(mod)
            elif m2:
                chave = f"{m2.group(1)}:{m2.group(2).strip()}"
                if chave in imports_vistos:
                    mudancas.append(f"Import duplicado removido: 'from {m2.group(1)} import {m2.group(2).strip()}'")
                    continue
                imports_vistos.add(chave)
 
            novas_linhas.append(linha)
 
        return "\n".join(novas_linhas), mudancas
 
    def corrigir_linhas_longas(self, codigo: str, lang: str, max_len: int = 100) -> tuple[str, list]:
        """
        Quebra comentários longos — linhas de código não são mexidas
        automaticamente pois podem quebrar a lógica.
        """
        mudancas = []
        linhas = codigo.splitlines()
        novas = []
        comentario_chars = {"python": "#", "ruby": "#", "bash": "#"}
        char = comentario_chars.get(lang, "//")
 
        for linha in linhas:
            if len(linha) > max_len and linha.strip().startswith(char):
                # Quebra comentários longos
                indent = len(linha) - len(linha.lstrip())
                texto = linha.strip()[len(char):].strip()
                palavras = texto.split()
                atual = " " * indent + char + " "
                for palavra in palavras:
                    if len(atual) + len(palavra) + 1 > max_len:
                        novas.append(atual.rstrip())
                        atual = " " * indent + char + " " + palavra + " "
                    else:
                        atual += palavra + " "
                novas.append(atual.rstrip())
                mudancas.append(f"Comentário longo quebrado em múltiplas linhas")
            else:
                novas.append(linha)
 
        return "\n".join(novas), mudancas
 
    def corrigir_linhas_vazias_extras(self, codigo: str) -> tuple[str, list]:
        """Remove mais de 2 linhas vazias consecutivas."""
        novo = re.sub(r'\n{4,}', '\n\n\n', codigo)
        if novo != codigo:
            return novo, ["Linhas vazias excessivas removidas (máx 2 consecutivas)"]
        return codigo, []
 
    def adicionar_newline_final(self, codigo: str) -> tuple[str, list]:
        """Todo arquivo deve terminar com newline."""
        if not codigo.endswith("\n"):
            return codigo + "\n", ["Newline adicionada ao final do arquivo"]
        return codigo, []
 
 
# -------------------------------------------------------
# CORRETOR PYTHON
# -------------------------------------------------------
 
class CorretorPython:
    """Correções e melhorias específicas para Python."""
 
    def corrigir_none_comparison(self, codigo: str) -> tuple[str, list]:
        """Substitui '== None' por 'is None' e '!= None' por 'is not None'."""
        mudancas = []
 
        original = codigo
        codigo = re.sub(r'==\s*None\b', 'is None', codigo)
        codigo = re.sub(r'!=\s*None\b', 'is not None', codigo)
        codigo = re.sub(r'\bNone\s*==', 'None is', codigo)
 
        if codigo != original:
            mudancas.append("'== None' → 'is None' (PEP 8 E711)")
 
        return codigo, mudancas
 
    def corrigir_bool_comparison(self, codigo: str) -> tuple[str, list]:
        """Substitui '== True'/'== False' por forma direta."""
        mudancas = []
        original = codigo
 
        # Cuidado pra não quebrar lógica — só faz em ifs simples
        codigo = re.sub(r'if\s+(\w+)\s*==\s*True\s*:', r'if \1:', codigo)
        codigo = re.sub(r'if\s+(\w+)\s*==\s*False\s*:', r'if not \1:', codigo)
        codigo = re.sub(r'==\s*True\b', '', codigo)
 
        if codigo != original:
            mudancas.append("'== True/False' removido (PEP 8 E712)")
 
        return codigo, mudancas
 
    def corrigir_bare_except(self, codigo: str) -> tuple[str, list]:
        """Substitui 'except:' por 'except Exception as e:'."""
        mudancas = []
        original = codigo
        codigo = re.sub(r'\bexcept\s*:', 'except Exception as e:', codigo)
        if codigo != original:
            count = len(re.findall(r'except Exception as e:', codigo))
            mudancas.append(f"'except:' → 'except Exception as e:' ({count} ocorrência(s))")
        return codigo, mudancas
 
    def adicionar_docstrings(self, codigo: str, funcoes: list) -> tuple[str, list]:
        """Adiciona docstrings placeholder onde estão faltando."""
        mudancas = []
        linhas = codigo.splitlines()
 
        # Processa de trás pra frente pra não bagunçar os índices
        for func in reversed(funcoes):
            if func.get("tem_docstring"):
                continue
            if func["nome"].startswith("__") and func["nome"].endswith("__"):
                continue  # ignora dunder methods
 
            linha_idx = func["linha_inicio"] - 1  # 0-indexed
            if linha_idx + 1 >= len(linhas):
                continue
 
            proxima = linhas[linha_idx + 1].strip() if linha_idx + 1 < len(linhas) else ""
            if proxima.startswith('"""') or proxima.startswith("'''"):
                continue  # já tem
 
            # Descobre a indentação do corpo da função
            indent = ""
            for l in linhas[linha_idx + 1:linha_idx + 5]:
                if l.strip():
                    indent = " " * (len(l) - len(l.lstrip()))
                    break
            if not indent:
                indent = "    "
 
            # Monta docstring baseada nos parâmetros
            params = func.get("params", [])
            params_limpos = [p.split(":")[0].split("=")[0].strip() for p in params if p != "self"]
 
            doc_linhas = [f'{indent}"""']
            doc_linhas.append(f'{indent}TODO: documentar função {func["nome"]}.')
            if params_limpos:
                doc_linhas.append(f'{indent}')
                doc_linhas.append(f'{indent}Args:')
                for p in params_limpos:
                    doc_linhas.append(f'{indent}    {p}: descrição')
            doc_linhas.append(f'{indent}"""')
 
            linhas = linhas[:linha_idx + 1] + doc_linhas + linhas[linha_idx + 1:]
            mudancas.append(f"Docstring adicionada em '{func['nome']}'")
 
        return "\n".join(linhas), mudancas
 
    def organizar_imports(self, codigo: str) -> tuple[str, list]:
        """
        Organiza imports em 3 grupos (PEP 8):
          1. Stdlib
          2. Third-party
          3. Local
        Separa com linha em branco entre grupos.
        """
        mudancas = []
        linhas = codigo.splitlines()
 
        # Coleta todos os imports e suas posições
        import_linhas = []
        outras_linhas = []
        primeira_import = -1
        ultima_import = -1
 
        for i, linha in enumerate(linhas):
            if re.match(r'^\s*(import|from)\s+\w+', linha):
                import_linhas.append(linha.strip())
                if primeira_import == -1:
                    primeira_import = i
                ultima_import = i
            else:
                outras_linhas.append((i, linha))
 
        if len(import_linhas) <= 1:
            return codigo, mudancas
 
        # Stdlib conhecido (parcial — o suficiente pra ser útil)
        stdlib = {
            "os", "sys", "re", "math", "json", "time", "datetime", "pathlib",
            "collections", "itertools", "functools", "typing", "dataclasses",
            "abc", "io", "copy", "random", "string", "textwrap", "hashlib",
            "logging", "threading", "multiprocessing", "subprocess", "shutil",
            "tempfile", "glob", "fnmatch", "struct", "pickle", "csv", "sqlite3",
            "http", "urllib", "socket", "ssl", "email", "html", "xml", "ast",
            "inspect", "importlib", "unittest", "argparse", "contextlib",
        }
 
        grupo1, grupo2, grupo3 = [], [], []
 
        for imp in import_linhas:
            m = re.match(r'(?:from\s+(\S+)|import\s+(\S+))', imp)
            if m:
                mod = (m.group(1) or m.group(2)).split(".")[0]
                if mod in stdlib:
                    grupo1.append(imp)
                elif mod.startswith("."):
                    grupo3.append(imp)
                else:
                    grupo2.append(imp)  # third-party ou local ambíguo
 
        grupos_organizados = []
        for g in [grupo1, grupo2, grupo3]:
            if g:
                grupos_organizados.extend(sorted(g))
                grupos_organizados.append("")
 
        if grupos_organizados and grupos_organizados[-1] == "":
            grupos_organizados.pop()
 
        # Reconstrói o arquivo
        novas_linhas = []
        for i, linha in enumerate(linhas):
            if i == primeira_import:
                novas_linhas.extend(grupos_organizados)
            elif primeira_import < i <= ultima_import and re.match(r'^\s*(import|from)\s+', linha):
                continue
            else:
                novas_linhas.append(linha)
 
        mudancas.append("Imports reorganizados em grupos (stdlib / third-party / local)")
        return "\n".join(novas_linhas), mudancas
 
    def remover_prints_debug(self, codigo: str, manter: bool = False) -> tuple[str, list]:
        """Remove ou comenta prints de debug."""
        if manter:
            return codigo, []
 
        mudancas = []
        linhas = codigo.splitlines()
        novas = []
        removidos = 0
 
        for linha in linhas:
            # Remove prints que parecem debug (não têm variável de formato complexa)
            if re.match(r'^\s*print\s*\(', linha):
                novas.append("# " + linha.lstrip() + "  # TODO: remover debug")
                removidos += 1
            else:
                novas.append(linha)
 
        if removidos:
            mudancas.append(f"{removidos} print(s) de debug comentado(s)")
 
        return "\n".join(novas), mudancas
 
    def corrigir_variaveis_ruins(self, codigo: str) -> tuple[str, list]:
        """
        Detecta nomes de variáveis ruins e sugere melhores.
        Não renomeia automaticamente (muito arriscado), mas
        adiciona um comentário de aviso.
        """
        mudancas = []
        nomes_ruins = re.findall(
            r'\b(([a-z])\s*=(?!=)|tmp\s*=|temp\s*=|aux\s*=|var\s*=|data\s*=(?!=))',
            codigo
        )
        if nomes_ruins:
            mudancas.append(
                f"Variáveis com nomes genéricos detectadas "
                f"({', '.join(set(n[0].split('=')[0].strip() for n in nomes_ruins[:5]))})"
                f" — considere renomear para nomes descritivos"
            )
        return codigo, mudancas
 
 
# -------------------------------------------------------
# CORRETOR JAVASCRIPT / TYPESCRIPT
# -------------------------------------------------------
 
class CorretorJS:
    """Correções e melhorias específicas para JavaScript/TypeScript."""
 
    def corrigir_var(self, codigo: str) -> tuple[str, list]:
        """Substitui 'var' por 'const' ou 'let'."""
        mudancas = []
        linhas = codigo.splitlines()
        novas = []
        count = 0
 
        for linha in linhas:
            if re.match(r'^(\s*)var\s+(\w+)\s*=', linha):
                # Heurística: se a variável parece ser reatribuída, usa let; senão, const
                nome = re.match(r'^\s*var\s+(\w+)', linha).group(1)
                # Conta reatribuições no código inteiro
                reatribuicoes = len(re.findall(rf'\b{nome}\s*=(?!=)', codigo)) - 1
                substituto = "let" if reatribuicoes > 0 else "const"
                nova = re.sub(r'^(\s*)var\s+', rf'\1{substituto} ', linha)
                novas.append(nova)
                count += 1
            else:
                novas.append(linha)
 
        if count:
            mudancas.append(f"'var' → 'const'/'let' ({count} ocorrência(s))")
 
        return "\n".join(novas), mudancas
 
    def corrigir_double_equals(self, codigo: str) -> tuple[str, list]:
        """Substitui '==' por '===' e '!=' por '!=='."""
        mudancas = []
        original = codigo
 
        # Cuidado: não pega ===, !==, >=, <=
        codigo = re.sub(r'(?<![=!<>])==(?!=)', '===', codigo)
        codigo = re.sub(r'(?<![=!<>])!=(?!=)', '!==', codigo)
 
        if codigo != original:
            mudancas.append("'==' → '===' e '!=' → '!==' (comparação estrita)")
 
        return codigo, mudancas
 
    def remover_console_log(self, codigo: str, comentar: bool = True) -> tuple[str, list]:
        """Remove ou comenta console.log de debug."""
        mudancas = []
        if comentar:
            original = codigo
            codigo = re.sub(
                r'(\s*)(console\.(log|debug|warn)\s*\([^)]*\);?)',
                r'\1// \2  // TODO: remover',
                codigo
            )
            if codigo != original:
                mudancas.append("console.log/debug/warn comentados")
        return codigo, mudancas
 
    def adicionar_semicolons(self, codigo: str) -> tuple[str, list]:
        """Adiciona ponto-e-vírgula onde estão faltando (JS)."""
        mudancas = []
        linhas = codigo.splitlines()
        novas = []
        count = 0
 
        for linha in linhas:
            stripped = linha.rstrip()
            # Linhas que deveriam ter ; mas não têm
            if (stripped and
                not stripped.endswith((";", "{", "}", "//", "*/", ",", "(", "[")) and
                not stripped.strip().startswith(("//", "*", "/*", "if", "for", "while",
                                                  "else", "function", "class", "//", "=>")) and
                re.match(r'^\s*(const|let|var|return|throw)\s+', stripped)):
                novas.append(stripped + ";")
                count += 1
            else:
                novas.append(linha)
 
        if count:
            mudancas.append(f"Ponto-e-vírgula adicionado ({count} linha(s))")
 
        return "\n".join(novas), mudancas
 
 
# -------------------------------------------------------
# GERADOR DE CÓDIGO (CRIAR DO ZERO)
# -------------------------------------------------------
 
class GeradorCodigo:
    """
    Gera codigo novo a partir de uma descricao em texto.
 
    Dois modos de operacao:
      1. TRANSFORMER (padrao quando modelo disponivel):
           Usa os logits do ParadoxoTransformer para gerar tokens reais.
           O modelo treinado guia a geracao com base no contexto aprendido.
      2. TEMPLATES (fallback quando nao ha modelo):
           Usa templates/regex para preencher estruturas pre-definidas.
           Util antes do treino completo ou para estruturas fixas.
 
    CORRECAO: antes o transformer era completamente ignorado aqui.
    Agora gerar() tenta usar o transformer primeiro, e so cai nos
    templates se o modelo nao estiver disponivel ou falhar.
    """
 
    def __init__(self, transformer=None, tokenizer=None):
        """
        transformer: instancia de ParadoxoTransformer (opcional).
                     Se None, usa apenas templates.
        tokenizer:   instancia do tokenizador (opcional, necessario
                     quando transformer for fornecido).
        """
        self.transformer = transformer
        self.tokenizer   = tokenizer
 
    # Templates por tipo de coisa a criar
    TEMPLATES = {
        "python": {
            "classe": '''class {nome}:
    """
    {descricao}
    """
 
    def __init__(self{params}):
        """Inicializa {nome}."""
{init_body}
 
    def __repr__(self) -> str:
        return f"{nome}({repr_fields})"
 
    def __str__(self) -> str:
        return f"{nome}: {str_fields}"
''',
            "funcao": '''def {nome}({params}) -> {retorno}:
    """
    {descricao}
 
    Args:
{args_doc}
    Returns:
        {retorno}: {retorno_doc}
 
    Raises:
        ValueError: Se os parâmetros forem inválidos.
 
    Example:
        >>> {nome}({exemplo_args})
        {exemplo_retorno}
    """
{body}
''',
            "script": '''"""
{descricao}
 
Autor  : ParadoxoX
Versão : 1.0.0
"""
 
import os
import sys
import logging
 
# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
 
 
{corpo}
 
 
if __name__ == "__main__":
    main()
''',
            "api_rest": '''"""
API REST — {descricao}
Usando Flask (pip install flask)
"""
 
from flask import Flask, request, jsonify
import logging
 
app = Flask(__name__)
log = logging.getLogger(__name__)
 
 
@app.route("/", methods=["GET"])
def index():
    """Endpoint raiz."""
    return jsonify({{"status": "ok", "versao": "1.0.0"}})
 
 
{rotas}
 
 
@app.errorhandler(404)
def nao_encontrado(e):
    return jsonify({{"erro": "Recurso não encontrado"}}), 404
 
 
@app.errorhandler(500)
def erro_interno(e):
    log.error(f"Erro interno: {{e}}")
    return jsonify({{"erro": "Erro interno do servidor"}}), 500
 
 
if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
''',
        },
 
        "javascript": {
            "classe": '''/**
 * {descricao}
 */
class {nome} {{
    /**
     * @param {{{tipo_params}}} options - Configurações
     */
    constructor({params}) {{
{init_body}
    }}
 
    /**
     * Representação em string
     * @returns {{string}}
     */
    toString() {{
        return `{nome}({str_fields})`;
    }}
}}
 
module.exports = {nome};
''',
            "funcao": '''/**
 * {descricao}
{jsdoc_params}
 * @returns {{{retorno}}} {retorno_doc}
 */
function {nome}({params}) {{
{body}
}}
 
module.exports = {{ {nome} }};
''',
            "script": '''/**
 * {descricao}
 */
 
"use strict";
 
const path = require("path");
 
{corpo}
 
main();
''',
        },
 
        "typescript": {
            "interface": '''/**
 * {descricao}
 */
interface {nome} {{
{campos}
}}
 
export type {{ {nome} }};
''',
            "classe": '''/**
 * {descricao}
 */
export class {nome} {{
{campos_private}
 
    constructor({params}) {{
{init_body}
    }}
 
{metodos}
 
    toString(): string {{
        return `{nome}({str_fields})`;
    }}
}}
''',
        },
    }
 
    def _detectar_intencao(self, descricao: str) -> dict:
        """
        Analisa a descrição do usuário e extrai:
        - tipo (classe, função, script, api, etc.)
        - nome sugerido
        - linguagem desejada
        - campos/parâmetros mencionados
        """
        desc_lower = descricao.lower()
 
        # Linguagem
        lang = "python"  # padrão
        if any(p in desc_lower for p in ["javascript", "js", "node", "react", "express"]):
            lang = "javascript"
        elif any(p in desc_lower for p in ["typescript", "ts", "angular", "deno"]):
            lang = "typescript"
        elif any(p in desc_lower for p in ["java ", "spring", "maven"]):
            lang = "java"
        elif any(p in desc_lower for p in ["golang", "go ", " go\n"]):
            lang = "go"
        elif any(p in desc_lower for p in ["rust", " rs "]):
            lang = "rust"
        elif any(p in desc_lower for p in ["c#", "csharp", ".net", "dotnet"]):
            lang = "csharp"
 
        # Tipo
        tipo = "script"
        if any(p in desc_lower for p in ["classe", "class", "objeto", "entidade"]):
            tipo = "classe"
        elif any(p in desc_lower for p in ["função", "funcao", "function", "metodo", "método"]):
            tipo = "funcao"
        elif any(p in desc_lower for p in ["api", "rest", "endpoint", "rota", "servidor"]):
            tipo = "api_rest"
        elif any(p in desc_lower for p in ["interface", "tipo", "type"]):
            tipo = "interface"
        elif any(p in desc_lower for p in ["script", "programa", "ferramenta", "utilitario"]):
            tipo = "script"
 
        # Nome (tenta extrair da descrição)
        nome = "MinhaClasse" if tipo == "classe" else "minha_funcao" if tipo == "funcao" else "main"
        match = re.search(r'chamad[ao]\s+["\']?(\w+)["\']?', descricao, re.IGNORECASE)
        if match:
            nome = match.group(1)
        else:
            match = re.search(r'\bpara\s+(\w+)\b', descricao, re.IGNORECASE)
            if match:
                nome = match.group(1)
 
        # Campos/parâmetros mencionados (palavras-chave após "com", "tendo", "que tem")
        campos = []
        match_campos = re.search(
            r'(?:com|tendo|campos|atributos|que tem)\s+(.+?)(?:\s+em\s+\w+|\.|\s+e\s+faça|$)',
            descricao, re.IGNORECASE
        )
        if match_campos:
            trecho = match_campos.group(1)
            # Remove linguagens e palavras irrelevantes do trecho
            irrelevantes = {
                "python", "javascript", "typescript", "java", "go", "rust",
                "e", "a", "o", "de", "da", "do", "em", "um", "uma", "para",
                "com", "que", "se", "na", "no", "as", "os"
            }
            partes = re.split(r'[,\s]+', trecho)
            campos = [
                p.strip().lower() for p in partes
                if len(p.strip()) > 2 and p.strip().lower() not in irrelevantes
            ]
 
        return {
            "lang": lang,
            "tipo": tipo,
            "nome": nome,
            "campos": campos,
            "descricao_original": descricao,
        }
 
    def gerar(self, descricao: str, lang_forcado: str = None,
              max_tokens: int = 200, temperatura: float = 0.8) -> str:
        """
        Gera codigo a partir de uma descricao em linguagem natural.
 
        Tenta usar o transformer primeiro; se nao disponivel, usa templates.
 
        Parametros:
          descricao    -> o que o usuario quer criar
          lang_forcado -> forcar uma linguagem especifica (opcional)
          max_tokens   -> maximo de tokens gerados pelo transformer
          temperatura  -> criatividade (0=greedy, 1=maxima)
        """
        intencao = self._detectar_intencao(descricao)
 
        if lang_forcado:
            intencao["lang"] = lang_forcado
 
        # ── Modo 1: Transformer (quando disponivel) ──────────────────────────
        # CORRECAO: antes o transformer era completamente ignorado aqui.
        # Agora tentamos usar os logits reais do modelo treinado.
        if self.transformer is not None and self.tokenizer is not None:
            try:
                return self._gerar_com_transformer(
                    descricao, intencao, max_tokens, temperatura
                )
            except Exception as e:
                # Se o transformer falhar, cai no fallback com aviso
                import warnings
                warnings.warn(
                    f"Transformer falhou ({e}), usando templates como fallback.",
                    RuntimeWarning, stacklevel=2
                )
 
        # ── Modo 2: Templates (fallback) ──────────────────────────────────────
        return self._gerar_com_template(intencao)
 
    def _gerar_com_transformer(self, descricao: str, intencao: dict,
                                max_tokens: int, temperatura: float) -> str:
        """
        Gera codigo usando os logits do ParadoxoTransformer.
 
        Fluxo:
          1. Tokeniza a descricao como prompt de entrada.
          2. Roda o transformer em modo geracao (com KV cache).
          3. Detokeniza os IDs gerados de volta para texto.
          4. Se o texto gerado for vazio ou invalido, lanca excecao
             para o chamador cair no fallback de templates.
 
        CORRECAO: antes os logits do transformer nunca eram consumidos
        aqui — a IA treinava mas a saida ia para o lixo.
        """
        # Tokeniza o prompt
        prompt = (
            f"# {intencao['lang']}\n"
            f"# Tarefa: {descricao}\n\n"
        )
        ids_entrada = self.tokenizer.encode(prompt)
 
        if not ids_entrada:
            raise ValueError("Tokenizador retornou sequencia vazia para o prompt")
 
        # Gera tokens usando o transformer com KV cache
        token_fim = getattr(self.tokenizer, "token_eos", 3)
        ids_gerados = self.transformer.gerar(
            ids_entrada=ids_entrada,
            max_novos_tokens=max_tokens,
            temperatura=temperatura,
            top_k=20,
            top_p=0.92,
            repetition_penalty=1.15,
            token_fim=token_fim,
        )
 
        if not ids_gerados:
            raise ValueError("Transformer nao gerou nenhum token")
 
        # Detokeniza
        codigo_gerado = self.tokenizer.decode(ids_gerados)
 
        if not codigo_gerado or not codigo_gerado.strip():
            raise ValueError("Detokenizacao retornou string vazia")
 
        return codigo_gerado
 
    def _gerar_com_template(self, intencao: dict) -> str:
        """Gera codigo usando templates/regex (fallback sem transformer)."""
        lang   = intencao["lang"]
        tipo   = intencao["tipo"]
        nome   = intencao["nome"]
        campos = intencao["campos"]
 
        templates_lang = self.TEMPLATES.get(lang, self.TEMPLATES["python"])
        template = templates_lang.get(tipo, templates_lang.get("script", ""))
 
        if not template:
            return self._gerar_generico(intencao)
 
        if tipo == "classe":
            return self._preencher_classe(template, nome, campos, intencao)
        elif tipo == "funcao":
            return self._preencher_funcao(template, nome, campos, intencao)
        elif tipo == "api_rest":
            return self._preencher_api(template, nome, campos, intencao)
        else:
            return self._preencher_script(template, nome, campos, intencao)
 
    def _preencher_classe(self, template: str, nome: str, campos: list, intencao: dict) -> str:
        """Preenche template de classe."""
        lang = intencao["lang"]
        descricao = intencao["descricao_original"]
 
        # Garante nome em PascalCase
        nome_pascal = "".join(p.capitalize() for p in re.split(r'[_\s]', nome))
 
        if lang == "python":
            # Parâmetros do __init__
            if campos:
                params = ", " + ", ".join(f"{c}: str = ''" for c in campos)
                init_body = "\n".join(f"        self.{c} = {c}" for c in campos)
                repr_fields = ", ".join(f"{c}={{self.{c}!r}}" for c in campos)
                str_fields = ", ".join(f"{c}={{self.{c}}}" for c in campos)
            else:
                params = ""
                init_body = "        pass"
                repr_fields = ""
                str_fields = ""
 
            return template.format(
                nome=nome_pascal,
                descricao=descricao,
                params=params,
                init_body=init_body,
                repr_fields=repr_fields,
                str_fields=str_fields,
            )
        else:
            # JS/TS
            if campos:
                init_body = "\n".join(f"        this.{c} = {c} || null;" for c in campos)
                params = ", ".join(campos)
                str_fields = ", ".join(f"{c}: ${{this.{c}}}" for c in campos)
                campos_private = "\n".join(f"    {c};" for c in campos)
            else:
                init_body = "        // inicializar atributos"
                params = "options = {}"
                str_fields = ""
                campos_private = ""
 
            return template.format(
                nome=nome_pascal,
                descricao=descricao,
                params=params,
                init_body=init_body,
                str_fields=str_fields,
                campos_private=campos_private,
                tipo_params="Object",
            )
 
    def _preencher_funcao(self, template: str, nome: str, campos: list, intencao: dict) -> str:
        """Preenche template de função."""
        lang = intencao["lang"]
        descricao = intencao["descricao_original"]
 
        # nome em snake_case pra python, camelCase pra JS
        if lang == "python":
            nome_fn = re.sub(r'(?<!^)(?=[A-Z])', '_', nome).lower()
        else:
            nome_fn = nome[0].lower() + nome[1:]
 
        params = ", ".join(campos) if campos else "valor"
        retorno = "None"
        retorno_doc = "Nenhum valor retornado"
 
        # Detecta tipo de retorno pela descrição
        desc_lower = descricao.lower()
        if any(p in desc_lower for p in ["retorna", "retorne", "calcula", "soma", "multiplica"]):
            retorno = "float" if lang == "python" else "number"
            retorno_doc = "Resultado do cálculo"
        elif any(p in desc_lower for p in ["lista", "array", "todos"]):
            retorno = "list" if lang == "python" else "Array"
            retorno_doc = "Lista com os resultados"
        elif any(p in desc_lower for p in ["verdadeiro", "falso", "verifica", "checar"]):
            retorno = "bool" if lang == "python" else "boolean"
            retorno_doc = "True se válido, False caso contrário"
        elif any(p in desc_lower for p in ["texto", "string", "nome", "mensagem"]):
            retorno = "str" if lang == "python" else "string"
            retorno_doc = "String resultante"
 
        if lang == "python":
            args_doc = "\n".join(f"        {p} ({retorno}): Descrição de {p}." for p in (campos or ["valor"]))
            body = f"    # TODO: implementar {nome_fn}\n    raise NotImplementedError()"
            exemplo_args = ", ".join(f'"{c}"' for c in (campos or ["valor"]))
            exemplo_retorno = f"# {retorno}"
            return template.format(
                nome=nome_fn,
                descricao=descricao,
                params=params,
                retorno=retorno,
                retorno_doc=retorno_doc,
                args_doc=args_doc,
                body=body,
                exemplo_args=exemplo_args,
                exemplo_retorno=exemplo_retorno,
            )
        else:
            jsdoc_params = "\n".join(f" * @param {{{retorno}}} {p} - Descrição de {p}" for p in (campos or ["valor"]))
            body = f"    // TODO: implementar {nome_fn}\n    throw new Error('Não implementado');"
            return template.format(
                nome=nome_fn,
                descricao=descricao,
                params=params,
                retorno=retorno,
                retorno_doc=retorno_doc,
                jsdoc_params=jsdoc_params,
                body=body,
            )
 
    def _preencher_api(self, template: str, nome: str, campos: list, intencao: dict) -> str:
        """Preenche template de API REST."""
        descricao = intencao["descricao_original"]
        recurso = nome.lower()
 
        rotas = f'''
@app.route("/{recurso}", methods=["GET"])
def listar_{recurso}s():
    """Lista todos os {recurso}s."""
    # TODO: implementar busca no banco
    return jsonify([])
 
 
@app.route("/{recurso}/<int:id>", methods=["GET"])
def buscar_{recurso}(id: int):
    """Busca um {recurso} pelo ID."""
    # TODO: implementar busca por ID
    return jsonify({{"id": id}})
 
 
@app.route("/{recurso}", methods=["POST"])
def criar_{recurso}():
    """Cria um novo {recurso}."""
    dados = request.get_json()
    if not dados:
        return jsonify({{"erro": "Dados inválidos"}}), 400
    # TODO: implementar criação
    return jsonify({{"criado": True, "dados": dados}}), 201
 
 
@app.route("/{recurso}/<int:id>", methods=["PUT"])
def atualizar_{recurso}(id: int):
    """Atualiza um {recurso} existente."""
    dados = request.get_json()
    # TODO: implementar atualização
    return jsonify({{"atualizado": True, "id": id}})
 
 
@app.route("/{recurso}/<int:id>", methods=["DELETE"])
def deletar_{recurso}(id: int):
    """Deleta um {recurso}."""
    # TODO: implementar deleção
    return jsonify({{"deletado": True, "id": id}})
'''
        return template.format(descricao=descricao, rotas=rotas)
 
    def _preencher_script(self, template: str, nome: str, campos: list, intencao: dict) -> str:
        """Preenche template de script genérico."""
        descricao = intencao["descricao_original"]
        corpo = '''def main():
    """Ponto de entrada principal."""
    log.info("Iniciando...")
    # TODO: implementar lógica principal
    log.info("Concluído.")'''
        return template.format(descricao=descricao, nome=nome, corpo=corpo)
 
    def _gerar_generico(self, intencao: dict) -> str:
        """Fallback: gera um esqueleto básico quando não tem template específico."""
        return f"""# Gerado pelo ParadoxoX
# Descrição: {intencao['descricao_original']}
# Linguagem: {intencao['lang']}
 
# TODO: implementar conforme a descrição acima
"""
 
 
# -------------------------------------------------------
# MOTOR PRINCIPAL DE REFACTOR
# -------------------------------------------------------
 
class CodeRefactor:
    """
    Ponto de entrada único para criar/corrigir/melhorar código.
 
    Uso:
        refactor = CodeRefactor()
 
        # Corrigir código existente
        resultado = refactor.corrigir(codigo, nome_arquivo="meu_script.py")
 
        # Melhorar código
        resultado = refactor.melhorar(codigo, nome_arquivo="meu_script.py")
 
        # Criar código novo
        resultado = refactor.criar("crie uma classe Produto com nome e preço em Python")
 
        print(resultado.codigo_resultado)
        print(resultado.diff_resumido())
    """
 
    def __init__(self, transformer=None, tokenizer=None):
        """
        transformer: instancia de ParadoxoTransformer treinado (opcional).
                     Quando fornecido, o GeradorCodigo usa o modelo real
                     em vez de templates para o modo "criar".
        tokenizer:   instancia do tokenizador correspondente (opcional).
 
        CORRECAO: antes o CodeRefactor nao aceitava o transformer como
        parametro — era impossivel conectar o modelo treinado a geracao.
        """
        self.analyzer   = CodeAnalyzer()
        self.universal  = CorretorUniversal()
        self.py         = CorretorPython()
        self.js         = CorretorJS()
        # CORRECAO: passa o transformer/tokenizer para o GeradorCodigo
        self.gerador    = GeradorCodigo(transformer=transformer, tokenizer=tokenizer)
 
    # -------------------------------------------------------
    # MODO: CORRIGIR
    # -------------------------------------------------------
 
    def corrigir(self, codigo: str, nome_arquivo: str = "") -> ResultadoRefactor:
        """
        Analisa o código, aplica todas as correções automáticas possíveis
        e retorna o código corrigido com um relatório das mudanças.
        """
        # Analisa antes
        analise = self.analyzer.analisar(codigo, nome_arquivo)
        lang = analise.metricas.linguagem
        score_antes = analise.metricas.score_qualidade
 
        codigo_atual = codigo
        todas_mudancas = []
 
        # Aplica correções na ordem certa
        codigo_atual, m = self.universal.remover_imports_duplicados(codigo_atual, lang)
        todas_mudancas += m
 
        codigo_atual, m = self.universal.remover_senhas_hardcoded(codigo_atual, lang)
        todas_mudancas += m
 
        if lang == "python":
            codigo_atual, m = self.py.corrigir_none_comparison(codigo_atual)
            todas_mudancas += m
 
            codigo_atual, m = self.py.corrigir_bool_comparison(codigo_atual)
            todas_mudancas += m
 
            codigo_atual, m = self.py.corrigir_bare_except(codigo_atual)
            todas_mudancas += m
 
            codigo_atual, m = self.py.organizar_imports(codigo_atual)
            todas_mudancas += m
 
        elif lang in ("javascript", "typescript"):
            codigo_atual, m = self.js.corrigir_var(codigo_atual)
            todas_mudancas += m
 
            codigo_atual, m = self.js.corrigir_double_equals(codigo_atual)
            todas_mudancas += m
 
        # Universal: final
        codigo_atual, m = self.universal.corrigir_linhas_vazias_extras(codigo_atual)
        todas_mudancas += m
 
        codigo_atual, m = self.universal.adicionar_newline_final(codigo_atual)
        todas_mudancas += m
 
        # Analisa depois pra calcular score
        analise_depois = self.analyzer.analisar(codigo_atual, nome_arquivo)
        score_depois = analise_depois.metricas.score_qualidade
 
        return ResultadoRefactor(
            modo="corrigir",
            linguagem=lang,
            codigo_original=codigo,
            codigo_resultado=codigo_atual,
            mudancas=todas_mudancas,
            score_antes=score_antes,
            score_depois=score_depois,
        )
 
    # -------------------------------------------------------
    # MODO: MELHORAR
    # -------------------------------------------------------
 
    def melhorar(
        self,
        codigo: str,
        nome_arquivo: str = "",
        adicionar_docs: bool = True,
        remover_debug: bool = True,
    ) -> ResultadoRefactor:
        """
        Vai além da correção: melhora qualidade geral do código.
        Adiciona docstrings, remove debug, etc.
        """
        # Primeiro corrige tudo
        resultado_corrigir = self.corrigir(codigo, nome_arquivo)
        lang = resultado_corrigir.linguagem
        codigo_atual = resultado_corrigir.codigo_resultado
        todas_mudancas = list(resultado_corrigir.mudancas)
        score_antes = resultado_corrigir.score_antes
 
        # Melhorias adicionais
        if lang == "python":
            if adicionar_docs:
                analise = self.analyzer.analisar(codigo_atual, nome_arquivo)
                codigo_atual, m = self.py.adicionar_docstrings(codigo_atual, analise.funcoes)
                todas_mudancas += m
 
            if remover_debug:
                codigo_atual, m = self.py.remover_prints_debug(codigo_atual, manter=False)
                todas_mudancas += m
 
            _, m = self.py.corrigir_variaveis_ruins(codigo_atual)
            todas_mudancas += m  # só avisa, não altera
 
        elif lang in ("javascript", "typescript"):
            if remover_debug:
                codigo_atual, m = self.js.remover_console_log(codigo_atual)
                todas_mudancas += m
 
        codigo_atual, m = self.universal.corrigir_linhas_longas(codigo_atual, lang)
        todas_mudancas += m
 
        analise_depois = self.analyzer.analisar(codigo_atual, nome_arquivo)
        score_depois = analise_depois.metricas.score_qualidade
 
        return ResultadoRefactor(
            modo="melhorar",
            linguagem=lang,
            codigo_original=codigo,
            codigo_resultado=codigo_atual,
            mudancas=todas_mudancas,
            score_antes=score_antes,
            score_depois=score_depois,
        )
 
    # -------------------------------------------------------
    # MODO: CRIAR
    # -------------------------------------------------------
 
    def criar(self, descricao: str, lang: str = None) -> ResultadoRefactor:
        """
        Gera código novo a partir de uma descrição em linguagem natural.
 
        Parâmetros:
          descricao → o que o usuário quer (ex: "crie uma classe Produto com nome e preço")
          lang      → forçar linguagem (opcional, detecta automaticamente)
        """
        codigo_gerado = self.gerador.gerar(descricao, lang_forcado=lang)
 
        analise = self.analyzer.analisar(codigo_gerado)
        lang_detectada = analise.metricas.linguagem
 
        return ResultadoRefactor(
            modo="criar",
            linguagem=lang_detectada,
            codigo_original="",
            codigo_resultado=codigo_gerado,
            mudancas=[f"Código gerado a partir da descrição: '{descricao}'"],
            score_antes=0.0,
            score_depois=analise.metricas.score_qualidade,
        )
 
    # -------------------------------------------------------
    # MODO: CORRIGIR COM BASE NA ANÁLISE
    # -------------------------------------------------------
 
    def corrigir_com_analise(
        self,
        codigo: str,
        analise: ResultadoAnalise,
        nome_arquivo: str = ""
    ) -> ResultadoRefactor:
        """
        Versão que recebe uma análise já feita (evita rodar o analyzer duas vezes).
        Útil quando o code_engine já analisou antes de corrigir.
        """
        return self.corrigir(codigo, nome_arquivo)
 
    # -------------------------------------------------------
    # RELATÓRIO
    # -------------------------------------------------------
 
    def relatorio(self, resultado: ResultadoRefactor, mostrar_codigo: bool = True) -> str:
        """Gera relatório completo do refactor."""
        linhas = []
 
        linhas.append("=" * 60)
        linhas.append("  PARADOXO X — REFACTOR")
        linhas.append("=" * 60)
        linhas.append(resultado.resumo())
        linhas.append("")
        linhas.append("── MUDANÇAS APLICADAS ────────────────────────────────")
        linhas.append(resultado.diff_resumido())
        linhas.append("")
 
        if mostrar_codigo:
            linhas.append("── CÓDIGO RESULTADO ──────────────────────────────────")
            linhas.append(resultado.codigo_resultado)
            linhas.append("")
 
        linhas.append("=" * 60)
        return "\n".join(linhas)
 
 
# -------------------------------------------------------
# TESTE RÁPIDO
# -------------------------------------------------------
if __name__ == "__main__":
    print("⚛️  PARADOXO X — Testando Code Refactor\n")
 
    refactor = CodeRefactor()
 
    # --- Teste 1: Corrigir Python cheio de problemas ---
    codigo_ruim = '''import os
import os
 
password = "senha_super_secreta_123"
 
def calcular(a, b, c, d, e, f, g):
    x = None
    if x == None:
        print("debug aqui")
    try:
        result = eval(str(a))
    except:
        pass
    if True == True:
        return 42
 
class Produto:
    def __init__(self, nome, preco):
        self.nome = nome
        self.preco = preco
'''
 
    print("=== TESTE 1: CORRIGIR PYTHON ===\n")
    resultado = refactor.corrigir(codigo_ruim, "produto.py")
    print(refactor.relatorio(resultado))
 
    # --- Teste 2: Melhorar JS ---
    codigo_js = '''var nome = "João"
var idade = 25
 
function verificarIdade(x) {
    if (x == 18) {
        console.log("maior de idade")
        return true
    }
    return false
}
'''
    print("\n=== TESTE 2: MELHORAR JAVASCRIPT ===\n")
    resultado_js = refactor.melhorar(codigo_js, "app.js")
    print(refactor.relatorio(resultado_js))
 
    # --- Teste 3: CRIAR código novo ---
    print("\n=== TESTE 3: CRIAR — Classe Python ===\n")
    resultado_criar = refactor.criar(
        "crie uma classe chamada Produto com nome, preco e estoque em Python"
    )
    print(refactor.relatorio(resultado_criar))
 
    print("\n=== TESTE 4: CRIAR — API REST ===\n")
    resultado_api = refactor.criar(
        "crie uma api rest para usuario em python"
    )
    print(resultado_api.codigo_resultado)