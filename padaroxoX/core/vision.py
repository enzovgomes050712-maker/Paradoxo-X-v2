
"""
PARADOXO X — Vision
====================
Módulo de visão computacional do ParadoxoX.
 
Responsabilidade única: receber uma imagem e devolver um
ResultadoVisao estruturado — com metadados, objetos detectados,
texto reconhecido e uma descrição gerada.
 
Arquitetura em camadas (espelha o padrão do analyzer.py):
  1. Dataclasses de resultado  → estrutura de saída limpa e serializável
  2. ValidadorImagem           → valida formato, tamanho e integridade
  3. AnalisadorVisual          → detecta regiões, cores dominantes, objetos simples
  4. ReconhecedorTexto         → OCR via pytesseract (opcional, com fallback)
  5. GeradorDescricao          → monta descrição textual a partir dos resultados
  6. VisionProcessor           → orquestra tudo, ponto de entrada principal
 
Dependências:
  OBRIGATÓRIA:  Pillow (pip install Pillow)
  OPCIONAL:     pytesseract (pip install pytesseract)
                → Se não instalado, o OCR retorna string vazia com aviso.
                → O resto do módulo funciona normalmente.
 
Integração com brain.py (quando quiser ativar):
  # No __init__ do ParadoxoBrain, acrescente:
  from vision.vision import VisionProcessor
  self.vision = VisionProcessor()
 
  # No chat(), acrescente um novo elif no bloco de intenções:
  elif tipo == "visao":
      resposta = self._handle_visao(intencao, contexto)
 
  # E implemente _handle_visao() como os outros handlers existentes.
  # Nenhum arquivo existente precisa ser modificado agora.
 
Uso standalone:
  from vision.vision import VisionProcessor
 
  vp = VisionProcessor()
  resultado = vp.processar("foto.jpg")
  print(resultado.resumo())
  print(resultado.para_dict())
"""
 
import os
import io
import math
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
 
# ── Pillow: obrigatório ────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageStat, ExifTags
    _PIL_DISPONIVEL = True
except ImportError:
    _PIL_DISPONIVEL = False
 
# ── pytesseract: opcional ──────────────────────────────────────────────────────
try:
    import pytesseract
    _TESSERACT_DISPONIVEL = True
except ImportError:
    _TESSERACT_DISPONIVEL = False
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
 
# Formatos de imagem aceitos pelo módulo
FORMATOS_ACEITOS = {"JPEG", "JPG", "PNG", "BMP", "GIF", "WEBP", "TIFF", "ICO"}
 
# Tamanho máximo aceito para carregar (50 MB)
TAMANHO_MAXIMO_BYTES = 50 * 1024 * 1024
 
# Dimensão máxima para processamento (redimensiona internamente se maior)
DIMENSAO_MAXIMA_PX = 4096
 
# Nomes das cores dominantes por intervalo HSV simplificado
_NOMES_COR = [
    (0,   30,  "vermelho"),
    (30,  60,  "laranja/amarelo"),
    (60,  90,  "amarelo/verde"),
    (90,  150, "verde"),
    (150, 180, "ciano"),
    (180, 210, "azul claro"),
    (210, 270, "azul"),
    (270, 330, "roxo/violeta"),
    (330, 360, "rosa/vermelho"),
]
 
# Extensões de arquivo → nome de formato Pillow
_EXT_PARA_FORMATO = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG",
    ".bmp": "BMP",  ".gif": "GIF",  ".webp": "WEBP",
    ".tif": "TIFF", ".tiff": "TIFF", ".ico": "ICO",
}
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATACLASSES DE RESULTADO
#    Espelham o padrão de ResultadoAnalise / ContextoCodigo do ecossistema.
# ═══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class MetadadosImagem:
    """
    Metadados técnicos da imagem.
 
    Equivalente a MetricasCodigo no analyzer.py —
    guarda os números brutos antes de qualquer interpretação.
    """
    caminho:        str   = ""
    nome_arquivo:   str   = ""
    formato:        str   = ""       # JPEG, PNG, etc.
    modo_cor:       str   = ""       # RGB, RGBA, L (grayscale), etc.
    largura:        int   = 0        # pixels
    altura:         int   = 0        # pixels
    canais:         int   = 0        # 1 = grayscale, 3 = RGB, 4 = RGBA
    tamanho_bytes:  int   = 0        # tamanho do arquivo em disco
    hash_md5:       str   = ""       # identificação única da imagem
    tem_exif:       bool  = False    # se possui metadados EXIF
    exif:           dict  = field(default_factory=dict)  # dados EXIF decodificados
    data_captura:   str   = ""       # data de captura se existir no EXIF
 
    @property
    def megapixels(self) -> float:
        """Resolução em megapixels."""
        return round((self.largura * self.altura) / 1_000_000, 2)
 
    @property
    def proporcao(self) -> str:
        """Proporção aproximada (ex: '16:9', '4:3', '1:1')."""
        if self.altura == 0:
            return "?"
        razao = self.largura / self.altura
        tabela = [(1.0, "1:1"), (1.33, "4:3"), (1.5, "3:2"),
                  (1.78, "16:9"), (2.39, "21:9")]
        melhor = min(tabela, key=lambda x: abs(x[0] - razao))
        return melhor[1] if abs(melhor[0] - razao) < 0.15 else f"{self.largura}×{self.altura}"
 
 
@dataclass
class ObjetoDetectado:
    """
    Objeto ou região identificada na imagem.
 
    Equivalente a Problema no analyzer.py — unidade atômica de descoberta.
    """
    tipo:       str   = ""    # "rosto", "texto", "objeto", "cor_dominante", "regiao"
    descricao:  str   = ""    # texto legível
    confianca:  float = 0.0   # 0.0 a 1.0
    regiao:     dict  = field(default_factory=dict)  # {x, y, w, h} em pixels se disponível
    extra:      dict  = field(default_factory=dict)  # dados adicionais livres
 
    def __str__(self):
        conf = f" (confiança: {self.confianca:.0%})" if self.confianca > 0 else ""
        return f"[{self.tipo}] {self.descricao}{conf}"
 
 
@dataclass
class ResultadoVisao:
    """
    Resultado completo do processamento de visão.
 
    Espelha ResultadoAnalise do analyzer.py:
      - metadados     ↔ metricas
      - objetos       ↔ problemas
      - cores         ↔ sugestoes_gerais
      - texto_ocr     ↔ novo campo específico de visão
      - descricao     ↔ novo campo específico de visão
    """
    # Metadados técnicos da imagem
    metadados:      MetadadosImagem = field(default_factory=MetadadosImagem)
 
    # Objetos, regiões e elementos detectados
    objetos:        list = field(default_factory=list)   # lista de ObjetoDetectado
 
    # Cores dominantes identificadas
    cores_dominantes: list = field(default_factory=list) # lista de str
 
    # Texto extraído via OCR (vazio se pytesseract não disponível)
    texto_ocr:      str  = ""
 
    # Descrição textual gerada pelo GeradorDescricao
    descricao:      str  = ""
 
    # Avisos gerados durante o processamento
    avisos:         list = field(default_factory=list)   # lista de str
 
    # Timestamp do processamento
    processado_em:  str  = field(default_factory=lambda: datetime.now().isoformat())
 
    # Se o processamento foi bem-sucedido
    sucesso:        bool = True
 
    def resumo(self) -> str:
        """
        Retorna uma linha de resumo.
        Espelha o .resumo() de ResultadoAnalise.
        """
        if not self.sucesso:
            return f"❌ Falha no processamento: {'; '.join(self.avisos)}"
        n_obj  = len(self.objetos)
        n_cor  = len(self.cores_dominantes)
        tem_txt = "✅" if self.texto_ocr.strip() else "❌"
        return (
            f"🖼️  {self.metadados.nome_arquivo} "
            f"({self.metadados.largura}×{self.metadados.altura}px, "
            f"{self.metadados.megapixels}MP) | "
            f"🔍 {n_obj} objetos | 🎨 {n_cor} cores | "
            f"📝 OCR: {tem_txt}"
        )
 
    def para_dict(self) -> dict:
        """
        Serializa para dicionário compatível com JSON.
        Útil para salvar na memory (MemoryManager / BancoDados).
        """
        return {
            "metadados": {
                "caminho":       self.metadados.caminho,
                "nome_arquivo":  self.metadados.nome_arquivo,
                "formato":       self.metadados.formato,
                "modo_cor":      self.metadados.modo_cor,
                "largura":       self.metadados.largura,
                "altura":        self.metadados.altura,
                "canais":        self.metadados.canais,
                "tamanho_bytes": self.metadados.tamanho_bytes,
                "hash_md5":      self.metadados.hash_md5,
                "tem_exif":      self.metadados.tem_exif,
                "megapixels":    self.metadados.megapixels,
                "proporcao":     self.metadados.proporcao,
                "data_captura":  self.metadados.data_captura,
            },
            "objetos": [
                {
                    "tipo":      o.tipo,
                    "descricao": o.descricao,
                    "confianca": o.confianca,
                    "regiao":    o.regiao,
                }
                for o in self.objetos
            ],
            "cores_dominantes": self.cores_dominantes,
            "texto_ocr":        self.texto_ocr,
            "descricao":        self.descricao,
            "avisos":           self.avisos,
            "processado_em":    self.processado_em,
            "sucesso":          self.sucesso,
        }
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 2. VALIDADOR DE IMAGEM
#    Separado da classe principal para ser testável de forma isolada.
#    Espelha o DetectorLinguagem do analyzer.py — responsabilidade única.
# ═══════════════════════════════════════════════════════════════════════════════
 
class ValidadorImagem:
    """
    Valida se um arquivo pode ser processado pelo VisionProcessor.
 
    Verificações realizadas:
      - Arquivo existe no sistema de arquivos
      - Extensão está na lista de formatos aceitos
      - Tamanho em bytes está dentro do limite
      - Arquivo não está corrompido (Pillow consegue abrir)
 
    Uso:
        val = ValidadorImagem()
        ok, motivo = val.validar("foto.jpg")
        if not ok:
            print(f"Imagem inválida: {motivo}")
    """
 
    def validar(self, caminho: str) -> tuple[bool, str]:
        """
        Valida o arquivo de imagem.
 
        Args:
            caminho: caminho para o arquivo de imagem.
 
        Returns:
            (True, "ok") se válido.
            (False, motivo) se inválido.
        """
        if not _PIL_DISPONIVEL:
            return False, "Pillow não está instalado (pip install Pillow)"
 
        # ── Existência ──
        if not caminho:
            return False, "Caminho não informado"
        if not os.path.exists(caminho):
            return False, f"Arquivo não encontrado: {caminho}"
        if not os.path.isfile(caminho):
            return False, f"O caminho não aponta para um arquivo: {caminho}"
 
        # ── Extensão ──
        _, ext = os.path.splitext(caminho)
        formato_ext = _EXT_PARA_FORMATO.get(ext.lower())
        if formato_ext is None:
            return False, (
                f"Extensão '{ext}' não suportada. "
                f"Use: {', '.join(_EXT_PARA_FORMATO.keys())}"
            )
 
        # ── Tamanho em bytes ──
        tamanho = os.path.getsize(caminho)
        if tamanho == 0:
            return False, "Arquivo está vazio (0 bytes)"
        if tamanho > TAMANHO_MAXIMO_BYTES:
            mb = tamanho / (1024 * 1024)
            return False, (
                f"Arquivo muito grande ({mb:.1f} MB). "
                f"Limite: {TAMANHO_MAXIMO_BYTES // (1024*1024)} MB"
            )
 
        # ── Integridade (Pillow consegue abrir?) ──
        try:
            with Image.open(caminho) as img:
                img.verify()  # verifica sem carregar pixels na memória
        except Exception as e:
            return False, f"Arquivo corrompido ou formato inválido: {e}"
 
        return True, "ok"
 
    def validar_objeto_pil(self, imagem) -> tuple[bool, str]:
        """
        Valida um objeto PIL.Image já carregado em memória.
        Útil para imagens recebidas de stream/bytes sem arquivo no disco.
        """
        if not _PIL_DISPONIVEL:
            return False, "Pillow não disponível"
        if imagem is None:
            return False, "Objeto de imagem é None"
        if not hasattr(imagem, "size"):
            return False, "Objeto não é um PIL.Image válido"
        w, h = imagem.size
        if w == 0 or h == 0:
            return False, "Imagem com dimensões zero"
        return True, "ok"
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALISADOR VISUAL
#    Detecta regiões, cores dominantes e elementos visuais básicos.
#    Não usa modelos de ML pesados — funciona só com Pillow.
# ═══════════════════════════════════════════════════════════════════════════════
 
class AnalisadorVisual:
    """
    Analisa o conteúdo visual de uma imagem usando apenas Pillow.
 
    O que detecta:
      - Cores dominantes (por quantização de paleta)
      - Brilho e contraste gerais
      - Proporção de pixels escuros/claros
      - Presença de regiões de texto (alta frequência espacial)
      - Número aproximado de regiões distintas
 
    Por que sem ML pesado?
      O ecossistema ParadoxoX é construído do zero, sem dependências
      de frameworks como PyTorch/TensorFlow. O AnalisadorVisual segue
      o mesmo princípio — extrai informações reais e úteis usando
      apenas operações matemáticas sobre pixels.
 
    Expansão futura:
      Quando quiser detecção de objetos via YOLO ou classificação via
      torchvision, basta criar um AnalisadorML e plugar no VisionProcessor
      sem alterar esta classe.
    """
 
    def extrair_metadados(
        self, imagem, caminho: str = ""
    ) -> MetadadosImagem:
        """
        Extrai metadados técnicos completos de um objeto PIL.Image.
 
        Args:
            imagem:  objeto PIL.Image já carregado.
            caminho: caminho original do arquivo (para nome e hash).
 
        Returns:
            MetadadosImagem preenchido.
        """
        meta = MetadadosImagem()
        meta.caminho      = caminho
        meta.nome_arquivo = os.path.basename(caminho) if caminho else "imagem_memoria"
        meta.formato      = imagem.format or _EXT_PARA_FORMATO.get(
            os.path.splitext(caminho)[1].lower(), "PNG"
        )
        meta.modo_cor     = imagem.mode
        meta.largura, meta.altura = imagem.size
        meta.canais       = len(imagem.getbands())
 
        # Tamanho em bytes
        if caminho and os.path.exists(caminho):
            meta.tamanho_bytes = os.path.getsize(caminho)
        else:
            # Estimativa a partir do buffer em memória
            buf = io.BytesIO()
            imagem.save(buf, format=meta.formato or "PNG")
            meta.tamanho_bytes = buf.tell()
 
        # Hash MD5 para identificação única
        if caminho and os.path.exists(caminho):
            meta.hash_md5 = self._hash_arquivo(caminho)
        else:
            meta.hash_md5 = self._hash_imagem(imagem)
 
        # EXIF
        meta.exif, meta.tem_exif, meta.data_captura = self._extrair_exif(imagem)
 
        return meta
 
    def detectar_cores_dominantes(
        self, imagem, n_cores: int = 5
    ) -> list[str]:
        """
        Retorna até n_cores nomes de cores dominantes na imagem.
 
        Método: quantização de paleta do Pillow (reduz imagem para
        N cores e nomeia cada uma pelo ângulo HSV aproximado).
 
        Args:
            imagem:  PIL.Image
            n_cores: quantas cores retornar
 
        Returns:
            Lista de strings com nomes das cores (ex: ["azul", "branco", "verde"]).
        """
        # Reduz para RGB se necessário
        img_rgb = imagem.convert("RGB")
 
        # Redimensiona para acelerar (max 200x200 para quantização)
        img_pequena = img_rgb.copy()
        img_pequena.thumbnail((200, 200))
 
        try:
            # Quantiza em n_cores+5 para depois filtrar brancos/pretos
            quantizada = img_pequena.quantize(colors=n_cores + 5, method=2)
            paleta_raw = quantizada.getpalette()
            if not paleta_raw:
                return []
 
            # Conta pixels por cor quantizada
            pixels = list(quantizada.getdata())
            contagem = {}
            for p in pixels:
                contagem[p] = contagem.get(p, 0) + 1
 
            # Ordena por frequência
            indices_ordenados = sorted(contagem, key=lambda i: contagem[i], reverse=True)
 
            nomes = []
            vistos = set()
            for idx in indices_ordenados:
                r = paleta_raw[idx * 3]
                g = paleta_raw[idx * 3 + 1]
                b = paleta_raw[idx * 3 + 2]
                nome = self._rgb_para_nome(r, g, b)
                if nome not in vistos:
                    vistos.add(nome)
                    nomes.append(nome)
                if len(nomes) >= n_cores:
                    break
            return nomes
 
        except Exception:
            # Fallback: usa estatísticas simples de canal
            return self._cores_por_estatistica(img_rgb)
 
    def analisar_regioes(self, imagem) -> list[ObjetoDetectado]:
        """
        Detecta características visuais gerais como regiões claras/escuras,
        presença de gradientes (indicativo de foto vs. diagrama), etc.
 
        Retorna lista de ObjetoDetectado com tipo="regiao".
        """
        objetos = []
        img_rgb = imagem.convert("RGB")
 
        # Estatísticas de brilho
        stat = ImageStat.Stat(img_rgb)
        brilho_medio = sum(stat.mean) / 3
        desvio_brilho = sum(stat.stddev) / 3
 
        # ── Classificação de brilho ──
        if brilho_medio > 200:
            objetos.append(ObjetoDetectado(
                tipo="caracteristica",
                descricao="Imagem predominantemente clara (fundo branco ou overexposed)",
                confianca=0.8,
            ))
        elif brilho_medio < 50:
            objetos.append(ObjetoDetectado(
                tipo="caracteristica",
                descricao="Imagem predominantemente escura (noturna ou subexposed)",
                confianca=0.8,
            ))
        else:
            objetos.append(ObjetoDetectado(
                tipo="caracteristica",
                descricao=f"Imagem com exposição equilibrada (brilho médio: {brilho_medio:.0f}/255)",
                confianca=0.9,
            ))
 
        # ── Contraste ──
        if desvio_brilho > 60:
            objetos.append(ObjetoDetectado(
                tipo="caracteristica",
                descricao="Alto contraste — muita variação entre áreas claras e escuras",
                confianca=0.75,
            ))
        elif desvio_brilho < 15:
            objetos.append(ObjetoDetectado(
                tipo="caracteristica",
                descricao="Baixo contraste — imagem com tonalidade uniforme",
                confianca=0.75,
            ))
 
        # ── Tipo de imagem (foto vs diagrama/screenshot) ──
        tipo_imagem = self._classificar_tipo_imagem(img_rgb, desvio_brilho)
        objetos.append(tipo_imagem)
 
        # ── Orientação ──
        w, h = imagem.size
        if w > h * 1.5:
            objetos.append(ObjetoDetectado(
                tipo="orientacao",
                descricao="Paisagem (landscape) — imagem mais larga que alta",
                confianca=1.0,
            ))
        elif h > w * 1.5:
            objetos.append(ObjetoDetectado(
                tipo="orientacao",
                descricao="Retrato (portrait) — imagem mais alta que larga",
                confianca=1.0,
            ))
        else:
            objetos.append(ObjetoDetectado(
                tipo="orientacao",
                descricao="Quadrada ou próxima de quadrada",
                confianca=0.9,
            ))
 
        return objetos
 
    # ── Internos ────────────────────────────────────────────────────────────────
 
    def _classificar_tipo_imagem(
        self, img_rgb, desvio_brilho: float
    ) -> ObjetoDetectado:
        """
        Heurística simples: screenshots/diagramas tendem a ter pouquíssimas
        cores distintas e bordas muito nítidas; fotos têm muita variação.
        """
        # Amostra pequena para contar cores únicas rapidamente
        img_mini = img_rgb.copy()
        img_mini.thumbnail((64, 64))
        pixels = list(img_mini.getdata())
        cores_unicas = len(set(pixels))
        total_pixels = len(pixels)
        razao = cores_unicas / max(total_pixels, 1)
 
        if razao < 0.05:
            return ObjetoDetectado(
                tipo="tipo_imagem",
                descricao="Provavelmente um diagrama, ícone ou imagem com paleta limitada",
                confianca=0.7,
                extra={"razao_cores": round(razao, 4)},
            )
        elif razao > 0.6:
            return ObjetoDetectado(
                tipo="tipo_imagem",
                descricao="Provavelmente uma fotografia com gradações ricas de cor",
                confianca=0.75,
                extra={"razao_cores": round(razao, 4)},
            )
        else:
            return ObjetoDetectado(
                tipo="tipo_imagem",
                descricao="Imagem com complexidade visual média (ilustração ou screenshot)",
                confianca=0.6,
                extra={"razao_cores": round(razao, 4)},
            )
 
    @staticmethod
    def _rgb_para_nome(r: int, g: int, b: int) -> str:
        """Converte RGB para nome de cor humano via HSV simplificado."""
        # Normaliza para [0, 1]
        rf, gf, bf = r / 255, g / 255, b / 255
        cmax = max(rf, gf, bf)
        cmin = min(rf, gf, bf)
        delta = cmax - cmin
 
        # Preto, branco e cinza
        if cmax < 0.15:
            return "preto"
        if cmin > 0.85:
            return "branco"
        if delta < 0.12:
            return "cinza"
 
        # Saturação baixa = cinza médio
        s = delta / cmax if cmax > 0 else 0
        if s < 0.2:
            return "cinza"
 
        # Hue em graus
        if delta == 0:
            h = 0.0
        elif cmax == rf:
            h = 60 * (((gf - bf) / delta) % 6)
        elif cmax == gf:
            h = 60 * (((bf - rf) / delta) + 2)
        else:
            h = 60 * (((rf - gf) / delta) + 4)
 
        for inicio, fim, nome in _NOMES_COR:
            if inicio <= h < fim:
                return nome
        return "vermelho"  # 330-360 fecha o ciclo
 
    @staticmethod
    def _cores_por_estatistica(img_rgb) -> list[str]:
        """Fallback: nomeia cores pelo canal dominante médio."""
        stat = ImageStat.Stat(img_rgb)
        r, g, b = stat.mean
        nomes = []
        if r > 150 and r > g + 30 and r > b + 30:
            nomes.append("vermelho")
        if g > 150 and g > r + 30 and g > b + 30:
            nomes.append("verde")
        if b > 150 and b > r + 30 and b > g + 30:
            nomes.append("azul")
        if r > 200 and g > 200 and b > 200:
            nomes.append("branco")
        if r < 60 and g < 60 and b < 60:
            nomes.append("preto")
        return nomes or ["indefinido"]
 
    @staticmethod
    def _extrair_exif(imagem) -> tuple[dict, bool, str]:
        """
        Tenta extrair metadados EXIF da imagem.
        Retorna (dict_exif, tem_exif, data_captura).
        """
        exif_dict = {}
        data_captura = ""
        try:
            raw = imagem._getexif()
            if raw:
                for tag_id, valor in raw.items():
                    tag_nome = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Serializa apenas tipos simples
                    if isinstance(valor, (str, int, float)):
                        exif_dict[tag_nome] = valor
                    elif isinstance(valor, bytes):
                        exif_dict[tag_nome] = valor.hex()[:64]
                    else:
                        exif_dict[tag_nome] = str(valor)[:128]
                # Tenta pegar data de captura
                data_captura = (
                    exif_dict.get("DateTime") or
                    exif_dict.get("DateTimeOriginal") or
                    ""
                )
                return exif_dict, True, data_captura
        except Exception:
            pass
        return {}, False, ""
 
    @staticmethod
    def _hash_arquivo(caminho: str) -> str:
        """MD5 do arquivo em disco (em chunks para arquivos grandes)."""
        h = hashlib.md5()
        with open(caminho, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
 
    @staticmethod
    def _hash_imagem(imagem) -> str:
        """MD5 dos dados brutos de um PIL.Image em memória."""
        h = hashlib.md5()
        buf = io.BytesIO()
        imagem.save(buf, format="PNG")
        h.update(buf.getvalue())
        return h.hexdigest()
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 4. RECONHECEDOR DE TEXTO (OCR)
#    Wrapper em torno do pytesseract com fallback gracioso.
# ═══════════════════════════════════════════════════════════════════════════════
 
class ReconhecedorTexto:
    """
    Reconhecimento óptico de caracteres (OCR) via pytesseract.
 
    Se pytesseract não estiver instalado, o método reconhecer()
    retorna string vazia e registra um aviso — o restante do
    VisionProcessor continua funcionando normalmente.
 
    Para instalar:
        pip install pytesseract
        # E instalar o binário Tesseract no sistema operacional:
        # Windows: https://github.com/UB-Mannheim/tesseract/wiki
        # Ubuntu:  sudo apt install tesseract-ocr
        # macOS:   brew install tesseract
 
    Configurações padrão:
        lang="por+eng"  → português + inglês (melhor cobertura)
        config="--psm 3"→ modo automático de segmentação de página
    """
 
    def __init__(
        self,
        lang: str = "por+eng",
        config: str = "--psm 3",
    ):
        self.lang   = lang
        self.config = config
 
    def reconhecer(self, imagem) -> tuple[str, list[str]]:
        """
        Extrai texto da imagem.
 
        Args:
            imagem: PIL.Image
 
        Returns:
            (texto_extraido, lista_de_avisos)
            texto_extraido é string vazia se OCR não disponível ou falhar.
        """
        avisos = []
 
        if not _TESSERACT_DISPONIVEL:
            avisos.append(
                "pytesseract não instalado — OCR desativado. "
                "Para ativar: pip install pytesseract"
            )
            return "", avisos
 
        try:
            # Pré-processa para melhorar OCR: converte para grayscale
            img_ocr = self._preprocessar(imagem)
 
            texto = pytesseract.image_to_string(
                img_ocr,
                lang=self.lang,
                config=self.config,
            )
            texto = texto.strip()
 
            if not texto:
                avisos.append("OCR não encontrou texto na imagem.")
 
            return texto, avisos
 
        except pytesseract.TesseractNotFoundError:
            avisos.append(
                "Binário do Tesseract não encontrado no sistema. "
                "Instale em: https://github.com/UB-Mannheim/tesseract/wiki"
            )
            return "", avisos
 
        except Exception as e:
            avisos.append(f"Erro no OCR: {type(e).__name__}: {e}")
            return "", avisos
 
    @staticmethod
    def _preprocessar(imagem) -> "Image":
        """
        Pré-processamento da imagem antes do OCR.
 
        Converte para escala de cinza para reduzir ruído de cor
        e melhorar a detecção de bordas de caracteres.
        """
        # Converte para RGB primeiro (normaliza RGBA, paleta, etc.)
        img = imagem.convert("RGB")
        # Grayscale — Tesseract performa melhor em escala de cinza
        img = img.convert("L")
        return img
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 5. GERADOR DE DESCRIÇÃO
#    Monta uma descrição textual coerente a partir dos dados coletados.
#    Análogo ao GeradorResposta do brain.py — transforma dados em linguagem.
# ═══════════════════════════════════════════════════════════════════════════════
 
class GeradorDescricao:
    """
    Gera uma descrição textual da imagem combinando todos os dados
    coletados pelas etapas anteriores.
 
    Não usa modelo de linguagem externo — monta a descrição de forma
    determinística a partir de templates e dados estruturados.
 
    Expansão futura:
        Para usar o ParadoxoTransformer para gerar a descrição,
        basta injetar o transformer como dependência aqui e substituir
        o método gerar() pela chamada ao modelo.
    """
 
    def gerar(
        self,
        metadados:       MetadadosImagem,
        objetos:         list,
        cores:           list[str],
        texto_ocr:       str,
    ) -> str:
        """
        Gera descrição textual completa.
 
        Args:
            metadados:  MetadadosImagem preenchido.
            objetos:    lista de ObjetoDetectado.
            cores:      lista de nomes de cores dominantes.
            texto_ocr:  texto extraído pelo OCR.
 
        Returns:
            String com descrição formatada e legível.
        """
        linhas = []
 
        # ── Cabeçalho ──
        linhas.append(f"📷 **{metadados.nome_arquivo}**")
        linhas.append(
            f"   Formato: {metadados.formato} | "
            f"Dimensões: {metadados.largura}×{metadados.altura}px | "
            f"Resolução: {metadados.megapixels}MP | "
            f"Proporção: {metadados.proporcao}"
        )
        if metadados.data_captura:
            linhas.append(f"   Data de captura: {metadados.data_captura}")
 
        # ── Características visuais ──
        if objetos:
            linhas.append("\n🔍 **Características detectadas:**")
            for obj in objetos:
                linhas.append(f"   • {obj.descricao}")
 
        # ── Cores ──
        if cores:
            lista_cores = ", ".join(cores)
            linhas.append(f"\n🎨 **Cores dominantes:** {lista_cores}")
 
        # ── Texto OCR ──
        if texto_ocr.strip():
            linhas.append("\n📝 **Texto encontrado na imagem:**")
            # Exibe até 500 caracteres para não sobrecarregar
            trecho = texto_ocr.strip()
            if len(trecho) > 500:
                trecho = trecho[:500] + "..."
            for linha in trecho.split("\n"):
                linha = linha.strip()
                if linha:
                    linhas.append(f"   {linha}")
        else:
            linhas.append("\n📝 **Texto:** Nenhum texto detectado na imagem.")
 
        # ── EXIF ──
        if metadados.tem_exif and metadados.exif:
            campos_uteis = [
                "Make", "Model", "Software",
                "ImageWidth", "ImageLength",
                "XResolution", "YResolution",
            ]
            exif_linha = []
            for campo in campos_uteis:
                if campo in metadados.exif:
                    exif_linha.append(f"{campo}: {metadados.exif[campo]}")
            if exif_linha:
                linhas.append("\n📋 **Metadados EXIF:** " + " | ".join(exif_linha))
 
        return "\n".join(linhas)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISIONPROCESSOR — Ponto de Entrada Principal
#    Orquestra todas as etapas e expõe a interface pública.
#    Equivale ao ParadoxoBrain para o módulo de visão.
# ═══════════════════════════════════════════════════════════════════════════════
 
class VisionProcessor:
    """
    Processador de visão computacional do ParadoxoX.
 
    Orquestra todas as etapas de análise de imagem e retorna
    um ResultadoVisao estruturado.
 
    Uso básico:
        vp = VisionProcessor()
        resultado = vp.processar("foto.jpg")
        print(resultado.resumo())
 
    Uso com objeto PIL já carregado:
        from PIL import Image
        img = Image.open("foto.png")
        resultado = vp.processar_pil(img, nome="foto.png")
 
    Integração com brain.py (adicionar em ParadoxoBrain.__init__):
        from vision.vision import VisionProcessor
        self.vision = VisionProcessor()
 
    Pipeline completo executado por processar():
        1. validar_imagem()         → verifica arquivo
        2. carregar_imagem()        → abre com Pillow
        3. extrair_metadados()      → AnalisadorVisual.extrair_metadados()
        4. detectar_objetos()       → AnalisadorVisual.analisar_regioes()
        5. detectar_cores()         → AnalisadorVisual.detectar_cores_dominantes()
        6. reconhecer_texto()       → ReconhecedorTexto.reconhecer()
        7. gerar_descricao()        → GeradorDescricao.gerar()
        8. → ResultadoVisao
    """
 
    def __init__(
        self,
        ocr_lang:   str = "por+eng",
        ocr_config: str = "--psm 3",
        n_cores:    int = 5,
    ):
        """
        Args:
            ocr_lang:   idiomas do OCR (padrão: português + inglês).
            ocr_config: configuração do tesseract.
            n_cores:    número de cores dominantes a detectar.
        """
        if not _PIL_DISPONIVEL:
            raise ImportError(
                "Pillow é obrigatório para o VisionProcessor. "
                "Instale com: pip install Pillow"
            )
 
        self.n_cores = n_cores
 
        # Componentes internos
        self._validador   = ValidadorImagem()
        self._analisador  = AnalisadorVisual()
        self._ocr         = ReconhecedorTexto(lang=ocr_lang, config=ocr_config)
        self._descricao   = GeradorDescricao()
 
    # ── API PÚBLICA ─────────────────────────────────────────────────────────────
 
    def processar(self, caminho: str) -> ResultadoVisao:
        """
        Pipeline completo: valida → carrega → analisa → descreve.
 
        Args:
            caminho: caminho para o arquivo de imagem.
 
        Returns:
            ResultadoVisao com todos os dados preenchidos.
            Em caso de erro, ResultadoVisao.sucesso = False e
            ResultadoVisao.avisos contém a descrição do problema.
        """
        resultado = ResultadoVisao()
 
        # ── Etapa 1: Validação ──
        ok, motivo = self.validar_imagem(caminho)
        if not ok:
            resultado.sucesso = False
            resultado.avisos.append(f"Validação falhou: {motivo}")
            return resultado
 
        # ── Etapa 2: Carregamento ──
        imagem, aviso_carga = self.carregar_imagem(caminho)
        if imagem is None:
            resultado.sucesso = False
            resultado.avisos.append(f"Falha ao carregar: {aviso_carga}")
            return resultado
        if aviso_carga:
            resultado.avisos.append(aviso_carga)
 
        # ── Etapas 3–7: Processamento ──
        return self.processar_pil(imagem, nome=os.path.basename(caminho), caminho=caminho)
 
    def processar_pil(
        self,
        imagem,
        nome:    str = "imagem",
        caminho: str = "",
    ) -> ResultadoVisao:
        """
        Processa um objeto PIL.Image já carregado.
 
        Útil quando a imagem vem de stream, bytes ou outra fonte
        que não seja um arquivo em disco.
 
        Args:
            imagem:  objeto PIL.Image.
            nome:    nome descritivo (para o resultado).
            caminho: caminho original se disponível.
 
        Returns:
            ResultadoVisao completo.
        """
        resultado = ResultadoVisao()
 
        # Valida o objeto PIL
        ok, motivo = self._validador.validar_objeto_pil(imagem)
        if not ok:
            resultado.sucesso = False
            resultado.avisos.append(f"Imagem PIL inválida: {motivo}")
            return resultado
 
        # Redimensiona se muito grande (não modifica o original)
        imagem, aviso_resize = self._redimensionar_se_necessario(imagem)
        if aviso_resize:
            resultado.avisos.append(aviso_resize)
 
        # ── Etapa 3: Metadados ──
        resultado.metadados = self.extrair_metadados(imagem, caminho=caminho)
        if not resultado.metadados.nome_arquivo or resultado.metadados.nome_arquivo == "imagem_memoria":
            resultado.metadados.nome_arquivo = nome
 
        # ── Etapa 4: Detecção de objetos/regiões ──
        resultado.objetos = self.detectar_objetos(imagem)
 
        # ── Etapa 5: Cores dominantes ──
        resultado.cores_dominantes = self.detectar_cores(imagem)
 
        # ── Etapa 6: OCR ──
        resultado.texto_ocr, avisos_ocr = self.reconhecer_texto(imagem)
        resultado.avisos.extend(avisos_ocr)
 
        # ── Etapa 7: Descrição ──
        resultado.descricao = self.gerar_descricao(
            resultado.metadados,
            resultado.objetos,
            resultado.cores_dominantes,
            resultado.texto_ocr,
        )
 
        resultado.sucesso = True
        return resultado
 
    def validar_imagem(self, caminho: str) -> tuple[bool, str]:
        """
        Valida um arquivo de imagem antes de processá-lo.
 
        Returns:
            (True, "ok") se válido, (False, motivo) se inválido.
        """
        return self._validador.validar(caminho)
 
    def carregar_imagem(self, caminho: str) :
        """
        Carrega um arquivo de imagem do disco.
 
        Returns:
            (PIL.Image, "") em sucesso.
            (None, mensagem_de_erro) em falha.
        """
        try:
            imagem = Image.open(caminho)
            # Converte para modo compatível se necessário
            if imagem.mode not in ("RGB", "RGBA", "L", "P", "LA"):
                imagem = imagem.convert("RGB")
            # Força carregamento completo (resolve lazy load do Pillow)
            imagem.load()
            return imagem, ""
        except FileNotFoundError:
            return None, f"Arquivo não encontrado: {caminho}"
        except Exception as e:
            return None, f"Erro ao abrir imagem: {type(e).__name__}: {e}"
 
    def extrair_metadados(self, imagem, caminho: str = "") -> MetadadosImagem:
        """
        Extrai metadados técnicos de um PIL.Image.
 
        Returns:
            MetadadosImagem preenchido.
        """
        return self._analisador.extrair_metadados(imagem, caminho=caminho)
 
    def detectar_objetos(self, imagem) -> list:
        """
        Detecta características visuais, regiões e classificações.
 
        Returns:
            Lista de ObjetoDetectado.
        """
        return self._analisador.analisar_regioes(imagem)
 
    def detectar_cores(self, imagem) -> list[str]:
        """
        Retorna lista de nomes de cores dominantes.
 
        Returns:
            Lista de strings com nomes das cores.
        """
        return self._analisador.detectar_cores_dominantes(imagem, n_cores=self.n_cores)
 
    def reconhecer_texto(self, imagem) -> tuple[str, list[str]]:
        """
        Extrai texto da imagem via OCR.
 
        Returns:
            (texto, lista_de_avisos)
        """
        return self._ocr.reconhecer(imagem)
 
    def gerar_descricao(
        self,
        metadados:    MetadadosImagem,
        objetos:      list,
        cores:        list[str],
        texto_ocr:    str,
    ) -> str:
        """
        Gera descrição textual completa da imagem.
 
        Returns:
            String formatada com a descrição.
        """
        return self._descricao.gerar(metadados, objetos, cores, texto_ocr)
 
    def resultado_para_json(self, resultado: ResultadoVisao) -> str:
        """
        Serializa um ResultadoVisao para JSON formatado.
 
        Útil para salvar no MemoryManager ou expor via API.
        """
        return json.dumps(resultado.para_dict(), ensure_ascii=False, indent=2)
 
    # ── Internos ────────────────────────────────────────────────────────────────
 
    @staticmethod
    def _redimensionar_se_necessario(imagem) :
        """
        Redimensiona a imagem se ultrapassar DIMENSAO_MAXIMA_PX.
        Mantém proporção. Não altera o objeto original.
        """
        w, h = imagem.size
        if w <= DIMENSAO_MAXIMA_PX and h <= DIMENSAO_MAXIMA_PX:
            return imagem, ""
        fator = DIMENSAO_MAXIMA_PX / max(w, h)
        novo_w = int(w * fator)
        novo_h = int(h * fator)
        img_redimensionada = imagem.copy()
        img_redimensionada = img_redimensionada.resize(
            (novo_w, novo_h), Image.LANCZOS
        )
        aviso = (
            f"Imagem redimensionada de {w}×{h} para {novo_w}×{novo_h} "
            f"(limite: {DIMENSAO_MAXIMA_PX}px)"
        )
        return img_redimensionada, aviso
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO STANDALONE (teste rápido)
# Rode diretamente: python vision.py caminho/para/imagem.jpg
# ═══════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    import sys
 
    print("=" * 60)
    print("PARADOXO X — Vision (teste standalone)")
    print("=" * 60)
 
    # Verifica dependências
    print(f"\n📦 Pillow:       {'✅ disponível' if _PIL_DISPONIVEL else '❌ NÃO instalado (pip install Pillow)'}")
    print(f"📦 pytesseract:  {'✅ disponível' if _TESSERACT_DISPONIVEL else '⚠️  não instalado (OCR desativado)'}")
    print()
 
    if not _PIL_DISPONIVEL:
        print("❌ Instale o Pillow antes de usar: pip install Pillow")
        sys.exit(1)
 
    if len(sys.argv) < 2:
        print("Uso: python vision.py <caminho_da_imagem>")
        print("Exemplo: python vision.py foto.jpg")
        sys.exit(0)
 
    caminho = sys.argv[1]
    print(f"🔍 Processando: {caminho}\n")
 
    vp = VisionProcessor()
    resultado = vp.processar(caminho)
 
    print(resultado.resumo())
    print()
 
    if resultado.sucesso:
        print(resultado.descricao)
        if resultado.avisos:
            print("\n⚠️  Avisos:")
            for av in resultado.avisos:
                print(f"   • {av}")
    else:
        print("❌ Falha no processamento:")
        for av in resultado.avisos:
            print(f"   • {av}")
 