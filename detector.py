import cv2
import numpy as np
from PIL import Image
import easyocr
import re
from utils import votar_placas, limpar_placa, validar_placa, corrigir_placa

reader = easyocr.Reader(["pt", "en"], gpu=False)

MIN_CONF = 0.15
MIN_CONF_7CHAR = 0.20
BORRAO_THRESHOLD = 100


def _realcar_contraste(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _aumentar_resolucao(img_bgr, escala=2):
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (w * escala, h * escala), interpolation=cv2.INTER_CUBIC)


def _nitidez(img_bgr):
    suave = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    return cv2.addWeighted(img_bgr, 1.5, suave, -0.5, 0)


def _detectar_borrao(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img_cinza, cv2.CV_64F).var()


def _deblur_se_necessario(img_bgr):
    if _detectar_borrao(img_bgr) < BORRAO_THRESHOLD:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img_bgr, -1, kernel)
    return img_bgr


def _classificar_iluminacao(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    media = np.mean(img_cinza)
    desvio = np.std(img_cinza)
    if media < 60:
        return "escuro"
    if media > 200:
        return "superexposto"
    if desvio < 30:
        return "baixo_contraste"
    return "normal"


def _ajuste_gamma(img_bgr, gamma=1.0):
    inv_gamma = 1.0 / gamma
    tabela = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img_bgr, tabela)


def _remover_sombra(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top_hat = cv2.morphologyEx(img_cinza, cv2.MORPH_TOPHAT, kernel)
    return cv2.cvtColor(top_hat, cv2.COLOR_GRAY2BGR)


def _preprocess_para_ocr(img_bgr):
    img_bgr = _deblur_se_necessario(img_bgr)
    condicao = _classificar_iluminacao(img_bgr)
    if condicao == "escuro":
        img_bgr = _ajuste_gamma(img_bgr, 0.5)
    elif condicao == "superexposto":
        img_bgr = _ajuste_gamma(img_bgr, 1.8)
    elif condicao == "baixo_contraste":
        img_bgr = _remover_sombra(img_bgr)
    img_clahe = _realcar_contraste(img_bgr)
    img_sharp = _nitidez(img_clahe)
    return _aumentar_resolucao(img_sharp, escala=2)


PALAVRAS_IGNORAR = {"BRASIL", "FIAT", "CHEVROLET", "VOLKSWAGEN", "FORD", "HONDA",
                    "TOYOTA", "HYUNDAI", "NISSAN", "RENAULT", "PEUGEOT", "CITROEN",
                    "MERCEDES", "BMW", "AUDI", "KIA", "MITSUBISHI", "SUZUKI",
                    "JEEP", "LAND", "ROVER", "MINI", "SMART", "CHRYSLER",
                    "DODGE", "VOLVO", "JAC", "CHANGAN", "BYD", "GWM",
                    "MERCOSUL", "MERCOSUR",
                    "DETRAN", "DETRANSP", "DETBAN", "REPORTAGEM", "UOL",
                    "GLOBO", "FOLHA", "ESTADAO", "TERRA", "R7"}

PADRAO_PLACA = re.compile(r'^(?:[A-Z]{3}\d[A-Z]\d{2}|[A-Z]{3}\d{4}|[A-Z]{2}\d{5})$')


def _item_deteccao(bbox, texto, conf):
    textou = texto.upper().strip()
    alnums = re.sub(r'[^A-Z0-9]', '', textou)
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "texto": textou, "alnums": alnums,
        "x": int(x_min), "y": int(y_min),
        "w": int(x_max - x_min), "h": int(y_max - y_min),
        "centro_y": (y_min + y_max) / 2, "conf": conf,
    }


def _tenta_combinar(itens):
    combinados = list(itens)
    for i, c1 in enumerate(itens):
        for c2 in itens[i + 1:]:
            margem_y = max(c1["h"], c2["h"], 15) * 1.8
            if abs(c1["centro_y"] - c2["centro_y"]) < margem_y:
                espaco = c2["x"] - (c1["x"] + c1["w"])
                if 0 < espaco < max(c1["w"], 20):
                    novo_texto = c1["texto"] + c2["texto"]
                    alnums = re.sub(r'[^A-Z0-9]', '', novo_texto.upper())
                    combinados.append({
                        "texto": novo_texto, "alnums": alnums,
                        "x": c1["x"], "y": min(c1["y"], c2["y"]),
                        "w": (c2["x"] + c2["w"]) - c1["x"],
                        "h": max(c1["y"] + c1["h"], c2["y"] + c2["h"]) - min(c1["y"], c2["y"]),
                        "centro_y": (c1["centro_y"] + c2["centro_y"]) / 2,
                        "conf": min(c1["conf"], c2["conf"]),
                    })
    return combinados


def _filtrar_por_posicao(resultados, altura_img, largura_img):
    todos = []
    for bbox, texto, conf in resultados:
        textou = texto.upper().strip()
        alnums = re.sub(r'[^A-Z0-9]', '', textou)
        if len(alnums) < 4:
            continue
        if conf < MIN_CONF and len(alnums) < 7:
            continue
        if len(alnums) == 7 and conf < MIN_CONF_7CHAR:
            continue
        item = _item_deteccao(bbox, texto, conf)
        item["ignorar"] = alnums in PALAVRAS_IGNORAR
        todos.append(item)
    candidatos = _tenta_combinar(todos)
    def pontuar(p):
        alnums = p["alnums"]
        ignorado = alnums in PALAVRAS_IGNORAR
        valida = 2 if PADRAO_PLACA.match(alnums) else (1.5 if len(alnums) == 7 and any(c.isalpha() for c in alnums) and any(c.isdigit() for c in alnums) else (1 if len(alnums) >= 7 else 0))
        return (-10 if ignorado else valida, p["conf"], -p["centro_y"])
    if not candidatos:
        return None
    return max(candidatos, key=pontuar)


def _extrair_ocr_detalhado(imagem_pil):
    img_np = np.array(imagem_pil)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    return reader.readtext(img_np)


def _encontrar_regiao_placa_contornos(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for thresh_min in [50, 30, 80]:
        img_suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
        img_borda = cv2.Canny(img_suave, thresh_min, thresh_min * 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_fechada = cv2.morphologyEx(img_borda, cv2.MORPH_CLOSE, kernel, iterations=3)
        contornos, _ = cv2.findContours(img_fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
        h_orig, w_orig = img_cinza.shape
        for contorno in contornos[:30]:
            perim = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perim, True)
            x, y, w, h = cv2.boundingRect(contorno)
            if h == 0:
                continue
            aspect = w / h
            area_ratio = (w * h) / (h_orig * w_orig)
            if 1.8 < aspect < 6.0 and 0.0005 < area_ratio < 0.5:
                margem_x = int(w * 0.15)
                margem_y = int(h * 0.15)
                x = max(0, x - margem_x)
                y = max(0, y - margem_y)
                w = min(w_orig - x, w + 2 * margem_x)
                h = min(h_orig - y, h + 2 * margem_y)
                roi = img_bgr[y:y + h, x:x + w]
                if len(approx) == 4:
                    try:
                        approx_roi = approx.copy()
                        approx_roi[:, :, 0] -= x
                        approx_roi[:, :, 1] -= y
                        roi = _corrigir_perspectiva(roi, approx_roi)
                    except Exception:
                        pass
                return roi, (x, y, w, h)
    return None, None


def _corrigir_perspectiva(img_bgr, approx):
    pts = approx.reshape(4, 2).astype(np.float32)
    retangulo = np.zeros((4, 2), dtype=np.float32)
    soma = pts.sum(axis=1)
    retangulo[0] = pts[np.argmin(soma)]
    retangulo[2] = pts[np.argmax(soma)]
    diff = np.diff(pts, axis=1)
    retangulo[1] = pts[np.argmin(diff)]
    retangulo[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = retangulo
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(retangulo, dst)
    return cv2.warpPerspective(img_bgr, M, (max_width, max_height))


def _preprocess_otsu(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    _, img_limpa = cv2.threshold(img_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_limpa


def _preprocess_adaptativo(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    return cv2.adaptiveThreshold(img_suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


def _preprocess_bilateral_otsu(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filtrada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
    suave = cv2.GaussianBlur(filtrada, (5, 5), 0)
    _, img_limpa = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_limpa


def _preprocess_clahe_otsu(img_bgr):
    img_clahe = _realcar_contraste(img_bgr)
    img_cinza = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    _, img_limpa = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_limpa


def _preprocess_gamma_adaptativo(img_bgr):
    img_gamma = _ajuste_gamma(img_bgr, 1.5)
    img_cinza = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    return cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


PIPELINES_ROI = [
    _preprocess_otsu,
    _preprocess_adaptativo,
    _preprocess_bilateral_otsu,
    _preprocess_clahe_otsu,
    _preprocess_gamma_adaptativo,
]


def _coletar_textos_roi(roi):
    textos = []
    img_guardada = None
    for preproc in PIPELINES_ROI:
        img_proc = preproc(roi)
        resultados = _extrair_ocr_detalhado(Image.fromarray(img_proc))
        for _, texto, conf in resultados:
            texto = texto.upper().strip()
            alnums = re.sub(r'[^A-Z0-9]', '', texto)
            if len(alnums) == 7 and any(c.isalpha() for c in alnums) and any(c.isdigit() for c in alnums):
                textos.append(alnums)
                if img_guardada is None:
                    img_guardada = img_proc
                break
    return textos, img_guardada


def _buscar_placas_no_resultado_bruto(resultados_ocr):
    placas = set()
    for _, texto, _ in resultados_ocr:
        alnums = limpar_placa(texto or "")
        if len(alnums) == 7 and validar_placa(alnums):
            placas.add(alnums)
    if placas:
        return max(set(placas), key=list(placas).count)

    partes_letras = []
    partes_digitos = []
    for _, texto, conf in resultados_ocr:
        alnums = limpar_placa(texto or "")
        if len(alnums) in (2, 3) and all(c.isalpha() for c in alnums):
            partes_letras.append((alnums, conf))
        elif len(alnums) == 4 and all(c.isdigit() for c in alnums):
            partes_digitos.append((alnums, conf))

    for letras, lc in partes_letras:
        for digitos, dc in partes_digitos:
            tentativa = letras + digitos
            if len(tentativa) == 7:
                if validar_placa(tentativa):
                    return tentativa
                corrigido = corrigir_placa(tentativa)
                if validar_placa(corrigido):
                    return corrigido
            elif len(tentativa) == 6:
                for prefixo in "CGOQ0BDRS":
                    com_prefixo = prefixo + tentativa
                    if validar_placa(com_prefixo):
                        return com_prefixo
    return None


def _capturar_regiao(resultados_ocr, img_bgr, texto_alvo):
    for bbox, texto, _ in resultados_ocr:
        alnums = limpar_placa(texto or "")
        if alnums == texto_alvo or corrigir_placa(alnums) == texto_alvo:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x, y = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            margem = 8
            x = max(0, x - margem)
            y = max(0, y - margem)
            x2 = min(img_bgr.shape[1], x2 + margem)
            y2 = min(img_bgr.shape[0], y2 + margem)
            if x2 - x > 20 and y2 - y > 10:
                return img_bgr[y:y2, x:x2]
    return None


def _extrair_bbox(resultados_ocr, texto_alvo):
    for bbox, texto, _ in resultados_ocr:
        alnums = limpar_placa(texto or "")
        if alnums == texto_alvo or corrigir_placa(alnums) == texto_alvo:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x, y = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            return (x, y, x2 - x, y2 - y)
    return None


def detectar_placa(caminho_imagem):
    img_bgr = cv2.imread(caminho_imagem)
    if img_bgr is None:
        return None, None, None, None
    img_original = img_bgr.copy()
    h_orig, w_orig = img_bgr.shape[:2]

    # ── Passo 1: contornos na imagem original (mais preciso) ──
    roi, bbox = _encontrar_regiao_placa_contornos(img_bgr)
    if roi is not None and bbox is not None:
        roi_prep = _aumentar_resolucao(roi, escala=2)
        textos_roi, melhor_img_roi = _coletar_textos_roi(roi_prep)
        votado = votar_placas(textos_roi) if textos_roi else ""
        if len(votado) == 7 and validar_placa(votado):
            return votado, melhor_img_roi, bbox, img_original

    # ── Passo 2: fallback full-image em resolução original ──
    textos_full = []
    img_full_guardada = None
    img_clahe = _nitidez(_realcar_contraste(img_bgr))

    def _extrair_7char(resultados):
        for _, texto, _ in resultados:
            alnums = limpar_placa(texto or "")
            if len(alnums) == 7 and any(c.isalpha() for c in alnums) and any(c.isdigit() for c in alnums):
                return alnums
        return None

    # 2a: CLAHE + nitidez (sem upscale) + filtro
    resultados_clahe = _extrair_ocr_detalhado(Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)))
    melhor = _filtrar_por_posicao(resultados_clahe, h_orig, w_orig)
    if melhor and len(melhor["alnums"]) == 7:
        textos_full.append(melhor["alnums"])
        x, y, w, h = melhor["x"], melhor["y"], melhor["w"], melhor["h"]
        regiao = img_clahe[y:y + h, x:x + w]
        img_full_guardada = _preprocess_otsu(regiao) if regiao.size else None

    # 2b: Otsu na original
    resultados_otsu = _extrair_ocr_detalhado(Image.fromarray(_preprocess_otsu(img_bgr)))
    texto_otsu = _extrair_7char(resultados_otsu)
    if texto_otsu:
        textos_full.append(texto_otsu)

    # 2c: Adaptativo na original
    resultados_adapt = _extrair_ocr_detalhado(Image.fromarray(_preprocess_adaptativo(img_bgr)))
    texto_adapt = _extrair_7char(resultados_adapt)
    if texto_adapt:
        textos_full.append(texto_adapt)

    if textos_full:
        votado = votar_placas(textos_full)
        if len(votado) == 7 and validar_placa(votado):
            bbox_fallback = None
            if melhor and melhor["alnums"] == votado:
                bbox_fallback = (melhor["x"], melhor["y"], melhor["w"], melhor["h"])
            if bbox_fallback is None:
                bbox_fallback = _extrair_bbox(resultados_clahe, votado)
            if bbox_fallback is None:
                bbox_fallback = _extrair_bbox(resultados_otsu, votado)
            if bbox_fallback is None:
                bbox_fallback = _extrair_bbox(resultados_adapt, votado)
            if img_full_guardada is None:
                if melhor and melhor["alnums"] == votado:
                    x, y, w, h = melhor["x"], melhor["y"], melhor["w"], melhor["h"]
                    regiao = img_clahe[y:y+h, x:x+w]
                    img_full_guardada = _preprocess_otsu(regiao) if regiao.size else _preprocess_otsu(img_bgr)
                else:
                    img_full_guardada = _preprocess_otsu(img_bgr)
            return votado, img_full_guardada, bbox_fallback, img_original

    # ── Passo 3: busca combinada no OCR bruto (une "MG" + "3164") ──
    resultados_bruto = _extrair_ocr_detalhado(
        Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    combinada = _buscar_placas_no_resultado_bruto(resultados_bruto)
    if combinada:
        bbox_fallback = _extrair_bbox(resultados_bruto, combinada)
        img_proc = _capturar_regiao(resultados_bruto, img_bgr, combinada)
        if img_proc is not None:
            img_proc = _preprocess_otsu(img_proc)
        return combinada, img_proc, bbox_fallback, img_original

    return "", None, None, img_original
