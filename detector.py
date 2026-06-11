import cv2
import numpy as np
from PIL import Image
import easyocr

reader = easyocr.Reader(["pt", "en"], gpu=False)


def _preprocess(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    _, img_limpa = cv2.threshold(img_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_limpa


def _aumentar_resolucao(img_bgr, escala=2):
    h, w = img_bgr.shape[:2]
    nova_largura = int(w * escala)
    nova_altura = int(h * escala)
    return cv2.resize(img_bgr, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)


def _realcar_contraste(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _encontrar_regiao_placa(img_bgr):
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_suave = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    img_borda = cv2.Canny(img_suave, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_fechada = cv2.morphologyEx(img_borda, cv2.MORPH_CLOSE, kernel, iterations=2)
    contornos, _ = cv2.findContours(img_fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    altura_original, largura_original = img_cinza.shape
    for contorno in contornos[:20]:
        perimetro = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            area_ratio = (w * h) / (altura_original * largura_original)
            if 2.0 < aspect_ratio < 5.0 and 0.005 < area_ratio < 0.5:
                roi = img_bgr[y : y + h, x : x + w]
                return roi, (x, y, w, h)
    candidatos = []
    for contorno in contornos[:30]:
        x, y, w, h = cv2.boundingRect(contorno)
        aspect_ratio = w / h
        area_ratio = (w * h) / (altura_original * largura_original)
        if 1.5 < aspect_ratio < 6.0 and 0.003 < area_ratio < 0.3:
            candidatos.append((x, y, w, h, cv2.contourArea(contorno)))
    if candidatos:
        candidatos.sort(key=lambda c: c[4], reverse=True)
        x, y, w, h = candidatos[0][:4]
        margem_x = int(w * 0.1)
        margem_y = int(h * 0.1)
        x = max(0, x - margem_x)
        y = max(0, y - margem_y)
        w = min(largura_original - x, w + 2 * margem_x)
        h = min(altura_original - y, h + 2 * margem_y)
        roi = img_bgr[y : y + h, x : x + w]
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


def _extrair_caracteres_ocr(imagem_pil):
    img_np = np.array(imagem_pil)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    resultados = reader.readtext(img_np)
    texto = "".join(r[1] for r in resultados)
    return texto.upper()


def detectar_placa(caminho_imagem):
    img_bgr = cv2.imread(caminho_imagem)
    if img_bgr is None:
        return None, None, None, None
    img_original = img_bgr.copy()
    img_contraste = _realcar_contraste(img_bgr)
    img_redimensionada = _aumentar_resolucao(img_contraste, escala=2)
    roi, bbox = _encontrar_regiao_placa(img_redimensionada)
    escala_w = img_redimensionada.shape[1] / img_original.shape[1]
    escala_h = img_redimensionada.shape[0] / img_original.shape[0]
    img_para_ocr = None
    if roi is not None and bbox is not None:
        img_para_ocr = _preprocess(roi)
    else:
        img_para_ocr = _preprocess(img_redimensionada)
    img_pil = Image.fromarray(img_para_ocr)
    texto_bruto = _extrair_caracteres_ocr(img_pil)
    if bbox:
        xr, yr, wr, hr = bbox
        x_orig = int(xr / escala_w)
        y_orig = int(yr / escala_h)
        w_orig = int(wr / escala_w)
        h_orig = int(hr / escala_h)
        bbox_original = (x_orig, y_orig, w_orig, h_orig)
    else:
        bbox_original = None
    return texto_bruto, img_para_ocr, bbox_original, img_original
