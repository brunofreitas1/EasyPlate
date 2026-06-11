import re
from itertools import product

# Formatos de placa brasileiros
PADRAO_MERCOSUL = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')
PADRAO_CINZA = re.compile(r'^[A-Z]{3}\d{4}$')
PADRAO_ANTIGO = re.compile(r'^[A-Z]{2}\d{5}$')

# Confusões comuns de dígitos em OCR
SUBSTITUICOES_DIGITO = {
    '0': ['6', '8'],
    '6': ['0', '8', '5'],
    '8': ['0', '6', '3'],
    '1': ['7'],
    '7': ['1'],
    '3': ['8'],
    '5': ['6'],
    '4': ['H', 'A'],
}

# Confusões comuns entre letras em OCR
SUBSTITUICOES_LETRA = {
    'I': ['L'],
    'L': ['I'],
    'H': ['A', 'N'],
    'A': ['H'],
    'D': ['O'],
    'O': ['D', 'Q'],
    'Q': ['O'],
    'B': ['R'],
    'R': ['B'],
}


def limpar_placa(texto):
    return re.sub(r'[^A-Za-z0-9]', '', texto).upper()


def formatar_placa(placa):
    raw = limpar_placa(placa)
    if len(raw) == 7:
        return f"{raw[:3]}{raw[3:4]}{raw[4:5]}{raw[5:]}"
    return raw


def validar_placa(placa):
    raw = limpar_placa(placa)
    return bool(PADRAO_MERCOSUL.match(raw) or PADRAO_CINZA.match(raw) or PADRAO_ANTIGO.match(raw))


def tipo_placa(placa):
    raw = limpar_placa(placa)
    if PADRAO_MERCOSUL.match(raw):
        return "mercosul"
    if PADRAO_CINZA.match(raw):
        return "cinza"
    if PADRAO_ANTIGO.match(raw):
        return "antigo"
    return "desconhecido"


def corrigir_erros_mercosul(placa):
    if len(placa) != 7:
        return placa
    chars = list(placa)
    letra_p_num = {'Z': '2', 'O': '0', 'I': '1', 'Q': '0', 'S': '5', 'G': '6', 'B': '8'}
    num_p_letra = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '7': 'T', '4': 'A', '3': 'E'}
    for pos in [0, 1, 2, 4]:
        if chars[pos] in num_p_letra:
            chars[pos] = num_p_letra[chars[pos]]
    for pos in [3, 5, 6]:
        if chars[pos] in letra_p_num:
            chars[pos] = letra_p_num[chars[pos]]
    return "".join(chars)


def corrigir_erros_cinza(placa):
    if len(placa) != 7:
        return placa
    chars = list(placa)
    num_p_letra = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
    letra_p_num = {'Z': '2', 'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8'}
    for pos in [0, 1, 2]:
        if chars[pos] in num_p_letra:
            chars[pos] = num_p_letra[chars[pos]]
    for pos in [3, 4, 5, 6]:
        if chars[pos] in letra_p_num:
            chars[pos] = letra_p_num[chars[pos]]
    return "".join(chars)


def corrigir_erros_antigo(placa):
    if len(placa) != 7:
        return placa
    chars = list(placa)
    num_p_letra = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
    letra_p_num = {'Z': '2', 'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8'}
    for pos in [0, 1]:
        if chars[pos] in num_p_letra:
            chars[pos] = num_p_letra[chars[pos]]
    for pos in [2, 3, 4, 5, 6]:
        if chars[pos] in letra_p_num:
            chars[pos] = letra_p_num[chars[pos]]
    return "".join(chars)


def corrigir_placa(placa):
    raw = limpar_placa(placa)
    if validar_placa(raw):
        return raw
    candidatos = []
    for fn in [corrigir_erros_mercosul, corrigir_erros_cinza, corrigir_erros_antigo]:
        tentativa = fn(raw)
        if validar_placa(tentativa):
            mudancas = sum(1 for i, c in enumerate(tentativa) if c != raw[i])
            candidatos.append((mudancas, tentativa))
    if candidatos:
        candidatos.sort()
        return candidatos[0][1]
    return raw


def extrair_placa(texto_ocr):
    placas = extrair_todas_placas(texto_ocr)
    return placas[0] if placas else limpar_placa(texto_ocr)[:7]


def extrair_todas_placas(texto_ocr):
    raw = limpar_placa(texto_ocr)
    if len(raw) < 7:
        return []
    encontradas = []
    for i in range(len(raw) - 6):
        cand = raw[i:i + 7]
        if validar_placa(cand) and cand not in encontradas:
            encontradas.append(cand)
    if not encontradas and len(raw) >= 7:
        ultimos = raw[-7:]
        corrigido = corrigir_placa(ultimos)
        if validar_placa(corrigido):
            encontradas.append(corrigido)
        elif validar_placa(ultimos):
            encontradas.append(ultimos)
    return encontradas


def _posicoes_numericas(placa):
    if PADRAO_MERCOSUL.match(placa):
        return [3, 5, 6]
    if PADRAO_CINZA.match(placa):
        return [3, 4, 5, 6]
    if PADRAO_ANTIGO.match(placa):
        return [2, 3, 4, 5, 6]
    return [i for i, c in enumerate(placa) if c.isdigit()]


def _posicoes_letra(placa):
    if PADRAO_MERCOSUL.match(placa):
        return [0, 1, 2, 4]
    if PADRAO_CINZA.match(placa):
        return [0, 1, 2]
    if PADRAO_ANTIGO.match(placa):
        return [0, 1]
    return [i for i, c in enumerate(placa) if c.isalpha()]


def gerar_variacoes(placa):
    placa = limpar_placa(placa)
    if len(placa) != 7 or not validar_placa(placa):
        return [placa]
    pos_nums = _posicoes_numericas(placa)
    pos_lets = _posicoes_letra(placa)
    candidatos = set()
    candidatos.add(placa)
    for pos in pos_nums:
        char = placa[pos]
        if char not in SUBSTITUICOES_DIGITO:
            continue
        for subst in SUBSTITUICOES_DIGITO[char]:
            nova = placa[:pos] + subst + placa[pos + 1:]
            if validar_placa(nova):
                candidatos.add(nova)
    for pos in pos_lets:
        char = placa[pos]
        if char not in SUBSTITUICOES_LETRA:
            continue
        for subst in SUBSTITUICOES_LETRA[char]:
            nova = placa[:pos] + subst + placa[pos + 1:]
            if validar_placa(nova):
                candidatos.add(nova)
    if len(candidatos) <= 2:
        grupos = []
        for pos in pos_nums:
            char = placa[pos]
            if char in SUBSTITUICOES_DIGITO:
                grupos.append([(pos, s) for s in SUBSTITUICOES_DIGITO[char] if validar_placa(placa[:pos] + s + placa[pos + 1:])])
            else:
                grupos.append([])
        grupos = [g for g in grupos if g]
        if grupos:
            for combinacao in product(*grupos):
                chars = list(placa)
                for pos, val in combinacao:
                    chars[pos] = val
                candidata = "".join(chars)
                if validar_placa(candidata):
                    candidatos.add(candidata)
    return sorted(candidatos, key=lambda p: (
        p == placa,
        sum(1 for i, c in enumerate(p) if i in pos_nums and c == placa[i]),
    ), reverse=True)


def distancia_levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(prev + (0 if a[i - 1] == b[j - 1] else 1), dp[j - 1] + 1, dp[j] + 1)
            prev = temp
    return dp[n]


def buscar_similar(placa_candidata, placas_registradas):
    placa_candidata = limpar_placa(placa_candidata)
    if len(placa_candidata) != 7:
        return None
    melhor_dist = float("inf")
    melhor_placa = None
    for placa_reg in placas_registradas:
        placa_reg = limpar_placa(placa_reg)
        if len(placa_reg) != 7:
            continue
        dist = distancia_levenshtein(placa_candidata, placa_reg)
        if dist < melhor_dist:
            melhor_dist = dist
            melhor_placa = placa_reg
    if melhor_dist <= 1:
        return melhor_placa
    if melhor_dist <= 2:
        type_diff = sum(1 for i in range(7)
                        if placa_candidata[i] != melhor_placa[i]
                        and placa_candidata[i].isdigit() != melhor_placa[i].isdigit())
        if type_diff <= 1:
            return melhor_placa
    return None


def buscar_por_janela(texto_ocr, placas_registradas):
    raw = limpar_placa(texto_ocr)
    placas_reg = sorted(set(limpar_placa(p) for p in placas_registradas if len(limpar_placa(p)) == 7))
    if not placas_reg or len(raw) < 6:
        return None
    for placa in placas_reg:
        if placa in raw:
            return placa
        for v in gerar_variacoes(placa):
            if v in raw:
                return placa
    for placa in placas_reg:
        for i in range(len(raw) - 6):
            janela = raw[i:i + 7]
            if len(janela) < 7:
                continue
            if distancia_levenshtein(janela, placa) <= 2:
                return placa
    for placa in placas_reg:
        for i in range(len(raw) - 6):
            janela = raw[i:i + 7]
            if len(janela) < 7:
                continue
            if distancia_levenshtein(janela, placa) <= 3:
                return placa
    return None


def votar_placas(textos):
    textos = [limpar_placa(t) for t in textos if t and len(limpar_placa(t)) >= 6]
    if not textos:
        return ""
    setes = [t[-7:] if len(t) > 7 else t for t in textos if len(t) >= 7]
    if not setes:
        return max(set(textos), key=textos.count)
    # Fase 1: placas diretamente válidas → maioria
    validos = [t for t in setes if validar_placa(t)]
    if validos:
        return max(set(validos), key=validos.count)
    # Fase 2: placas corrigidas → maioria
    corrigidos = []
    for t in setes:
        c = corrigir_placa(t)
        if len(c) == 7 and validar_placa(c):
            corrigidos.append(c)
    if corrigidos:
        return max(set(corrigidos), key=corrigidos.count)
    # Fase 3: voto posicional só entre textos próximos (dist ≤ 2)
    from collections import Counter
    referencia = Counter(setes).most_common(1)[0][0]
    alinhados = [t for t in setes if distancia_levenshtein(t, referencia) <= 2]
    if len(alinhados) < 2:
        return corrigir_placa(referencia)
    resultado = []
    for pos in range(7):
        votos = {}
        for t in alinhados:
            c = t[pos]
            votos[c] = votos.get(c, 0) + 1
        resultado.append(max(votos, key=votos.get))
    return corrigir_placa("".join(resultado))
