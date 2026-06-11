import re

# Formatos de placa brasileiros
PADRAO_MERCOSUL = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')
PADRAO_CINZA = re.compile(r'^[A-Z]{3}\d{4}$')


def limpar_placa(texto):
    return re.sub(r'[^A-Za-z0-9]', '', texto).upper()


def formatar_placa(placa):
    raw = limpar_placa(placa)
    if len(raw) == 7:
        return f"{raw[:3]}{raw[3:4]}{raw[4:5]}{raw[5:]}"
    return raw


def validar_placa(placa):
    raw = limpar_placa(placa)
    return bool(PADRAO_MERCOSUL.match(raw) or PADRAO_CINZA.match(raw))


def tipo_placa(placa):
    raw = limpar_placa(placa)
    if PADRAO_MERCOSUL.match(raw):
        return "mercosul"
    if PADRAO_CINZA.match(raw):
        return "cinza"
    return "desconhecido"


def corrigir_erros_mercosul(placa):
    if len(placa) != 7:
        return placa
    chars = list(placa)
    letra_p_num = {'Z': '2', 'O': '0', 'I': '1', 'Q': '0', 'S': '5', 'G': '6', 'B': '8'}
    num_p_letra = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '7': 'T'}
    for pos in [0, 1, 2]:
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


def corrigir_placa(placa):
    raw = limpar_placa(placa)
    if validar_placa(raw):
        return raw
    tentativa = corrigir_erros_mercosul(raw)
    if validar_placa(tentativa):
        return tentativa
    tentativa = corrigir_erros_cinza(raw)
    if validar_placa(tentativa):
        return tentativa
    return raw


def extrair_placa(texto_ocr):
    raw = limpar_placa(texto_ocr)
    if len(raw) < 7:
        return raw
    for i in range(len(raw) - 6):
        candidato = raw[i:i + 7]
        if validar_placa(candidato):
            return candidato
    if len(raw) > 7:
        ultimos = raw[-7:]
        corrigido = corrigir_placa(ultimos)
        if validar_placa(corrigido):
            return corrigido
        return ultimos
    return corrigir_placa(raw)
