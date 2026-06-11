import os
import time
import tempfile
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from database import (
    criar_tabelas,
    cadastrar_veiculo,
    listar_moradores,
    remover_veiculo,
    verificar_acesso,
    registrar_log,
    obter_historico,
    obter_estatisticas,
)
from detector import detectar_placa
from utils import limpar_placa, validar_placa, corrigir_placa, extrair_placa, extrair_todas_placas, gerar_variacoes, buscar_similar, buscar_por_janela, tipo_placa

st.set_page_config(
    page_title="EasyPlate - Controle de Acesso",
    page_icon="🚗",
    layout="centered",
)

st.markdown("""
<style>
    .main > div { padding-bottom: 2rem; }
    .stApp { background: #0e1117; }

    /* Result card */
    .result-card {
        background: #1a1d24;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2d35;
        margin-bottom: 16px;
    }
    .result-card h3 { margin-top: 0; color: #e0e0e0; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1d24;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #2a2d35;
    }

    /* Table */
    .stMarkdown table {
        background: #1a1d24;
        border-radius: 8px;
        overflow: hidden;
    }
    .stMarkdown table td, .stMarkdown table th {
        border-color: #2a2d35;
    }

    /* Input labels */
    .stTextInput label, .stFileUploader label, .stSelectbox label {
        color: #b0b0b0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #1a1d24;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: #b0b0b0;
    }
    .stTabs [aria-selected="true"] {
        background: #2a2d35;
        color: #ffffff;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }

    /* Dividers */
    hr { border-color: #2a2d35 !important; }

    /* Image captions */
    .stImage figcaption {
        color: #888 !important;
        font-size: 0.8rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1d24;
        border-radius: 8px;
        border: 1px solid #2a2d35;
    }

    @media (prefers-color-scheme: light) {
        .stApp { background: #f5f5f5; }
        div[data-testid="metric-container"],
        .result-card,
        .stMarkdown table,
        .streamlit-expanderHeader,
        .stTabs [data-baseweb="tab-list"] {
            background: #ffffff;
            border-color: #e0e0e0;
        }
        .stTabs [aria-selected="true"] { background: #e8e8e8; color: #000; }
    }
</style>
""", unsafe_allow_html=True)

criar_tabelas()

PASTA_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(PASTA_ASSETS, exist_ok=True)

if "resultado" not in st.session_state:
    st.session_state.resultado = None


def _buscar_placa_no_texto(texto_bruto):
    raw_texto = limpar_placa(texto_bruto)
    for placa, _, _, _ in listar_moradores():
        if placa in raw_texto:
            return placa
        for variante in gerar_variacoes(placa):
            if variante in raw_texto:
                return placa
    return None


def _padronizar_imagem(img_pil, largura=450, altura=320):
    if img_pil is None:
        return None
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    img_pil.thumbnail((largura, altura), Image.LANCZOS)
    canvas = Image.new("RGB", (largura, altura), (14, 17, 23))
    offset = ((largura - img_pil.width) // 2, (altura - img_pil.height) // 2)
    canvas.paste(img_pil, offset)
    return canvas


def _desenhar_bbox(img_bgr, bbox, texto, status):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if bbox:
        x, y, w, h = bbox
        cor = (0, 255, 0) if status == "Liberado" else (0, 0, 255)
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), cor, 3)
        cv2.putText(
            img_rgb,
            f"{texto} - {status}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor,
            2,
        )
    return img_rgb


st.title("🚗 EasyPlate")
st.markdown("Sistema de reconhecimento de placas para controle de acesso")

with st.sidebar:
    st.markdown("### EasyPlate v1.0")
    st.markdown("Reconhecimento de placas veiculares com EasyOCR")
    st.markdown("---")
    total_mor, total_lib, total_neg = obter_estatisticas()
    st.metric("Moradores Ativos", total_mor)
    st.metric("Acessos Liberados", total_lib)
    st.metric("Acessos Negados", total_neg)
    st.markdown("---")
    st.caption("EasyPlate © 2026")

aba_detectar, aba_cadastrar, aba_historico = st.tabs(
    ["📸 Detectar Placa", "👥 Cadastrar Morador", "📋 Histórico de Acessos"]
)

with aba_detectar:
    st.subheader("Faça o upload de uma imagem para detectar a placa")

    if "img_path" not in st.session_state:
        st.info("📸 Selecione uma imagem de exemplo abaixo ou faça upload de uma foto para começar.")
        st.markdown("---")

    imagens_exemplo = []
    if os.path.isdir(PASTA_ASSETS):
        imagens_exemplo = [
            f
            for f in os.listdir(PASTA_ASSETS)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    if imagens_exemplo:
        st.markdown("**Usar imagem de exemplo:**")
        cols = st.columns(len(imagens_exemplo))
        for i, nome_img in enumerate(imagens_exemplo):
            caminho = os.path.join(PASTA_ASSETS, nome_img)
            img = Image.open(caminho).resize((180, 120))
            with cols[i]:
                st.image(img, caption=nome_img, width='stretch')
                if st.button(f"Usar {nome_img}", key=f"exemplo_{nome_img}"):
                    st.session_state.img_path = caminho
                    st.session_state.img_nome = nome_img
                    st.rerun()

    arquivo = st.file_uploader(
        "Ou envie uma foto", type=["png", "jpg", "jpeg"], key="uploader"
    )

    img_path = None
    if arquivo:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(arquivo.getvalue())
            img_path = tmp.name
            st.session_state.img_path = img_path
            st.session_state.img_nome = arquivo.name
    elif "img_path" in st.session_state:
        img_path = st.session_state.img_path

    if img_path and os.path.exists(img_path):
        img_pil = Image.open(img_path)
        st.image(_padronizar_imagem(img_pil), caption="Imagem enviada", width=450)

        inicio = time.time()
        with st.spinner("Analisando placa..."):
            texto_bruto, img_processada, bbox, img_original = detectar_placa(img_path)
        tempo_gasto = time.time() - inicio

        if texto_bruto:
            todas = extrair_todas_placas(texto_bruto) or [limpar_placa(texto_bruto)[:7]]
            placa_final = todas[0]
            status, nome, ap = "Negado", None, None
            placas_tentar = []
            for p in todas:
                if p not in placas_tentar:
                    placas_tentar.append(p)
                for v in gerar_variacoes(p):
                    if v not in placas_tentar:
                        placas_tentar.append(v)
            for p in placas_tentar:
                s, n, a = verificar_acesso(p)
                if s == "Liberado":
                    status, nome, ap, placa_final = s, n, a, p
                    break
            if status == "Negado":
                placa_texto = _buscar_placa_no_texto(texto_bruto)
                if placa_texto:
                    s, n, a = verificar_acesso(placa_texto)
                    if s == "Liberado":
                        status, nome, ap, placa_final = s, n, a, placa_texto
            if status == "Negado":
                registradas = [p[0] for p in listar_moradores()]
                for p in todas:
                    similar = buscar_similar(p, registradas)
                    if similar:
                        s, n, a = verificar_acesso(similar)
                        if s == "Liberado":
                            status, nome, ap, placa_final = s, n, a, similar
                            break
            if status == "Negado" and len(limpar_placa(texto_bruto)) >= 7:
                placa_janela = buscar_por_janela(texto_bruto, [p[0] for p in listar_moradores()])
                if placa_janela:
                    s, n, a = verificar_acesso(placa_janela)
                    if s == "Liberado":
                        status, nome, ap, placa_final = s, n, a, placa_janela
            registrar_log(placa_final, status)

            col_img, col_res = st.columns(2)

            with st.expander("🔍 Debug - Tentativas de placa"):
                st.markdown(f"**OCR bruto:** `{texto_bruto}`")
                st.markdown(f"**Candidatos extraídos:** `{todas}`")
                st.markdown(f"**Todas as tentativas:** `{placas_tentar}`")
                st.markdown(f"**Resultado:** `{placa_final}` → **{status}**")

            with col_img:
                if img_original is not None and bbox is not None:
                    img_com_bbox = _desenhar_bbox(img_original, bbox, placa_final, status)
                    st.markdown("**Placa localizada:**")
                    st.image(_padronizar_imagem(Image.fromarray(img_com_bbox)), width=450)
                if img_processada is not None:
                    st.markdown("**Região processada para OCR:**")
                    st.image(_padronizar_imagem(Image.fromarray(img_processada)), width=450)

            with col_res:
                status_icon = "✅" if status == "Liberado" else "⛔"
                status_color = "#2ecc71" if status == "Liberado" else "#e74c3c"
                status_text = f"{status_icon} Acesso Liberado — {nome} ({ap})" if status == "Liberado" else f"{status_icon} Acesso Negado — Placa não autorizada"
                st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin-top:0;">Resultado</h3>
                    <p><strong>Placa:</strong> <code>{placa_final}</code></p>
                    <p><strong>Tipo:</strong> {tipo_placa(placa_final)}</p>
                    <p><strong>Tempo:</strong> {tempo_gasto:.1f}s</p>
                    <p style="color:{status_color};font-size:1.15rem;font-weight:600;">{status_text}</p>
                </div>
                """, unsafe_allow_html=True)

            st.session_state.resultado = {
                "placa": placa_final,
                "status": status,
                "nome": nome,
                "ap": ap,
            }
        else:
            st.warning("Nenhuma placa detectada. Tente outra imagem.")
            st.session_state.resultado = None

        if st.button("🔄 Nova detecção", use_container_width=True):
            st.session_state.img_path = None
            st.session_state.resultado = None
            st.rerun()

        # Limpa arquivo temporário
        if arquivo:
            try:
                os.unlink(img_path)
            except OSError:
                pass

        if not arquivo:
            st.session_state.img_path = None

with aba_cadastrar:
    st.subheader("Cadastrar novo veículo autorizado")

    placa_input = st.text_input(
        "Placa do veículo",
        placeholder="Ex: ABC1D23 ou ABC1234",
        max_chars=8,
        key="cadastro_placa",
    )
    placa_fmt = limpar_placa(placa_input) if placa_input else ""
    if placa_input:
        if validar_placa(placa_fmt):
            st.caption(f"✅ {tipo_placa(placa_fmt)} — formato válido")
        else:
            st.caption("❌ Formato inválido. Use ABC1D23 (Mercosul) ou ABC1234 (cinza)")

    with st.form("form_cadastro"):
        nome_input = st.text_input(
            "Nome do morador", placeholder="Ex: João Silva"
        )
        ap_input = st.text_input(
            "Apartamento / Bloco", placeholder="Ex: Ap 42 Bloco B"
        )
        submitted = st.form_submit_button("Cadastrar", width='stretch')
        if submitted:
            if not placa_input or not nome_input or not ap_input:
                st.error("Preencha todos os campos.")
            else:
                if not validar_placa(placa_fmt):
                    st.error(
                        f"Placa `{placa_fmt}` inválida. Use formato Mercosul (ABC1D23) ou cinza (ABC1234)."
                    )
                else:
                    sucesso, msg = cadastrar_veiculo(placa_fmt, nome_input, ap_input)
                    if sucesso:
                        st.success(msg)
                    else:
                        st.warning(msg)

    st.divider()
    st.subheader("Moradores cadastrados")
    moradores = listar_moradores()
    if moradores:
        for placa, nome, ap, status in moradores:
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f"""
                <div class="result-card" style="padding:12px 20px;margin-bottom:8px;">
                    <strong style="font-size:1.1rem;">{nome}</strong><br>
                    <span style="color:#888;">{placa}</span> · <span style="color:#666;">{ap}</span>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                if st.button("🗑️", key=f"del_{placa}", help="Remover"):
                    remover_veiculo(placa)
                    st.rerun()
    else:
        st.info("Nenhum morador cadastrado.")

with aba_historico:
    st.subheader("Últimos acessos registrados")
    registros = obter_historico(limite=100)
    if registros:
        for placa, data, status in registros:
            icon = "✅" if status == "Liberado" else "⛔"
            cor = "#2ecc71" if status == "Liberado" else "#e74c3c"
            st.markdown(f"""
            <div class="result-card" style="display:flex;justify-content:space-between;align-items:center;padding:10px 20px;margin-bottom:6px;">
                <div>
                    <span style="font-weight:600;">{placa}</span>
                    <span style="color:#666;margin-left:12px;">{data}</span>
                </div>
                <div>
                    <span style="color:{cor};font-weight:500;">{icon} {status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Nenhum registro de acesso ainda.")
