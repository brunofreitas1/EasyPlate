import os
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
from utils import limpar_placa, validar_placa, corrigir_placa, extrair_placa, tipo_placa

st.set_page_config(
    page_title="EasyPlate - Controle de Acesso",
    page_icon="🚗",
    layout="centered",
)

criar_tabelas()

PASTA_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(PASTA_ASSETS, exist_ok=True)

if "resultado" not in st.session_state:
    st.session_state.resultado = None


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

aba_detectar, aba_cadastrar, aba_historico = st.tabs(
    ["📸 Detectar Placa", "👥 Cadastrar Morador", "📋 Histórico de Acessos"]
)

with aba_detectar:
    st.subheader("Faça o upload de uma imagem para detectar a placa")

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
            img = Image.open(caminho).resize((200, 140))
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
        col1, col2 = st.columns(2)
        img_pil = Image.open(img_path)
        with col1:
            st.image(img_pil, caption="Imagem enviada", width='stretch')

        with st.spinner("Analisando placa..."):
            texto_bruto, img_processada, bbox, img_original = detectar_placa(img_path)

        if texto_bruto:
            placa_corrigida = extrair_placa(texto_bruto)
            tipo = tipo_placa(placa_corrigida)
            status, nome, ap = verificar_acesso(placa_corrigida)
            registrar_log(placa_corrigida, status)

            with col2:
                st.markdown("### Resultado")
                st.markdown(f"**Texto bruto:** `{texto_bruto}`")
                st.markdown(f"**Placa detectada:** `{placa_corrigida}`")
                st.markdown(f"**Tipo:** {tipo}")
                if status == "Liberado":
                    st.success(f"✅ Acesso Liberado - {nome} ({ap})")
                else:
                    st.error("⛔ Acesso Negado - Placa não autorizada")

            if img_processada is not None:
                st.markdown("**Região processada para OCR:**")
                st.image(img_processada, width='stretch')

            if img_original is not None and bbox is not None:
                img_com_bbox = _desenhar_bbox(img_original, bbox, placa_corrigida, status)
                st.markdown("**Placa localizada:**")
                st.image(img_com_bbox, width='stretch')

            st.session_state.resultado = {
                "placa": placa_corrigida,
                "status": status,
                "nome": nome,
                "ap": ap,
            }
        else:
            with col2:
                st.warning("Nenhuma placa detectada. Tente outra imagem.")
                st.session_state.resultado = None

        if not arquivo:
            st.session_state.img_path = None

with aba_cadastrar:
    st.subheader("Cadastrar novo veículo autorizado")
    with st.form("form_cadastro"):
        placa_input = st.text_input(
            "Placa do veículo",
            placeholder="Ex: ABC1D23 ou ABC1234",
            max_chars=8,
        )
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
                placa_fmt = limpar_placa(placa_input)
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
            col_a, col_b, col_c = st.columns([3, 3, 2])
            with col_a:
                st.markdown(f"**{nome}**")
            with col_b:
                st.text(f"{placa} | {ap}")
            with col_c:
                if st.button(f"Remover", key=f"del_{placa}"):
                    remover_veiculo(placa)
                    st.rerun()
    else:
        st.info("Nenhum morador cadastrado.")

with aba_historico:
    st.subheader("Últimos acessos registrados")
    total_mor, total_lib, total_neg = obter_estatisticas()
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Moradores Ativos", total_mor)
    col_m2.metric("Acessos Liberados", total_lib)
    col_m3.metric("Acessos Negados", total_neg)
    st.divider()
    registros = obter_historico(limite=100)
    if registros:
        st.markdown("| Placa | Data/Hora | Status |")
        st.markdown("|-------|-----------|--------|")
        for placa, data, status in registros:
            icon = "✅" if status == "Liberado" else "⛔"
            st.markdown(f"| `{placa}` | {data} | {icon} {status} |")
    else:
        st.info("Nenhum registro de acesso ainda.")
