import sqlite3
import os
from datetime import datetime
import pytz

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "condominio.db")


def conectar():
    return sqlite3.connect(DB_PATH)


def criar_tabelas():
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moradores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT NOT NULL UNIQUE,
            morador_nome TEXT NOT NULL,
            apartamento_bloco TEXT NOT NULL,
            status TEXT DEFAULT 'Ativo'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico_acessos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT NOT NULL,
            data_hora TEXT NOT NULL,
            status_acesso TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def cadastrar_veiculo(placa, nome, ap_bloco):
    conn = conectar()
    cursor = conn.cursor()
    placa_fmt = placa.upper().replace("-", "").replace(" ", "").strip()
    try:
        cursor.execute(
            "INSERT INTO moradores (placa, morador_nome, apartamento_bloco) VALUES (?, ?, ?)",
            (placa_fmt, nome, ap_bloco),
        )
        conn.commit()
        return True, f"Veículo {placa_fmt} cadastrado com sucesso."
    except sqlite3.IntegrityError:
        return False, f"Placa {placa_fmt} já está cadastrada."
    finally:
        conn.close()


def remover_veiculo(placa):
    conn = conectar()
    cursor = conn.cursor()
    placa_fmt = placa.upper().replace("-", "").replace(" ", "").strip()
    cursor.execute("DELETE FROM moradores WHERE placa = ?", (placa_fmt,))
    conn.commit()
    removido = cursor.rowcount > 0
    conn.close()
    return removido


def listar_moradores():
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT placa, morador_nome, apartamento_bloco, status FROM moradores ORDER BY morador_nome"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def verificar_acesso(placa):
    conn = conectar()
    cursor = conn.cursor()
    placa_fmt = placa.upper().replace("-", "").replace(" ", "").strip()
    cursor.execute(
        "SELECT morador_nome, apartamento_bloco FROM moradores WHERE placa = ? AND status = 'Ativo'",
        (placa_fmt,),
    )
    resultado = cursor.fetchone()
    conn.close()
    if resultado:
        return "Liberado", resultado[0], resultado[1]
    return "Negado", None, None


def registrar_log(placa, status_acesso):
    conn = conectar()
    cursor = conn.cursor()
    fuso = pytz.timezone("America/Sao_Paulo")
    agora = datetime.now(fuso).strftime("%d/%m/%Y %H:%M:%S")
    cursor.execute(
        "INSERT INTO historico_acessos (placa, data_hora, status_acesso) VALUES (?, ?, ?)",
        (placa, agora, status_acesso),
    )
    conn.commit()
    conn.close()


def obter_historico(limite=50, offset=0):
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT placa, data_hora, status_acesso FROM historico_acessos ORDER BY id DESC LIMIT ? OFFSET ?",
        (limite, offset),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def obter_estatisticas():
    conn = conectar()
    cursor = conn.cursor()
    total_moradores = cursor.execute("SELECT COUNT(*) FROM moradores WHERE status='Ativo'").fetchone()[0]
    total_liberados = cursor.execute(
        "SELECT COUNT(*) FROM historico_acessos WHERE status_acesso='Liberado'"
    ).fetchone()[0]
    total_negados = cursor.execute(
        "SELECT COUNT(*) FROM historico_acessos WHERE status_acesso='Negado'"
    ).fetchone()[0]
    conn.close()
    return total_moradores, total_liberados, total_negados
