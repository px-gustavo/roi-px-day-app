
# app.py — ROI PX Day (2 uploads, sem gráficos)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="ROI PX Day — Relatório", page_icon="📊", layout="wide")

# ==============================
# Funções utilitárias de leitura
# ==============================
def read_any_csv(uploaded_file) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (";", ",", "\t", "|"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, encoding=enc, engine="python")
                # heurística: descartar linhas totalmente vazias
                if df.empty or all(col.startswith("Unnamed") for col in df.columns):
                    continue
                return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Falha ao ler CSV. Último erro: {last_err}")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_mes_col(df: pd.DataFrame, col: str = "MES") -> pd.DataFrame:
    """Converte coluna MES para datetime (ignora hora) e cria AnoMes/Ano/MesNum."""
    out = df.copy()
    # tenta dd/mm/aaaa, depois yyyy-mm-dd e parsing geral
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            out[col] = pd.to_datetime(out[col], format=fmt, errors="raise")
            break
        except Exception:
            pass
    else:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out[col] = out[col].dt.tz_localize(None)
    out["AnoMes"] = out[col].dt.strftime("%Y-%m")
    out["Ano"] = out[col].dt.year
    out["MesNum"] = out[col].dt.month
    return out

def clean_num_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def trimestre_str(m: int) -> str:
    return "Q1" if m in (1,2,3) else ("Q2" if m in (4,5,6) else ("Q3" if m in (7,8,9) else "Q4"))

def media_trimestral_visita(agr_mes: pd.DataFrame, visit_month_str: str) -> tuple[float, str]:
    """
    Média dos DIAS DE CONTRATO no trimestre civil da visita.
    Retorna (media, "Qx-YYYY"). Logica compatível com seu script.  # fonte: script original
    """
    if not visit_month_str:
        return np.nan, ""
    try:
        visit_dt = pd.to_datetime(visit_month_str + "-01")
    except Exception:
        return np.nan, ""
    ano, m = visit_dt.year, visit_dt.month
    meses_q = [1,2,3] if m in (1,2,3) else ([4,5,6] if m in (4,5,6) else ([7,8,9] if m in (7,8,9) else [10,11,12]))
    mask = (agr_mes["Ano"] == ano) & (agr_mes["MesNum"].isin(meses_q))
    vals = agr_mes.loc[mask, "DIAS DE CONTRATO"].astype(float)
    media = float(vals.mean()) if not vals.empty else np.nan
    return media, f"{trimestre_str(m)}-{ano}"

def detectar_colunas_visitas(dfv: pd.DataFrame) -> tuple[str, str]:
    cols = [c.lower() for c in dfv.columns]
    # Nome do cliente
    if "cliente" in cols:
        col_cli = dfv.columns[cols.index("cliente")]
    elif "nome transportadora(s)" in cols:
        col_cli = dfv.columns[cols.index("nome transportadora(s)")]
    else:
        col_cli = dfv.columns[0]  # primeira coluna
    # Data da visita
    for cand in ("datavisita", "data visita", "visita", "data", "mesvisita", "mês da visita"):
        if cand in cols:
            col_dt = dfv.columns[cols.index(cand)]
            break
    else:
        raise ValueError("A base de visitas precisa ter uma coluna com a data/mês da visita (ex.: 'DataVisita').")
    return col_cli, col_dt

def preparar_visitas(dfv: pd.DataFrame) -> pd.DataFrame:
    dfv = normalize_cols(dfv)
    col_cli, col_dt = detectar_colunas_visitas(dfv)
    out = dfv[[col_cli, col_dt]].copy()
    out.columns = ["Cliente", "DataVisita"]
    # normaliza cliente
    out["Cliente_norm"] = out["Cliente"].astype(str).str.strip().str.upper()
    # normaliza data -> AnoMes
    # aceita "aaaa-mm", "mm/aaaa", data completa
    def to_ym(s):
        s = str(s).strip()
        # formatos comuns
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%Y", "%Y-%m"):
            try:
                dt = pd.to_datetime(s, format=fmt, errors="raise")
                return dt.strftime("%Y-%m")
            except Exception:
                pass
        # fallback
        dt = pd.to_datetime(s, errors="coerce")
        return dt.strftime("%Y-%m") if pd.notna(dt) else None
    out["VisitMonth"] = out["DataVisita"].map(to_ym)
    out = out.dropna(subset=["VisitMonth"]).drop_duplicates(subset=["Cliente_norm"], keep="last")
    return out[["Cliente", "Cliente_norm", "VisitMonth"]]

# =====================================
# UI — uploads e parâmetros do relatório
# =====================================
st.title("ROI PX Day — Relatório sem gráficos")

col1, col2 = st.columns(2)
with col1:
    comportamento_file = st.file_uploader("📥 Base Mensal — comportamento (CSV)", type=["csv"])
with col2:
    visitas_file = st.file_uploader("🎯 Base de Visitas PX Day — clientes e data (CSV)", type=["csv"])

with st.expander("⚙️ Parâmetros (opcional)"):
    meses_janela = st.number_input("Últimos N meses para a visão mensal", min_value=3, max_value=12, value=6, step=1)

if st.button("🚀 Gerar relatório", type="primary", use_container_width=True):
    if not comportamento_file or not visitas_file:
        st.warning("Envie os dois arquivos CSV para continuar.")
        st.stop()

    # Ler bases
    try:
        df = read_any_csv(comportamento_file)
        dfv = read_any_csv(visitas_file)
    except Exception as e:
        st.error(f"Erro ao ler os arquivos: {e}")
        st.stop()

    try:
        # Normalizações da base mensal — compatível com seu script original
        df = normalize_cols(df)
        # tenta localizar campos chave
        # MES
        if "MES" not in df.columns:
            # tenta variantes
            for cand in ("MÊS", "mes", "data", "Data"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "MES"})
                    break
        # NOME TRANSPORTADORA(S)
        nome_col = None
        for c in df.columns:
            if "transportadora" in c.lower() or "cliente" in c.lower() or "nome" in c.lower():
                nome_col = c; break
        if nome_col is None:
            raise ValueError("Não encontrei a coluna de cliente (ex.: 'NOME TRANSPORTADORA(S)').")
        if nome_col != "NOME TRANSPORTADORA(S)":
            df = df.rename(columns={nome_col: "NOME TRANSPORTADORA(S)"})
        # DIAS DE CONTRATO
        dias_col = None
        for c in df.columns:
            if "dias" in c.lower() and "contrato" in c.lower():
                dias_col = c; break
        if dias_col is None:
            raise ValueError("Não encontrei a coluna 'DIAS DE CONTRATO'.")
        if dias_col != "DIAS DE CONTRATO":
            df = df.rename(columns={dias_col: "DIAS DE CONTRATO"})

        # parse de datas e números (igual ao padrão do seu script)
        df = parse_mes_col(df, col="MES")
        df["DIAS DE CONTRATO"] = clean_num_series(df["DIAS DE CONTRATO"])
    except Exception as e:
        st.error(f"Erro ao padronizar a base mensal: {e}")
        st.stop()

    try:
        visitas = preparar_visitas(dfv)  # Cliente, Cliente_norm, VisitMonth
    except Exception as e:
        st.error(f"Erro na base de visitas: {e}")
        st.stop()

    # Descobrir último mês fechado a partir da base
    if df["MES"].notna().any():
        ultimo_mes_fechado = df["MES"].max().to_period("M").to_timestamp("M")
        current_month_str = ultimo_mes_fechado.strftime("%Y-%m")
    else:
        st.error("A coluna MES não contém datas válidas.")
        st.stop()

    # Janela dos N últimos meses
    mesesN = [p.strftime("%Y-%m") for p in pd.period_range(end=pd.Period(current_month_str, freq="M"), periods=meses_janela)]

    # Processar por cliente visitado
    linhas = []
    clientes_v = visitas["Cliente_norm"].tolist()
    df["Cliente_norm"] = df["NOME TRANSPORTADORA(S)"].astype(str).str.strip().str.upper()

    for _, rowv in visitas.iterrows():
        cliente_raw = rowv["Cliente"]
        cliente_norm = rowv["Cliente_norm"]
        visit_month = rowv["VisitMonth"]

        dcli = df[df["Cliente_norm"] == cliente_norm].copy()
        if dcli.empty:
            # tentativa fuzzy simples: contains
            dcli = df[df["NOME TRANSPORTADORA(S)"].str.contains(cliente_raw, case=False, na=False)]

        if dcli.empty:
            # linha vazia se não houver dados
            linha = {
                "Cliente": cliente_raw,
                "Visit Month": visit_month,
                "Visita: Trimestre": "",
                "Baseline (visit quarter avg)": np.nan,
                f"Atual ({current_month_str})": np.nan,
                "Impacto (dias)": np.nan,
                "Impacto (%)": np.nan,
                "Status (visita)": None,
                f"Status ({current_month_str})": None,
                f"Média {meses_janela}m": np.nan,
                "Observação": "Sem dados na base",
            }
            for m in mesesN:
                linha[m] = 0.0
            linhas.append(linha)
            continue

        # Agrega por AnoMes
        agr = dcli.groupby(["AnoMes"], as_index=False).agg(
            **{
                "DIAS DE CONTRATO": ("DIAS DE CONTRATO", "sum"),
                **({"ESTADO": ("ESTADO", "last")} if "ESTADO" in dcli.columns else {})
            }
        )
        # Prepara campos auxiliares para baseline
        agr["Ano"] = pd.to_datetime(agr["AnoMes"]).dt.year
        agr["MesNum"] = pd.to_datetime(agr["AnoMes"]).dt.month

        # Baseline (trimestre da visita) — mesma lógica do seu código
        baseline, rot_trim = media_trimestral_visita(agr_mes=agr, visit_month_str=visit_month)

        # Status no mês da visita (se existir)
        status_visit = agr.loc[agr["AnoMes"] == visit_month, "ESTADO"].iloc[0] if ("ESTADO" in agr.columns and visit_month in set(agr["AnoMes"])) else None

        # Atual (último mês)
        current_val = float(agr.loc[agr["AnoMes"] == current_month_str, "DIAS DE CONTRATO"].iloc[0]) if current_month_str in set(agr["AnoMes"]) else np.nan
        status_current = agr.loc[agr["AnoMes"] == current_month_str, "ESTADO"].iloc[0] if ("ESTADO" in agr.columns and current_month_str in set(agr["AnoMes"])) else None

        # Impacto
        impacto_dias, impacto_pct = np.nan, np.nan
        if visit_month and (not np.isnan(baseline)) and (not np.isnan(current_val)):
            impacto_dias = current_val - baseline
            if baseline > 0:
                impacto_pct = impacto_dias / baseline * 100.0

        # Série últimos N meses
        serieN = {m: float(agr.loc[agr["AnoMes"] == m, "DIAS DE CONTRATO"].iloc[0]) if m in set(agr["AnoMes"]) else 0.0 for m in mesesN}
        mediaN = float(np.mean(list(serieN.values()))) if len(serieN) > 0 else np.nan

        obs = ""
        # Exemplo da sua observação original: visitas de nov/2025 não têm mês completo pós-visita
        # Mantemos a ideia de avisar se a visita ocorreu no mesmo mês do "Atual"
        if visit_month == current_month_str:
            obs = "Sem mês completo pós-visita (visita no mês do 'Atual')"

        linha = {
            "Cliente": cliente_raw,
            "Visit Month": visit_month,
            "Visita: Trimestre": rot_trim,
            "Baseline (visit quarter avg)": baseline,
            f"Atual ({current_month_str})": current_val,
            "Impacto (dias)": impacto_dias,
            "Impacto (%)": impacto_pct,
            "Status (visita)": status_visit,
            f"Status ({current_month_str})": status_current,
            f"Média {meses_janela}m": mediaN,
            "Observação": obs,
        }
        linha.update(serieN)
        linhas.append(linha)

    resumo = pd.DataFrame(linhas)

    # Pivot mensal para a aba 2 (últimos N meses)
    dfN = df[df["AnoMes"].isin(mesesN)].copy()
    pivot = dfN.pivot_table(index="NOME TRANSPORTADORA(S)", columns="AnoMes", values="DIAS DE CONTRATO", aggfunc="sum").fillna(0.0).reset_index()

    st.success(f"Relatório gerado. {len(resumo):,} linhas.", icon="✅")
    st.dataframe(resumo, use_container_width=True)

    # ===== Downloads =====
    # CSV (Excel-friendly)
    csv_bytes = resumo.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("💾 Baixar CSV (Resumo)", data=csv_bytes, file_name="ROI_PX_Day_Resumo.csv", mime="text/csv", use_container_width=True)

    # Excel com duas abas
    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as wr:
        resumo.to_excel(wr, sheet_name=f"Resumo ({meses_janela}m)", index=False)
        pivot.to_excel(wr, sheet_name=f"Mensal por Cliente ({meses_janela}m)", index=False)
    xbuf.seek(0)
    st.download_button("📘 Baixar Excel (2 abas)", data=xbuf.getvalue(), file_name="ROI_PX_Day_relatorio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
