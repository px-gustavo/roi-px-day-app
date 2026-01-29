# app.py — ROI PX Day (agora por Nome, somando CNPJs)
# Visão: UMA LINHA POR NOME (somando CNPJs do cliente) + Expander de Diagnóstico
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import unicodedata
import re
from typing import Optional, Tuple, List

st.set_page_config(page_title="ROI PX Day — Relatório (por Nome)", page_icon="📊", layout="wide")

# ==============================
# Config / Regex pré-compilados
# ==============================
SUFIXOS_EXCLUIR = [
    r"LTDA", r"S\.?A\.?", r"EIRELI", r"ME", r"MEI",
    r"TRANSPORTES", r"LOGISTICA", r"COMERCIO", r"INDUSTRIA",
    r"TRANSPORTADORA", r"OPERADOR[AE]? LOG[ÍI]STIC[OA]",
    r"OPERA(C|Ç)ÕES LOG[ÍI]STIC[OA]S?"
]
SUFIXOS_RE = re.compile(r"\b(?:" + "|".join(SUFIXOS_EXCLUIR) + r")\b", flags=re.IGNORECASE)
CNPJ_RE = re.compile(r"\d{2}\.??\d{3}\.??\d{3}/??\d{4}-??\d{2}")

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_name(s: str) -> str:
    """
    Normaliza nomes: remove acentos, sufixos comuns (LTDA/SA/etc), símbolos indesejados.
    Mantém números e alguns símbolos úteis (/ & . -).
    """
    s = strip_accents(s).upper().strip()
    s = SUFIXOS_RE.sub(" ", s)
    s = re.sub(r"[^A-Z0-9/&.\- ]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\bS A\b", "SA", s)
    s = re.sub(r"\bS\/A\b", "SA", s)
    return s

# ==============================
# Funções utilitárias de leitura
# ==============================
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_norm = {strip_accents(str(c)).lower(): c for c in df.columns}
    for cand in candidates:
        key = strip_accents(cand).lower()
        if key in cols_norm:
            return cols_norm[key]
    return None

@st.cache_data(show_spinner=False)
def read_any_csv_bytes(data: bytes) -> pd.DataFrame:
    last_err = None
    bio = BytesIO(data)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (";", ",", "\t", "|"):
            try:
                bio.seek(0)
                df = pd.read_csv(bio, sep=sep, encoding=enc, engine="python")
                if df.empty or all(str(c).startswith("Unnamed") for c in df.columns):
                    continue
                return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Falha ao ler CSV. Último erro: {last_err}")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def parse_mes_col(df: pd.DataFrame, col: str = "MES") -> pd.DataFrame:
    out = df.copy()
    tried = False
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            out[col] = pd.to_datetime(out[col], format=fmt, errors="raise")
            tried = True
            break
        except Exception:
            pass
    if not tried:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    try:
        out[col] = out[col].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out["AnoMes"] = out[col].dt.strftime("%Y-%m")
    out["Ano"] = out[col].dt.year
    out["MesNum"] = out[col].dt.month
    return out

def clean_num_series(s: pd.Series) -> pd.Series:
    def clean_val(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip()
        if t == "":
            return np.nan
        neg = False
        if t.startswith("(") and t.endswith(")"):
            neg = True
            t = t[1:-1]
        t = re.sub(r"[^0-9,.\-]", "", t)
        t = t.replace(".", "").replace(",", ".")
        try:
            val = float(t) if t not in ("", ".", "-", ",") else np.nan
        except Exception:
            val = np.nan
        return -val if neg else val
    return s.apply(clean_val).astype(float)

def trimestre_str(m: int) -> str:
    return "Q1" if m in (1,2,3) else ("Q2" if m in (4,5,6) else ("Q3" if m in (7,8,9) else "Q4"))

def media_trimestral_visita(agr_mes: pd.DataFrame, visit_month_str: str) -> Tuple[float, str]:
    if not visit_month_str:
        return np.nan, ""
    try:
        visit_dt = pd.to_datetime(visit_month_str + "-01")
    except Exception:
        return np.nan, ""
    ano, m = visit_dt.year, visit_dt.month
    if m in (1,2,3): meses_q = [1,2,3]
    elif m in (4,5,6): meses_q = [4,5,6]
    elif m in (7,8,9): meses_q = [7,8,9]
    else: meses_q = [10,11,12]
    mask = (agr_mes["Ano"] == ano) & (agr_mes["MesNum"].isin(meses_q))
    vals = agr_mes.loc[mask, "DIAS DE CONTRATO"].astype(float)
    media = float(vals.mean()) if not vals.empty else np.nan
    return media, f"{trimestre_str(m)}-{ano}"

def detectar_colunas_visitas(dfv: pd.DataFrame) -> Tuple[str, str]:
    col_cli = find_column(dfv, ["cliente", "nome transportadora(s)", "nome", "transportadora", "transportadoras"])
    if not col_cli:
        col_cli = dfv.columns[0]
    col_dt = find_column(dfv, ["datavisita", "data visita", "visita", "data", "mesvisita", "mês da visita", "mes"])
    if not col_dt:
        raise ValueError("A base de visitas precisa ter uma coluna com a data/mês da visita (ex.: 'Data Visita').")
    return col_cli, col_dt

def preparar_visitas(dfv: pd.DataFrame) -> pd.DataFrame:
    dfv = normalize_cols(dfv)
    col_cli, col_dt = detectar_colunas_visitas(dfv)
    out = dfv[[col_cli, col_dt]].copy()
    out.columns = ["Cliente", "DataVisita"]
    out["Cliente_norm"] = out["Cliente"].astype(str).map(normalize_name)

    def to_ym(s):
        s = str(s).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%Y", "%Y-%m", "%d-%m-%Y"):
            try:
                dt = pd.to_datetime(s, format=fmt, errors="raise")
                return dt.strftime("%Y-%m")
            except Exception:
                pass
        dt = pd.to_datetime(s, errors="coerce")
        return dt.strftime("%Y-%m") if pd.notna(dt) else None

    out["VisitMonth"] = out["DataVisita"].map(to_ym)
    out = out.dropna(subset=["VisitMonth"]).drop_duplicates(subset=["Cliente_norm"], keep="last")
    return out[["Cliente", "Cliente_norm", "VisitMonth"]]

# =====================================
# UI — uploads e parâmetros do relatório
# =====================================
st.title("ROI PX Day — Relatório (por Nome, somando CNPJs)")

col1, col2 = st.columns(2)
with col1:
    comportamento_file = st.file_uploader("📥 Base Mensal — comportamento (CSV)", type=["csv"])
with col2:
    visitas_file = st.file_uploader("🎯 Base de Visitas PX Day — clientes e data (CSV)", type=["csv"])

with st.expander("⚙️ Parâmetros (opcional)"):
    meses_janela = st.number_input("Últimos N meses para a visão mensal", min_value=3, max_value=24, value=6, step=1)

if st.button("🚀 Gerar relatório", type="primary", use_container_width=True):
    if not comportamento_file or not visitas_file:
        st.warning("Envie os dois arquivos CSV para continuar.")
        st.stop()

    # ---------- Ler bases ----------
    try:
        df = read_any_csv_bytes(comportamento_file.getvalue())
        dfv = read_any_csv_bytes(visitas_file.getvalue())
    except Exception as e:
        st.error(f"Erro ao ler os arquivos: {e}")
        st.stop()

    # ---------- Padronizar base mensal ----------
    try:
        df = normalize_cols(df)

        # MES
        if "MES" not in df.columns:
            cand = find_column(df, ["mes", "mês", "data", "date"])
            if cand:
                df = df.rename(columns={cand: "MES"})
        if "MES" not in df.columns:
            raise ValueError("Não encontrei a coluna de data do mês (ex.: 'MES').")

        # NOME TRANSPORTADORA(S)
        nome_col = find_column(df, ["nome transportadora(s)", "cliente", "nome", "transportadora", "transportadoras"])
        if nome_col is None:
            raise ValueError("Não encontrei a coluna de cliente (ex.: 'NOME TRANSPORTADORA(S)').")
        if nome_col != "NOME TRANSPORTADORA(S)":
            df = df.rename(columns={nome_col: "NOME TRANSPORTADORA(S)"})

        # DIAS DE CONTRATO
        dias_col = find_column(df, ["dias de contrato", "dias contrato", "dias", "diascontrato"])
        if dias_col is None:
            raise ValueError("Não encontrei a coluna 'DIAS DE CONTRATO'.")
        if dias_col != "DIAS DE CONTRATO":
            df = df.rename(columns={dias_col: "DIAS DE CONTRATO"})

        # normalizações
        df = parse_mes_col(df, col="MES")
        df["DIAS DE CONTRATO"] = clean_num_series(df["DIAS DE CONTRATO"])
        df["Cliente_norm"] = df["NOME TRANSPORTADORA(S)"].astype(str).map(normalize_name)

        # CNPJ (opcional/robusto)
        cnpj_col = find_column(df, ["cnpj", "cnpj cliente", "cpf/cnpj", "cpf"])
        if cnpj_col and cnpj_col != "CNPJ":
            df = df.rename(columns={cnpj_col: "CNPJ"})
        if "CNPJ" not in df.columns:
            df["CNPJ"] = "__SEM_CNPJ__"
        df["CNPJ"] = df["CNPJ"].astype(str).str.strip().replace("", "__SEM_CNPJ__")

        # ESTADO (garante existência)
        if "ESTADO" not in df.columns:
            df["ESTADO"] = np.nan

    except Exception as e:
        st.error(f"Erro ao padronizar a base mensal: {e}")
        st.stop()

    # ---------- Padronizar base de visitas ----------
    try:
        visitas = preparar_visitas(dfv)
    except Exception as e:
        st.error(f"Erro na base de visitas: {e}")
        st.stop()

    # ---------- Último mês fechado ----------
    if df["MES"].notna().any():
        try:
            ultimo_mes_fechado = df["MES"].max().to_period("M").to_timestamp("M")
            current_month_str = ultimo_mes_fechado.strftime("%Y-%m")
        except Exception:
            st.error("Não foi possível determinar o último mês fechado a partir da coluna MES.")
            st.stop()
    else:
        st.error("A coluna MES não contém datas válidas.")
        st.stop()

    # ---------- Janela de meses N ----------
    mesesN = [p.strftime("%Y-%m") for p in pd.period_range(end=pd.Period(current_month_str, freq="M"),
                                                           periods=meses_janela)]

    # ---------- Diagnóstico ----------
    visitas_pre = visitas.copy()
    clientes_base = set(df["Cliente_norm"].dropna().unique())
    visitas = visitas[visitas["Cliente_norm"].isin(clientes_base)].copy()

    nao_casaram = sorted(list(set(visitas_pre["Cliente_norm"]) - clientes_base))
    amostra_match = sorted(list(set(visitas_pre["Cliente_norm"]).intersection(clientes_base)))[:10]

    amostras_cnpjs = []
    for cn in amostra_match:
        dcli = df[df["Cliente_norm"] == cn]
        cnpjs = sorted(dcli["CNPJ"].astype(str).unique().tolist())
        amostras_cnpjs.append({"Cliente_norm": cn, "Qtde CNPJs": len(cnpjs), "Exemplo CNPJs": "; ".join(cnpjs[:5])})

    with st.expander("🔍 Diagnóstico"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Visitas (arquivo)", value=f"{len(visitas_pre):,}")
        with c2:
            st.metric("Visitas com match por nome", value=f"{len(visitas):,}")
        with c3:
            st.metric("Clientes base (normalizados)", value=f"{len(clientes_base):,}")

        if nao_casaram:
            st.warning(f"Sem correspondência por nome (normalizado): {', '.join(nao_casaram[:20])}"
                       + (" ..." if len(nao_casaram) > 20 else ""))
        else:
            st.success("Todas as visitas encontraram correspondência por nome normalizado.")

        if amostras_cnpjs:
            st.markdown("**Amostra de clientes com match e seus CNPJs:**")
            st.dataframe(pd.DataFrame(amostras_cnpjs), use_container_width=True)

    # ---------- Processar por cliente (UMA LINHA POR NOME, somando CNPJs) ----------
    linhas = []
    for _, rowv in visitas.iterrows():
        cliente_raw = rowv["Cliente"]
        cliente_norm = rowv["Cliente_norm"]
        visit_month = rowv["VisitMonth"]

        dcli = df[df["Cliente_norm"] == cliente_norm].copy()
        if dcli.empty:
            continue

        # CNPJs desse nome (para transparência no output)
        cnpjs_cliente = dcli["CNPJ"].dropna().astype(str).unique().tolist()
        cnpjs_cliente = [c for c in cnpjs_cliente if c != "__SEM_CNPJ__"]
        qtde_cnpjs = len(cnpjs_cliente)
        exemplo_cnpjs = "; ".join(sorted(cnpjs_cliente)[:5])

        # AGREGA por Nome+AnoMes: soma DIAS DE CONTRATO de todos os CNPJs do cliente no mês
        agg = dcli.groupby("AnoMes", as_index=False).agg({
            "DIAS DE CONTRATO": "sum",
        })
        # Status: mantém o "último" do mês entre os registros daquele nome
        estado_mes = dcli.groupby("AnoMes", as_index=False)["ESTADO"].last()
        agr = pd.merge(agg, estado_mes, on="AnoMes", how="left")

        # Campos auxiliares + baseline trimestral (por Nome)
        agr["Ano"] = pd.to_datetime(agr["AnoMes"]).dt.year
        agr["MesNum"] = pd.to_datetime(agr["AnoMes"]).dt.month
        baseline, rot_trim = media_trimestral_visita(agr_mes=agr, visit_month_str=visit_month)

        # Status no mês da visita e no atual
        status_visit = agr.loc[agr["AnoMes"] == visit_month, "ESTADO"]
        status_visit_val = status_visit.iloc[0] if not status_visit.empty else np.nan

        cur_series = agr.loc[agr["AnoMes"] == current_month_str, "DIAS DE CONTRATO"]
        current_val = float(cur_series.iloc[0]) if not cur_series.empty else np.nan
        status_current_series = agr.loc[agr["AnoMes"] == current_month_str, "ESTADO"]
        status_current = status_current_series.iloc[0] if not status_current_series.empty else np.nan

        # Impacto (por Nome, já somado)
        impacto_dias, impacto_pct = np.nan, np.nan
        if visit_month and (not np.isnan(baseline)) and (not np.isnan(current_val)):
            impacto_dias = current_val - baseline
            if baseline != 0 and not np.isnan(baseline):
                impacto_pct = impacto_dias / baseline * 100.0

        # Série últimos N meses (por Nome)
        serieN = {}
        for m in mesesN:
            val_series = agr.loc[agr["AnoMes"] == m, "DIAS DE CONTRATO"]
            serieN[m] = float(val_series.iloc[0]) if not val_series.empty else 0.0
        mediaN = float(np.mean(list(serieN.values()))) if len(serieN) > 0 else np.nan

        obs = ""
        if visit_month == current_month_str:
            obs = "Sem mês completo pós-visita (visita no mês do 'Atual')"

        linha = {
            "Cliente": cliente_raw,
            "Cliente_norm": cliente_norm,
            "Visit Month": visit_month,
            "Visita: Trimestre": rot_trim,
            "Baseline (visit quarter avg)": baseline,
            f"Atual ({current_month_str})": current_val,
            "Impacto (dias)": impacto_dias,
            "Impacto (%)": impacto_pct,
            "Status (visita)": status_visit_val,
            f"Status ({current_month_str})": status_current,
            f"Média {meses_janela}m": mediaN,
            "Qtde CNPJs agregados": qtde_cnpjs,
            "Exemplo CNPJs": exemplo_cnpjs,
            "Observação": obs,
        }
        linha.update(serieN)
        linhas.append(linha)

    # ---------- Monta saídas ----------
    if len(linhas) == 0:
        st.warning("Nenhum cliente (por Nome) com visita encontrou correspondência na base mensal após normalização.")
        st.stop()

    resumo = pd.DataFrame(linhas)

    # Pivot mensal (últimos N meses) — por NOME (somando CNPJs)
    dfN = df[df["AnoMes"].isin(mesesN)].copy()
    pivot = (
        dfN.pivot_table(
            index=["NOME TRANSPORTADORA(S)"],
            columns="AnoMes",
            values="DIAS DE CONTRATO",
            aggfunc="sum"
        )
        .fillna(0.0)
        .reset_index()
    )

    st.success(f"Relatório gerado. {len(resumo):,} linhas (1 por Nome, somando CNPJs).", icon="✅")
    st.dataframe(resumo, use_container_width=True)

    # ===== Downloads =====
    # CSV (Excel-friendly)
    csv_bytes = resumo.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "💾 Baixar CSV (Resumo por Nome)",
        data=csv_bytes,
        file_name=f"ROI_PX_Day_Resumo_por_Nome_{current_month_str}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Excel com duas abas
    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as wr:
        resumo.to_excel(wr, sheet_name=f"Resumo_Nome_{meses_janela}m", index=False)
        pivot.to_excel(wr, sheet_name=f"Mensal_por_Nome_{meses_janela}m", index=False)
    xbuf.seek(0)
    st.download_button(
        "📘 Baixar Excel (2 abas)",
        data=xbuf.getvalue(),
        file_name=f"ROI_PX_Day_relatorio_por_Nome_{current_month_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
