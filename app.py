
# app.py — ROI PX Day (2 uploads, sem gráficos)
# Visão: UMA LINHA POR CNPJ (sem somar dias entre CNPJs) + Expander de Diagnóstico
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import unicodedata, re

st.set_page_config(page_title="ROI PX Day — Relatório", page_icon="📊", layout="wide")

# ==============================
# Utilitários de normalização
# ==============================
SUFIXOS_EXCLUIR = [
    r"LTDA", r"Ltda", r"S\.?A\.?", r"SA", r"S A", r"EIRELI", r"ME", r"MEI",
    r"TRANSPORTES LTDA", r"TRANSPORTES", r"LOGISTICA", r"LOGÍSTICA",
    r"COMERCIO", r"COMÉRCIO", r"INDUSTRIA", r"INDÚSTRIA", r"EIRELI - ME",
    r"TRANSPORTADORA", r"OPERADOR[AE]? LOG[ÍI]STIC[OA]", r"OPERA(C|Ç)ÕES LOG[ÍI]STIC[OA]S?"
]

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_name(s: str) -> str:
    s = strip_accents(s).upper().strip()
    for suf in SUFIXOS_EXCLUIR:
        s = re.sub(rf"\b{suf}\b", " ", s, flags=re.IGNORECASE)
    # remove símbolos estranhos (mantém / & . - e espaço)
    s = re.sub(r"[^A-Z0-9/&.\- ]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # padroniza variações
    s = re.sub(r"\bS A\b", "SA", s)
    s = re.sub(r"\bS\/A\b", "SA", s)
    return s

# ==============================
# Funções utilitárias de leitura
# ==============================
def read_any_csv(uploaded_file) -> pd.DataFrame:
    """
    Tenta ler um CSV com diferentes encodings/separadores.
    """
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (";", ",", "\t", "|"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=sep, encoding=enc, engine="python")
                # ignora leituras claramente inválidas
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
    """
    Converte coluna MES para datetime (ignora hora) e cria AnoMes/Ano/MesNum.
    Aceita dd/mm/aaaa, yyyy-mm-dd, ISO com timezone, etc.
    """
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
        out[col] = pd.to_datetime(out[col], errors="coerce")  # ISO, tz, etc.
    # remove timezone se existir (mas não quebra se já for naive)
    try:
        out[col] = out[col].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out["AnoMes"] = out[col].dt.strftime("%Y-%m")
    out["Ano"] = out[col].dt.year
    out["MesNum"] = out[col].dt.month
    return out

def clean_num_series(s: pd.Series) -> pd.Series:
    """
    Limpa números no padrão PT-BR:
    - remove separador de milhar (.)
    - troca vírgula decimal (,) por ponto (.)
    """
    s = s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def trimestre_str(m: int) -> str:
    return "Q1" if m in (1,2,3) else ("Q2" if m in (4,5,6) else ("Q3" if m in (7,8,9) else "Q4"))

def media_trimestral_visita(agr_mes: pd.DataFrame, visit_month_str: str) -> tuple[float, str]:
    """
    Média dos DIAS DE CONTRATO no trimestre civil da visita (para aquele CNPJ).
    Retorna (media, "Qx-YYYY").
    """
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

def detectar_colunas_visitas(dfv: pd.DataFrame) -> tuple[str, str]:
    cols = [c.lower() for c in dfv.columns]
    # nome
    if "cliente" in cols:
        col_cli = dfv.columns[cols.index("cliente")]
    elif "nome transportadora(s)" in cols:
        col_cli = dfv.columns[cols.index("nome transportadora(s)")]
    else:
        col_cli = dfv.columns[0]
    # data
    for cand in ("datavisita", "data visita", "visita", "data", "mesvisita", "mês da visita"):
        if cand in cols:
            col_dt = dfv.columns[cols.index(cand)]
            break
    else:
        raise ValueError("A base de visitas precisa ter uma coluna com a data/mês da visita (ex.: 'Data Visita').")
    return col_cli, col_dt

def preparar_visitas(dfv: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza a base de visitas, gera Cliente_norm (normalizado) e VisitMonth (YYYY-MM).
    Dedup: mantem última por Cliente_norm (caso enviado mais de uma visita).
    """
    dfv = normalize_cols(dfv)
    col_cli, col_dt = detectar_colunas_visitas(dfv)
    out = dfv[[col_cli, col_dt]].copy()
    out.columns = ["Cliente", "DataVisita"]

    # normaliza cliente
    out["Cliente_norm"] = out["Cliente"].astype(str).map(normalize_name)

    # normaliza DataVisita -> AnoMes (YYYY-MM)
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
st.title("ROI PX Day — Relatório (sem gráficos, 1 linha por CNPJ)")

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

    # ---------- Ler bases ----------
    try:
        df = read_any_csv(comportamento_file)
        dfv = read_any_csv(visitas_file)
    except Exception as e:
        st.error(f"Erro ao ler os arquivos: {e}")
        st.stop()

    # ---------- Padronizar base mensal ----------
    try:
        df = normalize_cols(df)

        # MES
        if "MES" not in df.columns:
            for cand in ("MÊS", "mes", "data", "Data", "Mes"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "MES"})
                    break
        if "MES" not in df.columns:
            raise ValueError("Não encontrei a coluna de data do mês (ex.: 'MES').")

        # NOME TRANSPORTADORA(S)
        nome_col = None
        for c in df.columns:
            cl = c.lower()
            if "transportadora" in cl or "cliente" in cl or "nome" in cl:
                nome_col = c; break
        if nome_col is None:
            raise ValueError("Não encontrei a coluna de cliente (ex.: 'NOME TRANSPORTADORA(S)').")
        if nome_col != "NOME TRANSPORTADORA(S)":
            df = df.rename(columns={nome_col: "NOME TRANSPORTADORA(S)"})

        # DIAS DE CONTRATO
        dias_col = None
        for c in df.columns:
            cl = c.lower()
            if "dias" in cl and "contrato" in cl:
                dias_col = c; break
        if dias_col is None:
            raise ValueError("Não encontrei a coluna 'DIAS DE CONTRATO'.")
        if dias_col != "DIAS DE CONTRATO":
            df = df.rename(columns={dias_col: "DIAS DE CONTRATO"})

        # normalizações
        df = parse_mes_col(df, col="MES")
        df["DIAS DE CONTRATO"] = clean_num_series(df["DIAS DE CONTRATO"])
        df["Cliente_norm"] = df["NOME TRANSPORTADORA(S)"].astype(str).map(normalize_name)

        # --- CNPJ robusto (rename + fallback) ---
        cnpj_col = next((c for c in df.columns if "cnpj" in c.lower()), None)
        if cnpj_col and cnpj_col != "CNPJ":
            df = df.rename(columns={cnpj_col: "CNPJ"})
        if not cnpj_col:
            df["CNPJ"] = "__SEM_CNPJ__"   # quando a base não traz CNPJ
        df["CNPJ"] = df["CNPJ"].astype(str).str.strip()

    except Exception as e:
        st.error(f"Erro ao padronizar a base mensal: {e}")
        st.stop()

    # ---------- Padronizar base de visitas ----------
    try:
        visitas = preparar_visitas(dfv)  # colunas: Cliente, Cliente_norm, VisitMonth
    except Exception as e:
        st.error(f"Erro na base de visitas: {e}")
        st.stop()

    # ---------- Determinar último mês fechado ----------
    if df["MES"].notna().any():
        ultimo_mes_fechado = df["MES"].max().to_period("M").to_timestamp("M")
        current_month_str = ultimo_mes_fechado.strftime("%Y-%m")
    else:
        st.error("A coluna MES não contém datas válidas.")
        st.stop()

    # ---------- Janela de meses N ----------
    mesesN = [p.strftime("%Y-%m") for p in pd.period_range(end=pd.Period(current_month_str, freq="M"),
                                                           periods=meses_janela)]

    # ---------- Diagnóstico (antes do loop) ----------
    # Match por nome normalizado entre visitas e base mensal
    visitas_pre = visitas.copy()
    clientes_base = set(df["Cliente_norm"].dropna().unique())
    visitas = visitas[visitas["Cliente_norm"].isin(clientes_base)].copy()

    nao_casaram = sorted(list(set(visitas_pre["Cliente_norm"]) - clientes_base))
    amostra_match = sorted(list(set(visitas["Cliente_norm"]).intersection(clientes_base)))[:10]

    # Quantidade de CNPJs por cliente com match (amostra)
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

    # ---------- Processar por cliente (UMA LINHA POR CNPJ) ----------
    linhas = []
    for _, rowv in visitas.iterrows():
        cliente_raw = rowv["Cliente"]
        cliente_norm = rowv["Cliente_norm"]
        visit_month = rowv["VisitMonth"]

        dcli = df[df["Cliente_norm"] == cliente_norm].copy()
        if dcli.empty:
            # sem match? ignora (não entra no relatório)
            continue

        # Todos os CNPJs desse cliente (nome)
        cnpjs_cliente = (
            dcli["CNPJ"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not cnpjs_cliente:
            cnpjs_cliente = ["__SEM_CNPJ__"]

        for cnpj in sorted(cnpjs_cliente):
            dcnpj = dcli[dcli["CNPJ"] == cnpj].copy()
            if dcnpj.empty:
                continue

            # AGREGA por CNPJ+AnoMes: soma dias dentro do MES para aquele CNPJ (filiais/duplicatas)
            # OBS.: não soma entre CNPJs diferentes — cada CNPJ sai em linha própria.
            agg = dcnpj.groupby("AnoMes", as_index=False).agg({
                "DIAS DE CONTRATO": "sum",     # apenas dentro do próprio CNPJ
            })
            # se existir coluna de status, mantém o "último" do mês (comportamento original por CNPJ)
            if "ESTADO" in dcnpj.columns:
                estado_mes = dcnpj.groupby("AnoMes", as_index=False)["ESTADO"].last()
                agr = pd.merge(agg, estado_mes, on="AnoMes", how="left")
            else:
                agr = agg
                agr["ESTADO"] = None

            # Campos auxiliares e baseline por CNPJ
            agr["Ano"] = pd.to_datetime(agr["AnoMes"]).dt.year
            agr["MesNum"] = pd.to_datetime(agr["AnoMes"]).dt.month
            baseline, rot_trim = media_trimestral_visita(agr_mes=agr, visit_month_str=visit_month)

            # Status (visita) e (atual) do CNPJ
            status_visit = agr.loc[agr["AnoMes"] == visit_month, "ESTADO"].iloc[0] \
                if (visit_month in set(agr["AnoMes"])) else None
            current_val = float(agr.loc[agr["AnoMes"] == current_month_str, "DIAS DE CONTRATO"].iloc[0]) \
                if current_month_str in set(agr["AnoMes"]) else np.nan
            status_current = agr.loc[agr["AnoMes"] == current_month_str, "ESTADO"].iloc[0] \
                if (current_month_str in set(agr["AnoMes"])) else None

            # Impacto por CNPJ
            impacto_dias, impacto_pct = np.nan, np.nan
            if visit_month and (not np.isnan(baseline)) and (not np.isnan(current_val)):
                impacto_dias = current_val - baseline
                if baseline > 0:
                    impacto_pct = impacto_dias / baseline * 100.0

            # Série últimos N meses (por CNPJ)
            serieN = {
                m: float(agr.loc[agr["AnoMes"] == m, "DIAS DE CONTRATO"].iloc[0]) if m in set(agr["AnoMes"]) else 0.0
                for m in mesesN
            }
            mediaN = float(np.mean(list(serieN.values()))) if len(serieN) > 0 else np.nan

            obs = ""
            if visit_month == current_month_str:
                obs = "Sem mês completo pós-visita (visita no mês do 'Atual')"

            linha = {
                "Cliente": cliente_raw,
                "Cliente_norm": cliente_norm,
                "CNPJ": cnpj,
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

    # ---------- Monta saídas ----------
    if len(linhas) == 0:
        st.warning("Nenhum CNPJ com visita encontrou correspondência na base mensal após normalização.")
        st.stop()

    resumo = pd.DataFrame(linhas)

    # Pivot mensal (últimos N meses) — por NOME + CNPJ (sem somar entre CNPJs)
    dfN = df[df["AnoMes"].isin(mesesN)].copy()
    pivot = (
        dfN.pivot_table(
            index=["NOME TRANSPORTADORA(S)", "CNPJ"],
            columns="AnoMes",
            values="DIAS DE CONTRATO",
            aggfunc="sum"
        )
        .fillna(0.0)
        .reset_index()
    )

    st.success(f"Relatório gerado. {len(resumo):,} linhas (1 por CNPJ).", icon="✅")
    st.dataframe(resumo, use_container_width=True)

    # ===== Downloads =====
    # CSV (Excel-friendly)
    csv_bytes = resumo.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "💾 Baixar CSV (Resumo por CNPJ)",
        data=csv_bytes,
        file_name="ROI_PX_Day_Resumo_por_CNPJ.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Excel com duas abas
    xbuf = BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as wr:
        resumo.to_excel(wr, sheet_name=f"Resumo_CNPJ ({meses_janela}m)", index=False)
        pivot.to_excel(wr, sheet_name=f"Mensal por CNPJ ({meses_janela}m)", index=False)
    xbuf.seek(0)
    st.download_button(
        "📘 Baixar Excel (2 abas)",
        data=xbuf.getvalue(),
        file_name="ROI_PX_Day_relatorio_por_CNPJ.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
