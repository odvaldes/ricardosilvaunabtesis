# -*- coding: utf-8 -*-
"""
SIAD - Sistema Inteligente de Apoyo a la Decisión
EXCON | App integrada equivalente a Notebooks 1, 2 y 3 + Dashboard Streamlit.

Ejecución:
    streamlit run SIAD_app_integrada.py

La app acepta:
1) un ZIP con las 7 bases Excel de EXCON, o
2) una carpeta local que contenga esas 7 bases.

Etapas:
- Notebook 1: auditoría, normalización e integración.
- Notebook 2: ingeniería de variables SKU-centro y construcción del target histórico.
- Notebook 3: Random Forest, validación temporal, probabilidad, IRI y recomendación.
- Dashboard: KPI y priorización para apoyo a la decisión.
"""
from __future__ import annotations

import io
import os
import re
import json
import zipfile
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="SIAD | EXCON", page_icon="📦", layout="wide")

ARCHIVOS = {
    "movimientos": "Movs. productos.xlsx",
    "productos": "Productos.xlsx",
    "inventario": "Reporte Inventario.xlsx",
    "lineas_compra": "Lins compra.xlsx",
    "pedidos": "Pedidos compra.xlsx",
    "recepciones": "Historicos recepciones de compra.xlsx",
    "proyectos": "Proyectos.xlsx",
}

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def norm_txt(x):
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().upper()

def norm_col(c):
    s = norm_txt(c).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def nserie(s):
    return pd.to_numeric(s, errors="coerce")

def dserie(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def buscar_col(df, candidatos):
    mapa = {norm_col(c): c for c in df.columns}
    for c in candidatos:
        nc = norm_col(c)
        if nc in mapa:
            return mapa[nc]
    for c in candidatos:
        nc = norm_col(c)
        for k, original in mapa.items():
            if nc in k or k in nc:
                return original
    return None

def leer_excel(ruta, sheet=0, header=0):
    return pd.read_excel(ruta, sheet_name=sheet, header=header, engine="openpyxl")

def inventario_excon(ruta):
    raw = pd.read_excel(ruta, sheet_name="Hoja1", header=None, engine="openpyxl")
    fila = None
    for i in range(min(len(raw), 80)):
        vals = [norm_txt(v) for v in raw.iloc[i].tolist()]
        if "ARTICULO" in vals and "CANTIDAD" in vals:
            fila = i
            break
    if fila is None:
        raise ValueError("No se encontró la fila de encabezados del Reporte Inventario.")
    df = pd.read_excel(ruta, sheet_name="Hoja1", header=fila, engine="openpyxl")
    df.columns = [norm_col(c) for c in df.columns]
    return df.dropna(how="all")

def preparar_fuentes(carpeta):
    carpeta = Path(carpeta)
    faltan = [v for v in ARCHIVOS.values() if not (carpeta / v).exists()]
    if faltan:
        raise FileNotFoundError("Faltan bases: " + ", ".join(faltan))

    mov = leer_excel(carpeta / ARCHIVOS["movimientos"], "Movs. productos")
    prod = leer_excel(carpeta / ARCHIVOS["productos"], "Productos")
    inv = inventario_excon(carpeta / ARCHIVOS["inventario"])
    lin = leer_excel(carpeta / ARCHIVOS["lineas_compra"], "Líns. compra")
    ped = leer_excel(carpeta / ARCHIVOS["pedidos"], "Pedidos compra")
    rec = leer_excel(carpeta / ARCHIVOS["recepciones"], 0)
    proy = leer_excel(carpeta / ARCHIVOS["proyectos"], "Proyectos")
    return {"mov": mov, "prod": prod, "inv": inv, "lin": lin, "ped": ped, "rec": rec, "proy": proy}

# ---------------------------------------------------------------------
# NOTEBOOK 1 - Auditoría e integración
# ---------------------------------------------------------------------
def notebook_1_integracion(fuentes):
    mov = fuentes["mov"].copy()
    prod = fuentes["prod"].copy()
    inv = fuentes["inv"].copy()
    lin = fuentes["lin"].copy()
    ped = fuentes["ped"].copy()
    rec = fuentes["rec"].copy()
    proy = fuentes["proy"].copy()

    # Movimientos
    c_sku = buscar_col(mov, ["Nº producto", "No producto"])
    c_fecha = buscar_col(mov, ["Fecha registro"])
    c_cc = buscar_col(mov, ["Centro de Costo"])
    c_cant = buscar_col(mov, ["Cantidad"])
    c_tipo = buscar_col(mov, ["Tipo movimiento"])
    c_desc = buscar_col(mov, ["Descripción"])
    c_cost = buscar_col(mov, ["Importe costo (Real)", "Importe costo Real"])

    m = pd.DataFrame({
        "sku": mov[c_sku].astype(str).str.strip(),
        "fecha": dserie(mov[c_fecha]),
        "centro_costo": mov[c_cc].fillna("SIN_CC").astype(str).str.strip() if c_cc else "SIN_CC",
        "cantidad_mov": nserie(mov[c_cant]).fillna(0),
        "tipo_movimiento": mov[c_tipo].astype(str) if c_tipo else "",
        "descripcion_mov": mov[c_desc].astype(str) if c_desc else "",
        "costo_mov": nserie(mov[c_cost]).fillna(0) if c_cost else 0,
    })
    m = m[(m["sku"] != "nan") & m["fecha"].notna()].copy()

    # Por convención ERP: cantidad negativa = salida/consumo; positiva = entrada.
    m["salida"] = (-m["cantidad_mov"]).clip(lower=0)
    m["entrada"] = m["cantidad_mov"].clip(lower=0)
    # Respaldo por texto cuando el signo no sea informativo.
    txt = m["tipo_movimiento"].map(norm_txt)
    es_salida = txt.str.contains("SALIDA|CONSUM|NEGATIVE|VENTA", regex=True, na=False)
    es_entrada = txt.str.contains("ENTRADA|COMPRA|POSITIVE|RECEPC", regex=True, na=False)
    m.loc[es_salida & (m["salida"] == 0), "salida"] = m.loc[es_salida & (m["cantidad_mov"] > 0), "cantidad_mov"]
    m.loc[es_entrada & (m["entrada"] == 0), "entrada"] = (-m.loc[es_entrada & (m["cantidad_mov"] < 0), "cantidad_mov"]).clip(lower=0)

    # Maestro productos
    p_sku = buscar_col(prod, ["Nº", "No"])
    p_desc = buscar_col(prod, ["Descripción"])
    p_inv = buscar_col(prod, ["Inventario"])
    p_cat = buscar_col(prod, ["Cód. categoría producto", "Cod categoria producto"])
    p_grupo = buscar_col(prod, ["Grupo registro inventario"])
    p_costo = buscar_col(prod, ["Costo unitario", "Costo ajustado"])
    p = pd.DataFrame({
        "sku": prod[p_sku].astype(str).str.strip(),
        "descripcion": prod[p_desc].astype(str) if p_desc else "",
        "inventario_maestro": nserie(prod[p_inv]) if p_inv else np.nan,
        "categoria": prod[p_cat].astype(str) if p_cat else "",
        "grupo_inventario": prod[p_grupo].astype(str) if p_grupo else "",
        "costo_unitario_maestro": nserie(prod[p_costo]) if p_costo else np.nan,
    }).drop_duplicates("sku")

    # Inventario actual
    i_sku = buscar_col(inv, ["articulo"])
    i_desc = buscar_col(inv, ["descripcion"])
    i_stock = buscar_col(inv, ["cantidad"])
    i_costo = buscar_col(inv, ["costo_unitario"])
    i_valor = buscar_col(inv, ["costo_extendido"])
    i_alm = buscar_col(inv, ["almacen"])
    i_unidad = buscar_col(inv, ["unidad_usuaria"])
    ii = pd.DataFrame({
        "sku": inv[i_sku].astype(str).str.strip(),
        "descripcion_inv": inv[i_desc].astype(str) if i_desc else "",
        "stock_actual": nserie(inv[i_stock]).fillna(0),
        "costo_unitario_actual": nserie(inv[i_costo]) if i_costo else np.nan,
        "valor_stock": nserie(inv[i_valor]).fillna(0) if i_valor else np.nan,
        "almacen": inv[i_alm].astype(str) if i_alm else "",
        "unidad_usuaria": inv[i_unidad].astype(str) if i_unidad else "",
    })
    ii = ii.groupby("sku", as_index=False).agg(
        stock_actual=("stock_actual", "sum"),
        valor_stock=("valor_stock", "sum"),
        costo_unitario_actual=("costo_unitario_actual", "median"),
        descripcion_inv=("descripcion_inv", "first"),
        almacen=("almacen", lambda x: ", ".join(sorted(set(map(str, x.dropna())))[:4])),
        unidad_usuaria=("unidad_usuaria", "first"),
    )

    # Compras
    l_sku = buscar_col(lin, ["Nº", "No"])
    l_fecha = buscar_col(lin, ["Fecha recepción esperada"])
    l_cant = buscar_col(lin, ["Cantidad"])
    l_rec = buscar_col(lin, ["Cantidad recibida"])
    l_pend = buscar_col(lin, ["Cantidad pendiente"])
    l_valor = buscar_col(lin, ["Importe Línea Sin IVA CLP"])
    l_cc = buscar_col(lin, ["Centro responsabilidad"])
    compras = pd.DataFrame({
        "sku": lin[l_sku].astype(str).str.strip(),
        "fecha_compra": dserie(lin[l_fecha]) if l_fecha else pd.NaT,
        "centro_costo": lin[l_cc].fillna("SIN_CC").astype(str).str.strip() if l_cc else "SIN_CC",
        "cantidad_compra": nserie(lin[l_cant]).fillna(0) if l_cant else 0,
        "cantidad_recibida": nserie(lin[l_rec]).fillna(0) if l_rec else 0,
        "cantidad_pendiente": nserie(lin[l_pend]).fillna(0) if l_pend else 0,
        "valor_compra": nserie(lin[l_valor]).fillna(0) if l_valor else 0,
    })
    compras = compras[compras["sku"] != "nan"]

    # Proyectos como diccionario de CC
    pr_n = buscar_col(proy, ["Nº", "No"])
    pr_cc = buscar_col(proy, ["Centro de Costo"])
    pr_desc = buscar_col(proy, ["Descripción"])
    pr = pd.DataFrame({
        "proyecto": proy[pr_n].astype(str) if pr_n else "",
        "centro_costo": proy[pr_cc].astype(str).str.strip() if pr_cc else "",
        "proyecto_desc": proy[pr_desc].astype(str) if pr_desc else "",
    }).drop_duplicates("centro_costo")

    auditoria = pd.DataFrame({
        "base": ["Movimientos", "Productos", "Inventario", "Líneas compra", "Pedidos", "Recepciones", "Proyectos"],
        "filas": [len(mov), len(prod), len(inv), len(lin), len(ped), len(rec), len(proy)],
        "columnas": [mov.shape[1], prod.shape[1], inv.shape[1], lin.shape[1], ped.shape[1], rec.shape[1], proy.shape[1]],
    })
    return {"mov": m, "productos": p, "inventario": ii, "compras": compras, "proyectos": pr, "auditoria": auditoria}

# ---------------------------------------------------------------------
# NOTEBOOK 2 - Ingeniería de variables
# ---------------------------------------------------------------------
def notebook_2_variables(n1):
    mov = n1["mov"].copy()
    prod = n1["productos"].copy()
    inv = n1["inventario"].copy()
    compras = n1["compras"].copy()

    fecha_max = mov["fecha"].max()
    if pd.isna(fecha_max):
        raise ValueError("No existen fechas válidas en movimientos.")

    mov["mes"] = mov["fecha"].dt.to_period("M").dt.to_timestamp()
    mensual = mov.groupby(["sku", "centro_costo", "mes"], as_index=False).agg(
        salida_mes=("salida", "sum"),
        entrada_mes=("entrada", "sum"),
        movimientos_mes=("cantidad_mov", "size"),
    )

    # Panel histórico sólo para combinaciones observadas.
    paneles = []
    for (sku, cc), g in mensual.groupby(["sku", "centro_costo"]):
        ini, fin = g["mes"].min(), g["mes"].max()
        idx = pd.date_range(ini, fin, freq="MS")
        z = g.set_index("mes").reindex(idx).fillna({"salida_mes": 0, "entrada_mes": 0, "movimientos_mes": 0})
        z.index.name = "mes"
        z = z.reset_index()
        z["sku"], z["centro_costo"] = sku, cc
        paneles.append(z)
    panel = pd.concat(paneles, ignore_index=True) if paneles else pd.DataFrame()

    panel = panel.sort_values(["sku", "centro_costo", "mes"])
    grp = panel.groupby(["sku", "centro_costo"], group_keys=False)

    for w in [3, 6, 12]:
        panel[f"consumo_{w}m"] = grp["salida_mes"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        panel[f"entradas_{w}m"] = grp["entrada_mes"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        panel[f"meses_con_consumo_{w}m"] = grp["salida_mes"].transform(lambda s: (s > 0).rolling(w, min_periods=1).sum())

    panel["prom_consumo_12m"] = panel["consumo_12m"] / 12.0
    panel["variabilidad_consumo_12m"] = grp["salida_mes"].transform(lambda s: s.rolling(12, min_periods=2).std()).fillna(0)

    # Meses desde la última salida, sin mirar el futuro.
    def meses_sin_salida(g):
        ult = None
        out = []
        for _, r in g.iterrows():
            if r["salida_mes"] > 0:
                ult = r["mes"]
                out.append(0)
            elif ult is None:
                out.append(np.nan)
            else:
                out.append((r["mes"].year - ult.year) * 12 + r["mes"].month - ult.month)
        return pd.Series(out, index=g.index)
    panel["meses_sin_salida"] = grp.apply(meses_sin_salida).reset_index(level=[0,1], drop=True)

    # Stock aproximado histórico: flujo acumulado por combinación.
    panel["stock_flujo"] = grp.apply(
        lambda g: (g["entrada_mes"] - g["salida_mes"]).cumsum()
    ).reset_index(level=[0,1], drop=True)
    panel["stock_estimado"] = panel["stock_flujo"].clip(lower=0)
    panel["cobertura_meses"] = np.where(
        panel["prom_consumo_12m"] > 0,
        panel["stock_estimado"] / panel["prom_consumo_12m"],
        np.where(panel["stock_estimado"] > 0, 99.0, 0.0)
    )
    panel["rotacion_12m"] = np.where(panel["stock_estimado"] > 0, panel["consumo_12m"] / panel["stock_estimado"], 0)

    # Target: con stock positivo y SIN consumo durante los 12 meses siguientes.
    futuro = grp["salida_mes"].transform(
        lambda s: s.shift(-1)[::-1].rolling(12, min_periods=12).sum()[::-1]
    )
    panel["consumo_futuro_12m"] = futuro
    panel["inmovilizado_12m"] = np.where(
        panel["consumo_futuro_12m"].notna(),
        ((panel["stock_estimado"] > 0) & (panel["consumo_futuro_12m"] <= 0)).astype(int),
        np.nan
    )

    # Scoring actual: último mes por SKU-CC + stock real actual por SKU.
    scoring = panel.sort_values("mes").groupby(["sku", "centro_costo"], as_index=False).tail(1).copy()
    scoring = scoring.merge(inv, on="sku", how="left").merge(prod, on="sku", how="left")
    scoring["stock_actual"] = scoring["stock_actual"].fillna(scoring["stock_estimado"]).fillna(0)
    scoring["valor_stock"] = scoring["valor_stock"].fillna(
        scoring["stock_actual"] * scoring["costo_unitario_actual"].fillna(scoring["costo_unitario_maestro"])
    )
    scoring["cobertura_meses"] = np.where(
        scoring["prom_consumo_12m"] > 0,
        scoring["stock_actual"] / scoring["prom_consumo_12m"],
        np.where(scoring["stock_actual"] > 0, 99.0, 0.0)
    )
    scoring["rotacion_12m"] = np.where(scoring["stock_actual"] > 0, scoring["consumo_12m"] / scoring["stock_actual"], 0)

    # Compras recientes agregadas por SKU
    if not compras.empty:
        fc = compras["fecha_compra"].max()
        if pd.notna(fc):
            lim = fc - pd.DateOffset(months=12)
            ca = compras[compras["fecha_compra"] >= lim].groupby("sku", as_index=False).agg(
                compras_12m=("cantidad_compra", "sum"),
                recibido_12m=("cantidad_recibida", "sum"),
                pendiente_compra=("cantidad_pendiente", "sum"),
                valor_compras_12m=("valor_compra", "sum"),
            )
            scoring = scoring.merge(ca, on="sku", how="left")
    for c in ["compras_12m", "recibido_12m", "pendiente_compra", "valor_compras_12m"]:
        if c not in scoring: scoring[c] = 0
        scoring[c] = scoring[c].fillna(0)

    return {"panel": panel, "scoring": scoring, "fecha_corte": fecha_max}

# ---------------------------------------------------------------------
# NOTEBOOK 3 - Random Forest + IRI
# ---------------------------------------------------------------------
FEATURES = [
    "consumo_3m", "consumo_6m", "consumo_12m",
    "entradas_3m", "entradas_6m", "entradas_12m",
    "meses_con_consumo_3m", "meses_con_consumo_6m", "meses_con_consumo_12m",
    "prom_consumo_12m", "variabilidad_consumo_12m",
    "meses_sin_salida", "stock_estimado", "cobertura_meses", "rotacion_12m"
]

def notebook_3_modelo(n2):
    panel = n2["panel"].copy()
    scoring = n2["scoring"].copy()

    train = panel[panel["inmovilizado_12m"].notna()].copy()
    for c in FEATURES:
        train[c] = nserie(train[c]).replace([np.inf, -np.inf], np.nan).fillna(0)
        scoring[c] = nserie(scoring[c]).replace([np.inf, -np.inf], np.nan).fillna(0)

    metricas = {}
    modelo = None
    if len(train) >= 100 and train["inmovilizado_12m"].nunique() == 2:
        meses = sorted(train["mes"].dropna().unique())
        corte = meses[max(1, int(len(meses) * 0.80) - 1)]
        tr = train[train["mes"] <= corte]
        te = train[train["mes"] > corte]
        if te["inmovilizado_12m"].nunique() < 2 or len(te) < 20:
            tr = train.iloc[:int(len(train)*0.8)]
            te = train.iloc[int(len(train)*0.8):]

        modelo = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        modelo.fit(tr[FEATURES], tr["inmovilizado_12m"].astype(int))
        p = modelo.predict_proba(te[FEATURES])[:, 1]
        y = te["inmovilizado_12m"].astype(int)
        yp = (p >= 0.50).astype(int)
        metricas = {
            "Registros entrenamiento": len(tr),
            "Registros test": len(te),
            "Prevalencia inmovilización": float(y.mean()),
            "Precisión": float(precision_score(y, yp, zero_division=0)),
            "Recall": float(recall_score(y, yp, zero_division=0)),
            "F1": float(f1_score(y, yp, zero_division=0)),
            "F2": float(fbeta_score(y, yp, beta=2, zero_division=0)),
            "PR-AUC": float(average_precision_score(y, p)) if y.nunique() == 2 else np.nan,
            "ROC-AUC": float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan,
        }
        scoring["probabilidad_inmovilizacion"] = modelo.predict_proba(scoring[FEATURES])[:, 1]
        importancia = pd.DataFrame({
            "variable": FEATURES,
            "importancia": modelo.feature_importances_
        }).sort_values("importancia", ascending=False)
    else:
        # Fallback transparente si la historia no permite entrenar dos clases.
        riesgo = (
            0.35 * np.clip(scoring["meses_sin_salida"].fillna(12) / 12, 0, 1) +
            0.30 * np.clip(scoring["cobertura_meses"] / 12, 0, 1) +
            0.20 * (scoring["consumo_6m"] <= 0).astype(float) +
            0.15 * np.clip(1 - scoring["rotacion_12m"], 0, 1)
        )
        scoring["probabilidad_inmovilizacion"] = np.clip(riesgo, 0, 1)
        importancia = pd.DataFrame({"variable": FEATURES, "importancia": np.nan})
        metricas = {"Advertencia": "Historia insuficiente para entrenar Random Forest; se aplicó scoring heurístico transparente."}

    scoring["IRI"] = (100 * scoring["probabilidad_inmovilizacion"]).round(1)
    scoring["nivel_iri"] = pd.cut(
        scoring["IRI"], bins=[-0.01, 40, 80, 100],
        labels=["BAJO", "MEDIO", "ALTO"], include_lowest=True
    ).astype(str)

    def recomendar(r):
        iri = r["IRI"]
        stock = r.get("stock_actual", 0)
        cob = r.get("cobertura_meses", 0)
        pend = r.get("pendiente_compra", 0)
        if iri > 80:
            if stock > 0 and cob > 2:
                return "PRIORIZAR TRANSFERENCIA/REDISTRIBUCIÓN Y REVISAR NUEVA COMPRA"
            return "REVISAR STOCK Y JUSTIFICAR NUEVA COMPRA"
        if iri > 40:
            if pend > 0:
                return "REVISAR COMPRA PENDIENTE Y STOCK CORPORATIVO"
            if stock > 0 and cob > 2:
                return "EVALUAR TRANSFERENCIA ANTES DE COMPRAR"
            return "REVISAR CANTIDAD SOLICITADA"
        return "CONTINUAR EVALUACIÓN NORMAL DE ABASTECIMIENTO"

    scoring["recomendacion_siad"] = scoring.apply(recomendar, axis=1)
    scoring["candidato_transferencia"] = (
        (scoring["IRI"] > 40) &
        (scoring["stock_actual"] > 0) &
        (scoring["cobertura_meses"] > 2)
    )
    return {"scoring": scoring, "metricas": metricas, "importancia": importancia, "modelo": modelo}

# ---------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------
st.title("Sistema Inteligente de Apoyo a la Decisión (SIAD)")
st.subheader("Riesgo de inmovilización por SKU · EXCON")
st.write(
    "El SIAD utiliza información histórica de **consumo, stock, antigüedad, cobertura, "
    "compras y movimientos** para estimar el riesgo de que un material quede inmovilizado "
    "y entregar una recomendación al responsable de abastecimiento."
)

with st.sidebar:
    st.header("Bases de datos")
    zip_file = st.file_uploader("Cargar ZIP con las 7 bases EXCON", type=["zip"])
    carpeta_local = st.text_input("O carpeta local", value=r"C:\Modelo IA")
    ejecutar = st.button("Procesar SIAD", type="primary", use_container_width=True)

@st.cache_data(show_spinner=False)
def ejecutar_desde_carpeta(carpeta):
    fuentes = preparar_fuentes(carpeta)
    n1 = notebook_1_integracion(fuentes)
    n2 = notebook_2_variables(n1)
    n3 = notebook_3_modelo(n2)
    return n1, n2, n3

def procesar_zip(upload):
    td = tempfile.mkdtemp(prefix="siad_")
    with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as z:
        z.extractall(td)
    # Busca la carpeta que contiene las 7 bases.
    for root, dirs, files in os.walk(td):
        if all(v in files for v in ARCHIVOS.values()):
            return ejecutar_desde_carpeta(root)
    raise FileNotFoundError("El ZIP no contiene juntas las 7 bases EXCON requeridas.")

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if ejecutar:
    try:
        with st.spinner("Ejecutando integración, ingeniería de variables y modelo predictivo..."):
            if zip_file is not None:
                st.session_state.resultado = procesar_zip(zip_file)
            else:
                st.session_state.resultado = ejecutar_desde_carpeta(carpeta_local)
        st.success("Proceso completado. SIAD está listo para análisis.")
    except Exception as e:
        st.exception(e)

if st.session_state.resultado is None:
    st.info("Carga el ZIP con las bases EXCON o indica su carpeta y presiona **Procesar SIAD**.")
    st.stop()

n1, n2, n3 = st.session_state.resultado
df = n3["scoring"].copy()

with st.sidebar:
    st.divider()
    st.header("Filtros")
    niveles = st.multiselect("Nivel IRI", ["BAJO", "MEDIO", "ALTO"], default=["BAJO", "MEDIO", "ALTO"])
    ccs = sorted(df["centro_costo"].dropna().astype(str).unique())
    cc_sel = st.multiselect("Centro de costo", ccs)
    buscar = st.text_input("SKU o descripción")

f = df[df["nivel_iri"].isin(niveles)].copy()
if cc_sel:
    f = f[f["centro_costo"].astype(str).isin(cc_sel)]
if buscar:
    mask = f["sku"].astype(str).str.contains(buscar, case=False, na=False)
    if "descripcion" in f:
        mask |= f["descripcion"].astype(str).str.contains(buscar, case=False, na=False)
    f = f[mask]

# KPI
sku = f["sku"].nunique()
alto = f.loc[f["nivel_iri"] == "ALTO", "sku"].nunique()
medio = f.loc[f["nivel_iri"] == "MEDIO", "sku"].nunique()
iri = f["IRI"].mean()
stock = f["stock_actual"].sum()
valor = f["valor_stock"].sum()
transf = f.loc[f["candidato_transferencia"], "sku"].nunique()

cols = st.columns(7)
cols[0].metric("SKU analizados", f"{sku:,}".replace(",", "."))
cols[1].metric("IRI promedio", f"{iri:.1f}")
cols[2].metric("Riesgo alto", f"{alto:,}".replace(",", "."))
cols[3].metric("Riesgo medio", f"{medio:,}".replace(",", "."))
cols[4].metric("Stock", f"{stock:,.0f}".replace(",", "."))
cols[5].metric("Valor stock", "$" + f"{valor:,.0f}".replace(",", "."))
cols[6].metric("Candidatos transferencia", f"{transf:,}".replace(",", "."))

t1, t2, t3, t4 = st.tabs(["Dashboard ejecutivo", "Priorización SKU", "Modelo predictivo", "Trazabilidad 1–2–3"])

with t1:
    a, b = st.columns(2)
    dist = f.groupby("nivel_iri", as_index=False).agg(SKU=("sku", "nunique"))
    with a:
        st.plotly_chart(px.bar(dist, x="nivel_iri", y="SKU", text_auto=True,
                               category_orders={"nivel_iri": ["BAJO","MEDIO","ALTO"]},
                               title="SKU por nivel de riesgo"), use_container_width=True)
    with b:
        fig = px.histogram(f, x="IRI", nbins=20, title="Distribución del IRI")
        fig.add_vline(x=40, line_dash="dash")
        fig.add_vline(x=80, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    top = f.sort_values(["IRI", "valor_stock"], ascending=False).head(20).copy()
    top["etiqueta"] = top["sku"].astype(str) + " · " + top["descripcion"].fillna(top["descripcion_inv"]).astype(str).str[:40]
    st.plotly_chart(px.bar(top.sort_values("IRI"), x="IRI", y="etiqueta", orientation="h",
                           hover_data=["stock_actual","valor_stock","cobertura_meses"],
                           title="Top 20 SKU priorizados por riesgo"), use_container_width=True)

with t2:
    mostrar = [c for c in [
        "sku","descripcion","descripcion_inv","centro_costo","stock_actual","valor_stock",
        "consumo_3m","consumo_6m","consumo_12m","meses_sin_salida","cobertura_meses",
        "rotacion_12m","compras_12m","pendiente_compra","probabilidad_inmovilizacion",
        "IRI","nivel_iri","recomendacion_siad"
    ] if c in f.columns]
    tabla = f.sort_values(["IRI","valor_stock"], ascending=False)[mostrar]
    st.dataframe(tabla, use_container_width=True, hide_index=True, height=560)
    csv = tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar priorización CSV", csv, "SIAD_priorizacion_SKU.csv", "text/csv")

with t3:
    st.markdown("### Desempeño")
    met = n3["metricas"]
    if "Advertencia" in met:
        st.warning(met["Advertencia"])
    else:
        mdf = pd.DataFrame({"Métrica": list(met.keys()), "Valor": list(met.values())})
        st.dataframe(mdf, hide_index=True, use_container_width=True)
    imp = n3["importancia"].dropna()
    if not imp.empty:
        st.plotly_chart(px.bar(imp.head(15).sort_values("importancia"), x="importancia", y="variable",
                               orientation="h", title="Importancia de variables - Random Forest"),
                        use_container_width=True)
    st.caption("La probabilidad se transforma a IRI = probabilidad × 100. Bajo: 0–40; Medio: >40–80; Alto: >80–100.")

with t4:
    st.markdown("### Notebook 1 · Auditoría e integración")
    st.dataframe(n1["auditoria"], hide_index=True, use_container_width=True)
    st.markdown("### Notebook 2 · Ingeniería de variables")
    st.write(f"Panel histórico: **{len(n2['panel']):,} registros** · Fecha de corte: **{n2['fecha_corte'].date()}**")
    st.write("Variables: consumo 3/6/12 meses, entradas, frecuencia, antigüedad sin salida, stock estimado, cobertura y rotación.")
    st.markdown("### Notebook 3 · Predicción y decisión")
    st.write("Random Forest con validación temporal cuando existen suficientes observaciones y ambas clases del target.")
    st.write("Target histórico: material con stock positivo y sin consumo durante los 12 meses siguientes.")
    st.info("SIAD es una herramienta de apoyo: la recomendación no sustituye la decisión del responsable de abastecimiento.")
