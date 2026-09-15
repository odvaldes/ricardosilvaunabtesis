# -*- coding: utf-8 -*-
"""
SIAD LITE - Sistema Inteligente de Apoyo a la Decisión
Dashboard liviano para Streamlit Community Cloud.

ARQUITECTURA RECOMENDADA
GitHub Actions / procesamiento local:
    Bases EXCON -> Notebook 1 -> Notebook 2 -> Notebook 3 -> scoring_siad.parquet

Streamlit:
    scoring_siad.parquet -> Dashboard SIAD

Ejecución:
    streamlit run SIAD_Lite_Streamlit.py

En GitHub puede renombrarse como:
    app.py
"""

from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="SIAD | EXCON",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------
ARCHIVO_DEFAULT = Path("data/scoring_siad.parquet")

COLUMNAS_NUMERICAS = [
    "stock_actual",
    "valor_stock",
    "consumo_3m",
    "consumo_6m",
    "consumo_12m",
    "meses_sin_salida",
    "cobertura_meses",
    "rotacion_12m",
    "compras_12m",
    "transferencias_12m",
    "probabilidad_inmovilizacion",
    "IRI",
]

# ---------------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------------
def normalizar_columnas(df):
    ren = {
        "SKU": "sku",
        "Sku": "sku",
        "descripcion_producto": "descripcion",
        "Descripción": "descripcion",
        "Centro de costo": "centro_costo",
        "Centro de Costo": "centro_costo",
        "Stock actual": "stock_actual",
        "Valor stock": "valor_stock",
        "Probabilidad inmovilización": "probabilidad_inmovilizacion",
        "Nivel IRI": "nivel_iri",
        "Recomendación SIAD": "recomendacion_siad",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df

def preparar(df):
    df = normalizar_columnas(df.copy())

    if "sku" not in df.columns:
        raise ValueError("El archivo de scoring debe contener una columna 'sku'.")

    for c in COLUMNAS_NUMERICAS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "IRI" not in df.columns:
        if "probabilidad_inmovilizacion" not in df.columns:
            raise ValueError(
                "El archivo debe contener 'IRI' o 'probabilidad_inmovilizacion'."
            )
        df["IRI"] = 100 * df["probabilidad_inmovilizacion"]

    df["IRI"] = df["IRI"].clip(0, 100)

    if "probabilidad_inmovilizacion" not in df.columns:
        df["probabilidad_inmovilizacion"] = df["IRI"] / 100

    # Clasificación final del proyecto.
    df["nivel_iri"] = pd.cut(
        df["IRI"],
        bins=[-0.001, 40, 80, 100],
        labels=["BAJO", "MEDIO", "ALTO"],
        include_lowest=True,
    ).astype(str)

    for c, default in {
        "descripcion": "",
        "centro_costo": "SIN INFORMACIÓN",
        "stock_actual": 0.0,
        "valor_stock": 0.0,
        "consumo_3m": np.nan,
        "consumo_6m": np.nan,
        "consumo_12m": np.nan,
        "meses_sin_salida": np.nan,
        "cobertura_meses": np.nan,
        "rotacion_12m": np.nan,
        "compras_12m": np.nan,
        "transferencias_12m": np.nan,
    }.items():
        if c not in df.columns:
            df[c] = default

    if "recomendacion_siad" not in df.columns:
        def recomendacion(r):
            if r["IRI"] > 80:
                if r["stock_actual"] > 0 and r["cobertura_meses"] > 2:
                    return "PRIORIZAR TRANSFERENCIA/REDISTRIBUCIÓN Y REVISAR NUEVA COMPRA"
                return "REVISAR STOCK Y JUSTIFICAR NUEVA COMPRA"
            if r["IRI"] > 40:
                if r["stock_actual"] > 0 and r["cobertura_meses"] > 2:
                    return "EVALUAR TRANSFERENCIA ANTES DE COMPRAR"
                return "REVISAR CANTIDAD SOLICITADA"
            return "CONTINUAR EVALUACIÓN NORMAL DE ABASTECIMIENTO"
        df["recomendacion_siad"] = df.apply(recomendacion, axis=1)

    df["candidato_transferencia"] = (
        (df["IRI"] > 40)
        & (df["stock_actual"] > 0)
        & (df["cobertura_meses"] > 2)
    )

    return df

@st.cache_data(show_spinner=False)
def cargar_parquet(path):
    return preparar(pd.read_parquet(path))

@st.cache_data(show_spinner=False)
def cargar_subido(bytes_archivo, nombre):
    bio = io.BytesIO(bytes_archivo)
    if nombre.lower().endswith(".parquet"):
        return preparar(pd.read_parquet(bio))
    if nombre.lower().endswith(".csv"):
        return preparar(pd.read_csv(bio))
    if nombre.lower().endswith(".xlsx"):
        return preparar(pd.read_excel(bio, engine="openpyxl"))
    raise ValueError("Formato no soportado.")

def clp(v):
    if pd.isna(v):
        return "—"
    return "$" + f"{v:,.0f}".replace(",", ".")

# ---------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------
st.title("Sistema Inteligente de Apoyo a la Decisión (SIAD)")
st.caption("EXCON · Gestión predictiva del riesgo de inmovilización de inventario")

st.markdown(
    """
El **SIAD** utiliza los resultados procesados por los Notebooks 1, 2 y 3 para
estimar y visualizar el **riesgo de inmovilización asociado a cada SKU**.
La aplicación constituye la capa de apoyo a la decisión y evita reentrenar
el modelo cada vez que se abre el dashboard.
"""
)

# ---------------------------------------------------------------------
# CARGA LIVIANA
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Datos SIAD")
    st.caption("Se recomienda mantener `data/scoring_siad.parquet` en el repositorio.")
    upload = st.file_uploader(
        "Opcional: cargar scoring actualizado",
        type=["parquet", "csv", "xlsx"],
    )

try:
    if upload is not None:
        df = cargar_subido(upload.getvalue(), upload.name)
        fuente = f"Archivo cargado: {upload.name}"
    elif ARCHIVO_DEFAULT.exists():
        df = cargar_parquet(str(ARCHIVO_DEFAULT))
        fuente = str(ARCHIVO_DEFAULT)
    else:
        st.warning(
            "No se encontró `data/scoring_siad.parquet`. "
            "Agrega la salida del Notebook 3 al repositorio o carga el archivo desde el panel lateral."
        )
        st.stop()
except Exception as e:
    st.error("No fue posible cargar el scoring SIAD.")
    st.exception(e)
    st.stop()

st.success(f"Datos cargados · {len(df):,} registros · {fuente}")

# ---------------------------------------------------------------------
# FILTROS
# ---------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.header("Filtros")

    niveles = st.multiselect(
        "Nivel de riesgo",
        ["BAJO", "MEDIO", "ALTO"],
        default=["BAJO", "MEDIO", "ALTO"],
    )

    centros = sorted(df["centro_costo"].dropna().astype(str).unique())
    centros_sel = st.multiselect("Centro de costo", centros)

    buscar = st.text_input("Buscar SKU o descripción")

f = df[df["nivel_iri"].isin(niveles)].copy()

if centros_sel:
    f = f[f["centro_costo"].astype(str).isin(centros_sel)]

if buscar:
    mask = f["sku"].astype(str).str.contains(buscar, case=False, na=False)
    mask |= f["descripcion"].astype(str).str.contains(buscar, case=False, na=False)
    f = f[mask]

# ---------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------
sku_total = f["sku"].nunique()
iri_promedio = f["IRI"].mean()
alto = f.loc[f["nivel_iri"] == "ALTO", "sku"].nunique()
medio = f.loc[f["nivel_iri"] == "MEDIO", "sku"].nunique()
valor_stock = f["valor_stock"].sum()
valor_riesgo = f.loc[f["nivel_iri"].isin(["MEDIO", "ALTO"]), "valor_stock"].sum()
candidatos = f.loc[f["candidato_transferencia"], "sku"].nunique()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("SKU analizados", f"{sku_total:,}".replace(",", "."))
k2.metric("IRI promedio", f"{iri_promedio:.1f}" if pd.notna(iri_promedio) else "—")
k3.metric("Riesgo alto", f"{alto:,}".replace(",", "."))
k4.metric("Riesgo medio", f"{medio:,}".replace(",", "."))
k5.metric("Candidatos transferencia", f"{candidatos:,}".replace(",", "."))
k6.metric("Valor stock", clp(valor_stock))

st.caption(
    f"Valor de stock asociado a riesgo medio/alto: **{clp(valor_riesgo)}**"
)

# ---------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Resumen ejecutivo", "Priorización por SKU", "Ficha de material"]
)

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        dist = (
            f.groupby("nivel_iri", as_index=False)
            .agg(SKU=("sku", "nunique"))
        )
        fig = px.bar(
            dist,
            x="nivel_iri",
            y="SKU",
            text_auto=True,
            category_orders={"nivel_iri": ["BAJO", "MEDIO", "ALTO"]},
            title="SKU por nivel de riesgo",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            f,
            x="IRI",
            nbins=20,
            title="Distribución del Índice de Riesgo de Inmovilización",
        )
        fig.add_vline(x=40, line_dash="dash")
        fig.add_vline(x=80, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    top = f.sort_values(["IRI", "valor_stock"], ascending=[False, False]).head(20).copy()
    top["material"] = (
        top["sku"].astype(str)
        + " · "
        + top["descripcion"].fillna("").astype(str).str[:42]
    )

    fig = px.bar(
        top.sort_values("IRI"),
        x="IRI",
        y="material",
        orientation="h",
        hover_data=["stock_actual", "valor_stock", "cobertura_meses"],
        title="Top 20 SKU priorizados",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    columnas = [
        "sku",
        "descripcion",
        "centro_costo",
        "stock_actual",
        "valor_stock",
        "consumo_3m",
        "consumo_6m",
        "consumo_12m",
        "meses_sin_salida",
        "cobertura_meses",
        "rotacion_12m",
        "compras_12m",
        "transferencias_12m",
        "probabilidad_inmovilizacion",
        "IRI",
        "nivel_iri",
        "recomendacion_siad",
    ]

    tabla = (
        f.sort_values(["IRI", "valor_stock"], ascending=[False, False])
        [[c for c in columnas if c in f.columns]]
    )

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True,
        height=550,
    )

    st.download_button(
        "Descargar priorización",
        tabla.to_csv(index=False).encode("utf-8-sig"),
        "SIAD_priorizacion_SKU.csv",
        "text/csv",
    )

with tab3:
    opciones = f["sku"].dropna().astype(str).unique().tolist()

    if opciones:
        sku_sel = st.selectbox("Seleccionar SKU", opciones)
        material = f[f["sku"].astype(str) == sku_sel].sort_values("IRI", ascending=False).iloc[0]

        st.subheader(f"SKU {sku_sel}")
        st.write(material.get("descripcion", ""))

        a, b, c, d = st.columns(4)
        a.metric("IRI", f"{material['IRI']:.1f}")
        b.metric("Nivel", material["nivel_iri"])
        c.metric("Stock actual", f"{material['stock_actual']:,.0f}".replace(",", "."))
        d.metric("Cobertura", f"{material['cobertura_meses']:.1f} meses")

        st.markdown("#### Recomendación SIAD")
        st.info(material["recomendacion_siad"])

        detalle = {
            "Consumo 3 meses": material.get("consumo_3m"),
            "Consumo 6 meses": material.get("consumo_6m"),
            "Consumo 12 meses": material.get("consumo_12m"),
            "Meses sin salida": material.get("meses_sin_salida"),
            "Rotación 12 meses": material.get("rotacion_12m"),
            "Valor stock": clp(material.get("valor_stock")),
            "Probabilidad de inmovilización": f"{100 * material['probabilidad_inmovilizacion']:.1f}%",
        }
        st.dataframe(
            pd.DataFrame(detalle.items(), columns=["Indicador", "Valor"]),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.caption(
    "SIAD es una herramienta de apoyo a la decisión. "
    "La decisión final de abastecimiento permanece en el profesional responsable."
)
