# -*- coding: utf-8 -*-
"""
Pipeline SIAD para GitHub Actions
Bases EXCON -> integración -> variables -> Random Forest -> scoring_siad.parquet
"""
from pathlib import Path
import re
import shutil
import unicodedata
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA = Path("data")
OUT = DATA / "scoring_siad.parquet"

FILES = {
    "inventario": "Reporte Inventario.xlsx",
    "recepciones": "Historicos recepciones de compra.xlsx",
    "lineas": "Lins compra.xlsx",
    "movimientos": "Movs. productos.xlsx",
    "pedidos": "Pedidos compra.xlsx",
    "proyectos": "Proyectos.xlsx",
    "productos": "Productos.xlsx",
}

def clean_col(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    x = re.sub(r"[^a-zA-Z0-9]+", "_", x).strip("_").lower()
    return x

def norm(df):
    df = df.copy()
    df.columns = [clean_col(c) for c in df.columns]
    return df

def read_inventory(path):
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    header = None
    for i in range(min(30, len(raw))):
        vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if "articulo" in vals and "cantidad" in vals:
            header = i
            break
    if header is None:
        raise ValueError("No se encontró la cabecera del Reporte Inventario.")
    return norm(pd.read_excel(path, header=header, engine="openpyxl"))


def reconstruir_movimientos():
    """Reconstruye exactamente Movs. productos.xlsx desde las tres partes binarias."""
    destino = DATA / "Movs. productos.xlsx"
    partes = [
        DATA / "Movs_productos.parte01",
        DATA / "Movs_productos.parte02",
        DATA / "Movs_productos.parte03",
    ]

    if destino.exists() and destino.stat().st_size > 8:
        print(f"Movimientos ya reconstruido: {destino}")
        return destino

    faltantes = [p.name for p in partes if not p.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Faltan partes de Movs. productos.xlsx en data/: " + ", ".join(faltantes)
        )

    print("Reconstruyendo Movs. productos.xlsx...")
    with destino.open("wb") as salida:
        for parte in partes:
            print(f"  + {parte.name} ({parte.stat().st_size / 1024 / 1024:.2f} MB)")
            with parte.open("rb") as entrada:
                shutil.copyfileobj(entrada, salida)

    if destino.stat().st_size < 8:
        raise ValueError("El archivo reconstruido es inválido.")

    print(f"Reconstruido: {destino} ({destino.stat().st_size / 1024 / 1024:.2f} MB)")
    return destino


def read_sources():
    reconstruir_movimientos()
    missing = [v for v in FILES.values() if not (DATA / v).exists()]
    if missing:
        raise FileNotFoundError("Faltan bases en data/: " + ", ".join(missing))
    return {
        "inventario": read_inventory(DATA / FILES["inventario"]),
        "recepciones": norm(pd.read_excel(DATA / FILES["recepciones"], engine="openpyxl")),
        "lineas": norm(pd.read_excel(DATA / FILES["lineas"], engine="openpyxl")),
        "movimientos": norm(pd.read_excel(DATA / FILES["movimientos"], engine="openpyxl")),
        "pedidos": norm(pd.read_excel(DATA / FILES["pedidos"], engine="openpyxl")),
        "proyectos": norm(pd.read_excel(DATA / FILES["proyectos"], engine="openpyxl")),
        "productos": norm(pd.read_excel(DATA / FILES["productos"], engine="openpyxl")),
    }

def pick(df, names, required=True):
    for n in names:
        n = clean_col(n)
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"No se encontró ninguna de estas columnas: {names}")
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def build_scoring(src):
    mov = src["movimientos"].copy()
    inv = src["inventario"].copy()
    prod = src["productos"].copy()

    sku_m = pick(mov, ["n_producto", "articulo", "producto"])
    date_m = pick(mov, ["fecha_registro", "fecha"])
    qty_m = pick(mov, ["cantidad"])
    cc_m = pick(mov, ["centro_de_costo", "centro_costo"], required=False)
    tipo_m = pick(mov, ["tipo_movimiento"], required=False)

    mov["sku"] = mov[sku_m].astype(str).str.strip()
    mov["fecha"] = pd.to_datetime(mov[date_m], errors="coerce")
    mov["cantidad_mov"] = to_num(mov[qty_m])
    mov["centro_costo"] = mov[cc_m].fillna("SIN INFORMACIÓN").astype(str) if cc_m else "SIN INFORMACIÓN"
    mov = mov[mov["fecha"].notna() & mov["sku"].ne("")].copy()
    mov["mes"] = mov["fecha"].dt.to_period("M").dt.to_timestamp()

    # Convención ERP usada por SIAD: negativo = salida; positivo = entrada.
    mov["salida"] = (-mov["cantidad_mov"]).clip(lower=0)
    mov["entrada"] = mov["cantidad_mov"].clip(lower=0)

    # Si el texto identifica transferencias, las contabiliza como variable descriptiva.
    if tipo_m:
        txt = mov[tipo_m].astype(str).str.lower()
        mov["transferencia"] = np.where(txt.str.contains("trasp|transf", regex=True, na=False), mov["cantidad_mov"].abs(), 0)
    else:
        mov["transferencia"] = 0.0

    monthly = (mov.groupby(["sku","centro_costo","mes"], as_index=False)
               .agg(salida=("salida","sum"), entrada=("entrada","sum"),
                    transferencias=("transferencia","sum")))

    # Inventario actual
    sku_i = pick(inv, ["articulo", "n_producto", "producto"])
    qty_i = pick(inv, ["cantidad"])
    cost_i = pick(inv, ["costo_extendido"], required=False)
    inv["sku"] = inv[sku_i].astype(str).str.strip()
    inv["stock_actual"] = to_num(inv[qty_i])
    inv["valor_stock"] = to_num(inv[cost_i]) if cost_i else 0.0
    invagg = inv.groupby("sku", as_index=False).agg(
        stock_actual=("stock_actual","sum"), valor_stock=("valor_stock","sum")
    )

    # Descripción de producto
    sku_p = pick(prod, ["n", "articulo", "n_producto"])
    desc_p = pick(prod, ["descripcion"], required=False)
    prod["sku"] = prod[sku_p].astype(str).str.strip()
    descr = (prod[["sku", desc_p]].drop_duplicates("sku")
             .rename(columns={desc_p:"descripcion"}) if desc_p else
             prod[["sku"]].drop_duplicates().assign(descripcion=""))

    max_mes = monthly["mes"].max()
    if pd.isna(max_mes):
        raise ValueError("La base de movimientos no contiene fechas válidas.")

    # Una fila por SKU-centro en el último estado disponible.
    pairs = monthly[["sku","centro_costo"]].drop_duplicates()
    rows = []
    for r in pairs.itertuples(index=False):
        g = monthly[(monthly["sku"] == r.sku) & (monthly["centro_costo"] == r.centro_costo)].sort_values("mes")
        def window(n):
            cutoff = max_mes - pd.DateOffset(months=n-1)
            return g[g["mes"] >= cutoff]
        w3, w6, w12 = window(3), window(6), window(12)
        consumed = g[g["salida"] > 0]
        if len(consumed):
            last = consumed["mes"].max()
            meses_sin = (max_mes.year-last.year)*12 + max_mes.month-last.month
        else:
            meses_sin = 999
        consumo12 = w12["salida"].sum()
        rows.append({
            "sku": r.sku, "centro_costo": r.centro_costo,
            "consumo_3m": w3["salida"].sum(),
            "consumo_6m": w6["salida"].sum(),
            "consumo_12m": consumo12,
            "entradas_3m": w3["entrada"].sum(),
            "entradas_6m": w6["entrada"].sum(),
            "entradas_12m": w12["entrada"].sum(),
            "meses_con_consumo_3m": int((w3["salida"] > 0).sum()),
            "meses_con_consumo_6m": int((w6["salida"] > 0).sum()),
            "meses_con_consumo_12m": int((w12["salida"] > 0).sum()),
            "prom_consumo_12m": consumo12/12,
            "variabilidad_consumo_12m": float(w12["salida"].std(ddof=0) if len(w12) else 0),
            "meses_sin_salida": meses_sin,
            "transferencias_12m": w12["transferencias"].sum(),
        })
    score = pd.DataFrame(rows).merge(invagg, on="sku", how="left").merge(descr, on="sku", how="left")
    score[["stock_actual","valor_stock"]] = score[["stock_actual","valor_stock"]].fillna(0)
    avg = score["prom_consumo_12m"].replace(0, np.nan)
    score["cobertura_meses"] = (score["stock_actual"] / avg).replace([np.inf,-np.inf], np.nan).fillna(999).clip(upper=999)
    score["rotacion_12m"] = np.where(score["stock_actual"] > 0, score["consumo_12m"]/score["stock_actual"], 0)

    # Entrenamiento histórico simplificado usando panel mensual completo.
    # Target: stock de flujo positivo y ausencia de consumo en los siguientes 12 meses.
    panel_rows = []
    for (sku, cc), g in monthly.groupby(["sku","centro_costo"], sort=False):
        g = g.set_index("mes").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        z = g.reindex(idx, fill_value=0)
        z["stock_estimado"] = (z["entrada"] - z["salida"]).cumsum().clip(lower=0)
        future = z["salida"].shift(-1)[::-1].rolling(12, min_periods=12).sum()[::-1]
        z["target"] = ((z["stock_estimado"] > 0) & (future <= 0)).astype(float)
        z.loc[future.isna(), "target"] = np.nan
        # Variables de ventana
        for n in (3,6,12):
            z[f"consumo_{n}m"] = z["salida"].rolling(n, min_periods=1).sum()
            z[f"entradas_{n}m"] = z["entrada"].rolling(n, min_periods=1).sum()
            z[f"meses_con_consumo_{n}m"] = (z["salida"]>0).rolling(n, min_periods=1).sum()
        z["prom_consumo_12m"] = z["salida"].rolling(12, min_periods=1).mean()
        z["variabilidad_consumo_12m"] = z["salida"].rolling(12, min_periods=1).std(ddof=0).fillna(0)
        last_idx = None
        ms = []
        for d, val in zip(z.index, z["salida"]):
            if val > 0: last_idx = d
            ms.append(999 if last_idx is None else (d.year-last_idx.year)*12+d.month-last_idx.month)
        z["meses_sin_salida"] = ms
        z["cobertura_meses"] = np.where(z["prom_consumo_12m"]>0, z["stock_estimado"]/z["prom_consumo_12m"], 999)
        z["rotacion_12m"] = np.where(z["stock_estimado"]>0, z["consumo_12m"]/z["stock_estimado"], 0)
        panel_rows.append(z.reset_index(drop=True))

    train = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    features = [
        "consumo_3m","consumo_6m","consumo_12m",
        "entradas_3m","entradas_6m","entradas_12m",
        "meses_con_consumo_3m","meses_con_consumo_6m","meses_con_consumo_12m",
        "prom_consumo_12m","variabilidad_consumo_12m","meses_sin_salida",
        "stock_estimado","cobertura_meses","rotacion_12m"
    ]

    # El scoring actual usa stock real como stock_estimado.
    score["stock_estimado"] = score["stock_actual"]
    Xscore = score[features].replace([np.inf,-np.inf], np.nan).fillna(0)

    valid = train.dropna(subset=["target"]) if len(train) else train
    if len(valid) >= 100 and valid["target"].nunique() == 2:
        X = valid[features].replace([np.inf,-np.inf], np.nan).fillna(0)
        y = valid["target"].astype(int)
        model = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        score["probabilidad_inmovilizacion"] = model.predict_proba(Xscore)[:,1]
    else:
        # Fallback transparente si el histórico no permite entrenar dos clases.
        p = (
            0.45*np.clip(score["meses_sin_salida"]/12,0,1)
            + 0.35*np.clip(score["cobertura_meses"]/12,0,1)
            + 0.20*(1-np.clip(score["rotacion_12m"],0,1))
        )
        score["probabilidad_inmovilizacion"] = np.clip(p,0,1)

    score["IRI"] = (100*score["probabilidad_inmovilizacion"]).round(1)
    score["nivel_iri"] = pd.cut(score["IRI"], [-0.001,40,80,100],
                                labels=["BAJO","MEDIO","ALTO"], include_lowest=True).astype(str)

    def rec(r):
        if r.IRI > 80:
            if r.stock_actual > 0 and r.cobertura_meses > 2:
                return "PRIORIZAR TRANSFERENCIA/REDISTRIBUCIÓN Y REVISAR NUEVA COMPRA"
            return "REVISAR STOCK Y JUSTIFICAR NUEVA COMPRA"
        if r.IRI > 40:
            if r.stock_actual > 0 and r.cobertura_meses > 2:
                return "EVALUAR TRANSFERENCIA ANTES DE COMPRAR"
            return "REVISAR CANTIDAD SOLICITADA"
        return "CONTINUAR EVALUACIÓN NORMAL DE ABASTECIMIENTO"

    score["recomendacion_siad"] = score.apply(rec, axis=1)
    score["compras_12m"] = score["entradas_12m"]
    return score

if __name__ == "__main__":
    print("1/3 Leyendo bases EXCON...")
    src = read_sources()
    print("2/3 Integrando, construyendo variables y ejecutando modelo...")
    scoring = build_scoring(src)
    print("3/3 Guardando scoring SIAD...")
    DATA.mkdir(exist_ok=True)
    scoring.to_parquet(OUT, index=False)
    print(f"OK: {OUT} | {len(scoring):,} registros")

    # El Excel reconstruido es temporal: no se guarda en GitHub.
    mov_reconstruido = DATA / "Movs. productos.xlsx"
    if mov_reconstruido.exists():
        mov_reconstruido.unlink()
        print("Archivo temporal Movs. productos.xlsx eliminado.")
