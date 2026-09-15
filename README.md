# SIAD EXCON — GitHub Actions + Streamlit

Arquitectura liviana:

`7 bases EXCON -> GitHub Actions -> pipeline_siad.py -> data/scoring_siad.parquet -> Streamlit`

## Primera configuración

1. Cree un repositorio **privado** en GitHub.
2. Suba todos los archivos y carpetas de este paquete conservando la estructura.
3. Suba las siete bases Excel a `data/` con los nombres indicados en `data/README.txt`.
4. En GitHub abra **Actions**.
5. Seleccione **Actualizar SIAD**.
6. Presione **Run workflow**.
7. Espere a que finalice en verde.
8. Compruebe que apareció `data/scoring_siad.parquet`.

## Streamlit

En Streamlit Community Cloud seleccione el repositorio y configure:

- Main file path: `app.py`

La app ya no procesa los siete Excel ni entrena el modelo: solo lee el scoring generado por GitHub Actions.

## Actualización

Cuando cambien las bases:
1. Reemplace los Excel de `data/`.
2. Ejecute **Actions > Actualizar SIAD > Run workflow**.
3. GitHub regenerará y hará commit de `data/scoring_siad.parquet`.
4. Streamlit leerá la versión actualizada.

## Seguridad

Las bases de inventario/compras/movimientos pueden contener información interna. Mantenga el repositorio privado y valide las políticas institucionales antes de alojarlas en servicios externos.
