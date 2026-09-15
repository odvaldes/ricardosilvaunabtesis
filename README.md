# SIAD – Sistema Inteligente de Apoyo a la Decisión

Aplicación Streamlit para estimar el riesgo de inmovilización por SKU en EXCON.

## Archivos del repositorio
- `app.py`: aplicación integrada (Notebooks 1, 2 y 3 + dashboard).
- `requirements.txt`: dependencias para Streamlit Community Cloud.
- `.streamlit/config.toml`: configuración de carga de archivos.

## Bases
La app permite cargar un ZIP con estas 7 bases:
- Historicos recepciones de compra.xlsx
- Lins compra.xlsx
- Movs. productos.xlsx
- Pedidos compra.xlsx
- Productos.xlsx
- Proyectos.xlsx
- Reporte Inventario.xlsx

No es necesario subir las bases al repositorio GitHub si se cargarán desde la interfaz.

## Despliegue
1. Crear repositorio en GitHub.
2. Subir el contenido de esta carpeta conservando `.streamlit/config.toml`.
3. En Streamlit Community Cloud seleccionar el repositorio.
4. Main file path: `app.py`.
5. Deploy.
