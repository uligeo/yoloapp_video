# Sistema de Detección y Conteo de Objetos con YOLO11

Sistema de detección y conteo automático de objetos en videos usando YOLO11 con tracking BoTSORT.

## Características

- Detección de objetos en tiempo real con YOLO11
- Tracking de objetos con BoTSORT para evitar duplicados
- Conteo bidireccional (entrada/salida)
- Detección de múltiples clases: personas, autos, bicicletas, motocicletas, buses y camiones
- Interfaz web interactiva con Streamlit
- Exportación de resultados en JSON
- Regiones de conteo configurables (vertical/horizontal)
- Ajuste de calidad y resolución del video

## Requisitos

- Python 3.13 o superior
- uv (gestor de paquetes de Python)

## Instalación

1. Clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd yolo_detect
```

2. Instalar dependencias con uv:

```bash
uv sync
```

## Uso

### Interfaz Web (Streamlit)

Ejecutar la aplicación web:

```bash
uv run streamlit run app_streamlit.py
```

La aplicación se abrirá en el navegador en `http://localhost:8501`

### Pasos para usar la aplicación:

1. Sube un video (formatos: MP4, AVI, MOV)
2. Configura la orientación del rectángulo de conteo:
   - **Vertical**: Detecta objetos de izquierda a derecha
   - **Horizontal**: Detecta objetos de arriba a abajo
3. Ajusta la resolución del video según tus necesidades
4. Haz clic en "Iniciar Procesamiento"
5. Descarga el video procesado y los resultados en JSON

## Clases Detectadas

El sistema puede detectar las siguientes clases de objetos:

- **Personas**: Peatones y personas caminando
- **Vehículos Motorizados**: Autos, motocicletas, buses, camiones
- **Vehículos No Motorizados**: Bicicletas

## Estructura del Proyecto

```
yolo_detect/
├── app_streamlit.py    # Aplicación web de Streamlit
├── pyproject.toml      # Configuración de dependencias
├── README.md           # Este archivo
├── .gitignore          # Archivos ignorados por Git
└── .python-version     # Versión de Python del proyecto
```

## Dependencias Principales

- **ultralytics**: Framework YOLO11
- **opencv-python**: Procesamiento de video
- **streamlit**: Framework de interfaz web
- **pandas**: Manejo de datos
- **shapely**: Geometría computacional
- **lap**: Solver para tracking BoTSORT

## Salida de Resultados

El sistema genera:

1. **Video procesado**: Video con detecciones y conteos visuales
2. **Archivo JSON**: Resultados detallados con:
   - Conteo total de objetos (entrada/salida)
   - Conteo por clase de objeto
   - Configuración utilizada
   - Timestamp del procesamiento

## Notas Técnicas

- El sistema utiliza tracking BoTSORT para asignar IDs únicos a cada objeto
- El tracking evita conteos duplicados al cruzar la región de conteo
- El codec H.264 se utiliza para optimizar el tamaño del video de salida
- Los modelos YOLO se descargan automáticamente la primera vez

## Versión

1.0.0
