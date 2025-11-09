import streamlit as st
import cv2
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
from ultralytics import solutions

# Mapeo de clases COCO
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CLASSES_TO_DETECT = [0, 1, 2, 3, 5, 7]

def process_video(
    video_path,
    orientation="vertical",
    resize_factor=0.5,
    rect_width=20,
    progress_bar=None,
    status_text=None
):
    """
    Procesa el video con detección y conteo de objetos
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error al leer el archivo de video")

    w, h, fps = (int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS
    ))

    proc_w = int(w * resize_factor)
    proc_h = int(h * resize_factor)

    # Configurar región según orientación
    if orientation == "vertical":
        center_x = int(proc_w / 2)
        region_points = [
            (center_x - rect_width, 0),
            (center_x + rect_width, 0),
            (center_x + rect_width, proc_h),
            (center_x - rect_width, proc_h)
        ]
    else:  # horizontal
        center_y = int(proc_h / 2)
        region_points = [
            (0, center_y - rect_width),
            (proc_w, center_y - rect_width),
            (proc_w, center_y + rect_width),
            (0, center_y + rect_width)
        ]

    # Crear archivo temporal para salida
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_file.name
    output_file.close()

    # Configurar video writer
    try:
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (proc_w, proc_h)
        )
        if not video_writer.isOpened():
            raise Exception("H.264 no disponible")
    except:
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (proc_w, proc_h)
        )

    # Inicializar contador
    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model="yolo11n.pt",
        classes=CLASSES_TO_DETECT,
        tracker="botsort.yaml",
        show_in=True,
        show_out=True,
        line_width=2
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    last_results = None

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        frame_num += 1

        # Actualizar progreso
        if progress_bar and frame_num % 10 == 0:
            progress = frame_num / total_frames
            progress_bar.progress(progress)
            if status_text:
                status_text.text(f"Procesando: {frame_num}/{total_frames} frames ({progress*100:.1f}%)")

        # Procesar frame
        resized_frame = cv2.resize(im0, (proc_w, proc_h))
        results = counter(resized_frame)
        last_results = results

        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()

    # Extraer conteos por clase
    conteo_por_clase = {}

    if last_results and hasattr(last_results, 'classwise_count') and last_results.classwise_count:
        for class_id, counts in last_results.classwise_count.items():
            try:
                numeric_id = int(class_id)
                class_name = CLASS_NAMES.get(numeric_id, f"clase_{class_id}")
            except (ValueError, TypeError):
                class_name = str(class_id).lower()

            if isinstance(counts, dict):
                in_count = counts.get('IN', counts.get('in', 0))
                out_count = counts.get('OUT', counts.get('out', 0))
                conteo_por_clase[class_name] = {
                    'in': int(in_count),
                    'out': int(out_count),
                    'total': int(in_count) + int(out_count)
                }

    elif hasattr(counter, 'classwise_count') and counter.classwise_count:
        for class_id, counts in counter.classwise_count.items():
            try:
                numeric_id = int(class_id)
                class_name = CLASS_NAMES.get(numeric_id, f"clase_{class_id}")
            except (ValueError, TypeError):
                class_name = str(class_id).lower()

            if isinstance(counts, dict):
                in_count = counts.get('IN', counts.get('in', 0))
                out_count = counts.get('OUT', counts.get('out', 0))
                conteo_por_clase[class_name] = {
                    'in': int(in_count),
                    'out': int(out_count),
                    'total': int(in_count) + int(out_count)
                }

    # Preparar resultados
    results_data = {
        "processing_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "fecha_procesamiento": datetime.now().isoformat(),
        "conteo_total": {
            "in_count": counter.in_count if hasattr(counter, 'in_count') else 0,
            "out_count": counter.out_count if hasattr(counter, 'out_count') else 0,
            "total": (counter.in_count + counter.out_count) if hasattr(counter, 'in_count') and hasattr(counter, 'out_count') else 0
        },
        "conteo_por_clase": conteo_por_clase,
        "configuracion": {
            "orientacion": orientation,
            "resize_factor": resize_factor,
            "rect_width": rect_width
        }
    }

    return output_path, results_data


def main():
    st.set_page_config(
        page_title="Contador de Objetos YOLO",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personalizado
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .stButton>button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            font-weight: 600;
            padding: 0.75rem;
            border-radius: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #1557a0;
        }
        div[data-testid="stExpander"] {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header principal
    st.markdown('<p class="main-header">Contador de Objetos con YOLO11</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detección y conteo automático de personas, vehículos y bicicletas en videos</p>', unsafe_allow_html=True)

    # Sidebar para configuración
    st.sidebar.title("Configuración del Procesamiento")

    # Upload de video
    st.sidebar.markdown("### Archivo de Video")
    uploaded_file = st.sidebar.file_uploader(
        "Arrastra tu video aquí",
        type=['mp4', 'avi', 'mov', 'MOV'],
        help="Formatos soportados: MP4, AVI, MOV"
    )

    st.sidebar.markdown("---")

    # Configuración de orientación
    st.sidebar.markdown("### Región de Conteo")
    orientation = st.sidebar.radio(
        "Orientación del rectángulo",
        options=["vertical", "horizontal"],
        help="Vertical: Cuenta objetos de izquierda a derecha\nHorizontal: Cuenta objetos de arriba a abajo"
    )

    rect_width = st.sidebar.slider(
        "Ancho del rectángulo (px)",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
        help="Mayor ancho = área de detección más grande"
    )

    st.sidebar.markdown("---")

    # Configuración de calidad
    st.sidebar.markdown("### Calidad y Rendimiento")
    resize_factor = st.sidebar.select_slider(
        "Resolución del video",
        options=[25, 50, 75, 100],
        value=50,
        help="25%: Muy rápido, baja calidad\n50%: Balanceado (recomendado)\n75%: Buena calidad, más lento\n100%: Máxima calidad, muy lento",
        format_func=lambda x: f"{x}%"
    ) / 100

    # Botón de procesamiento
    if uploaded_file is not None:
        # Mostrar información del video
        st.sidebar.markdown("---")
        st.sidebar.success("Video cargado")
        st.sidebar.info(f"**{uploaded_file.name}**")

        # Guardar video temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.sidebar.markdown("---")

        # Botón para procesar
        if st.sidebar.button("Iniciar Procesamiento", type="primary"):
            # Crear columnas para layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Procesando...")
                progress_bar = st.progress(0)
                status_text = st.empty()

            try:
                # Procesar video
                output_path, results = process_video(
                    video_path,
                    orientation=orientation,
                    resize_factor=resize_factor,
                    rect_width=rect_width,
                    progress_bar=progress_bar,
                    status_text=status_text
                )

                progress_bar.progress(100)
                status_text.text("Procesamiento completado")

                # Mostrar resultados
                st.success("Video procesado exitosamente")

                # Tabs para organizar mejor
                tab1, tab2, tab3 = st.tabs(["Video Procesado", "Resultados", "Descargas"])

                with tab1:
                    st.markdown("### Video con Detecciones")
                    # Mostrar video
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)

                with tab2:
                    # Métricas principales en tarjetas
                    st.markdown("### Resumen General")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Total de Objetos",
                            value=results['conteo_total']['total'],
                            delta=None
                        )

                    with col2:
                        st.metric(
                            label="Entrada (IN)",
                            value=results['conteo_total']['in_count'],
                            delta=None
                        )

                    with col3:
                        st.metric(
                            label="Salida (OUT)",
                            value=results['conteo_total']['out_count'],
                            delta=None
                        )

                    st.markdown("---")

                    # Tabla de conteo por clase
                    st.markdown("### Detalle por Clase")

                    # Preparar datos para la tabla
                    table_data = []

                    for clase, counts in results['conteo_por_clase'].items():
                        if isinstance(counts, dict) and counts.get('total', 0) > 0:
                            table_data.append({
                                'Tipo': clase.capitalize(),
                                'Entrada': counts.get('in', 0),
                                'Salida': counts.get('out', 0),
                                'Total': counts.get('total', 0)
                            })

                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Tipo": st.column_config.TextColumn("Tipo de Objeto", width="medium"),
                                "Entrada": st.column_config.NumberColumn("Entrada (IN)", width="small"),
                                "Salida": st.column_config.NumberColumn("Salida (OUT)", width="small"),
                                "Total": st.column_config.NumberColumn("Total", width="small")
                            }
                        )
                    else:
                        st.info("No se detectaron objetos en el video")

                with tab3:
                    st.markdown("### Descargar Resultados")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Video Procesado")
                        st.download_button(
                            label="Descargar Video MP4",
                            data=video_bytes,
                            file_name=f"processed_{uploaded_file.name}",
                            mime="video/mp4",
                            use_container_width=True
                        )

                    with col2:
                        st.markdown("#### Datos en JSON")
                        json_str = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Descargar JSON",
                            data=json_str,
                            file_name=f"results_{results['processing_id']}.json",
                            mime="application/json",
                            use_container_width=True
                        )

                    st.markdown("---")

                    # Mostrar resumen de configuración
                    with st.expander("Ver Configuración Usada"):
                        st.json(results['configuracion'])

                # Limpiar archivo temporal
                Path(video_path).unlink(missing_ok=True)

            except Exception as e:
                st.error(f"Error al procesar el video: {str(e)}")
                st.exception(e)

    else:
        # Instrucciones cuando no hay video
        st.info("Sube un video desde la barra lateral para comenzar")

        # Crear tabs para organizar información
        tab1, tab2, tab3 = st.tabs(["Guía de Uso", "Clases Detectadas", "Acerca de"])

        with tab1:
            st.markdown("""
            ### ¿Cómo usar esta aplicación?

            Sigue estos pasos para procesar tu video:
            """)

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown("**Paso 1**")
            with col2:
                st.markdown("**Sube tu video** usando la barra lateral (formatos: MP4, AVI, MOV)")

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown("**Paso 2**")
            with col2:
                st.markdown("""
                **Configura la orientación** del rectángulo de conteo:
                - **Vertical**: Detecta objetos que cruzan de izquierda a derecha
                - **Horizontal**: Detecta objetos que cruzan de arriba a abajo
                """)

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown("**Paso 3**")
            with col2:
                st.markdown("""
                **Ajusta la calidad** según tus necesidades:
                - **25%**: Procesamiento muy rápido, calidad básica
                - **50%**: Balance ideal (recomendado)
                - **75%**: Buena calidad, tiempo moderado
                - **100%**: Máxima calidad, procesamiento lento
                """)

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown("**Paso 4**")
            with col2:
                st.markdown("**Haz clic en Iniciar Procesamiento** y espera el resultado")

            col1, col2 = st.columns([1, 4])

            with col1:
                st.markdown("**Paso 5**")
            with col2:
                st.markdown("**Descarga** el video procesado y los resultados en JSON")

        with tab2:
            st.markdown("### Objetos que el sistema puede detectar:")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                #### Personas
                - Peatones
                - Personas caminando
                - Grupos de personas
                """)

            with col2:
                st.markdown("""
                #### Vehículos Motorizados
                - Autos
                - Motocicletas
                - Buses
                - Camiones
                """)

            with col3:
                st.markdown("""
                #### Vehículos No Motorizados
                - Bicicletas
                - Scooters
                """)

            st.info("""
            **Nota**: El sistema utiliza YOLO11, uno de los modelos de detección de objetos más avanzados,
            con tracking BoTSORT para evitar conteos duplicados.
            """)

        with tab3:
            st.markdown("""
            ### Acerca de esta aplicación

            **Tecnologías utilizadas:**
            - **YOLO11**: Modelo de detección de objetos de última generación
            - **BoTSORT**: Sistema de tracking avanzado para seguimiento de objetos
            - **Python**: Lenguaje de programación
            - **Streamlit**: Framework de interfaz web
            - **OpenCV**: Procesamiento de video

            **Características:**
            - Detección en tiempo real
            - Tracking con IDs únicos
            - Conteo bidireccional (IN/OUT)
            - Múltiples clases de objetos
            - Exportación de resultados

            ---

            **Versión**: 1.0.0
            """)

            st.success("**Consejo**: Para mejores resultados, usa videos con buena iluminación y objetos claramente visibles.")


if __name__ == "__main__":
    main()
