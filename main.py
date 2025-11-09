import cv2
import json
from datetime import datetime
from pathlib import Path

from ultralytics import solutions

# Mapeo de clases COCO
CLASS_NAMES = {
    0: "person",      # Persona
    1: "bicycle",     # Bicicleta
    2: "car",         # Auto
    3: "motorcycle",  # Motocicleta
    5: "bus",         # Bus
    7: "truck"        # Camión
}

# Clases a detectar (puedes agregar más del diccionario anterior)
CLASSES_TO_DETECT = [0, 1, 2, 3, 5, 7]  # personas, bicicletas, autos, motos, buses, camiones

cap = cv2.VideoCapture("20251030_134222_8f9126cb_input.MOV")
assert cap.isOpened(), "Error reading video file"

# Obtener dimensiones del video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Factor de redimensionamiento para mejor calidad (50% del tamaño original)
resize_factor = 0.5
proc_w = int(w * resize_factor)
proc_h = int(h * resize_factor)

# Rectángulo vertical centrado horizontalmente
# Ancho del rectángulo: 40 píxeles (20 a cada lado del centro)
rect_width = 20
center_x = int(proc_w / 2)
region_points = [
    (center_x - rect_width, 0),           # superior izquierda
    (center_x + rect_width, 0),           # superior derecha
    (center_x + rect_width, proc_h),      # inferior derecha
    (center_x - rect_width, proc_h)       # inferior izquierda
]

# Video writer con mejor codec para menor tamaño de archivo
# Intentar H.264 primero (mejor compresión), fallback a mp4v
try:
    video_writer = cv2.VideoWriter("object_counting_output.mp4", cv2.VideoWriter_fourcc(*"avc1"), fps, (proc_w, proc_h))
    if not video_writer.isOpened():
        raise Exception("H.264 no disponible")
    codec_usado = "H.264 (avc1)"
except:
    video_writer = cv2.VideoWriter("object_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (proc_w, proc_h))
    codec_usado = "MPEG-4 (mp4v)"

print(f"🎥 Codec de salida: {codec_usado}")

# Inicializar ObjectCounter con todas las clases configuradas
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolo11n.pt",
    classes=CLASSES_TO_DETECT,
    tracker="botsort.yaml",
    show_in=True,   # Mostrar conteo de entradas
    show_out=True,  # Mostrar conteo de salidas
    line_width=2
)

# Procesar video (SIN salto de frames para tracking preciso)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0
last_results = None  # Guardar el último resultado para extraer conteos

print(f"\n🎬 Procesando video: {total_frames} frames")
print(f"⚙️ Configuración: {proc_w}x{proc_h} @ {fps} fps")

while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("\n✅ Video frame is empty or video processing has been successfully completed.")
        break

    frame_num += 1

    # Mostrar progreso cada 30 frames
    if frame_num % 30 == 0:
        progress = (frame_num / total_frames) * 100
        print(f"📊 Progreso: {progress:.1f}% ({frame_num}/{total_frames} frames)", end='\r')

    # Redimensionar frame para mejor rendimiento
    resized_frame = cv2.resize(im0, (proc_w, proc_h))

    # Realizar detección y conteo en TODOS los frames
    results = counter(resized_frame)
    last_results = results  # Guardar último resultado

    # Escribir frame procesado
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Guardar resultados en JSON
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

processing_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Obtener conteos por clase usando múltiples métodos (como en app/services/video_processor.py)
conteo_por_clase = {}

# Método 1: Desde last_results.classwise_count (más confiable)
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

# Método 2: Desde counter.classwise_count (fallback)
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

results_data = {
    "processing_id": processing_id,
    "fecha_procesamiento": datetime.now().isoformat(),
    "video_entrada": "20251030_134222_8f9126cb_input.MOV",
    "video_salida": "object_counting_output.mp4",
    "configuracion": {
        "resize_factor": resize_factor,
        "rect_width": rect_width,
        "tracker": "botsort.yaml",
        "modelo": "yolo11n.pt",
        "codec": codec_usado,
        "clases_detectadas": {CLASS_NAMES[c]: c for c in CLASSES_TO_DETECT}
    },
    "conteo_total": {
        "in_count": counter.in_count if hasattr(counter, 'in_count') else 0,
        "out_count": counter.out_count if hasattr(counter, 'out_count') else 0,
        "total": (counter.in_count + counter.out_count) if hasattr(counter, 'in_count') and hasattr(counter, 'out_count') else 0
    },
    "conteo_por_clase": conteo_por_clase if conteo_por_clase else {
        "person": {"in": 0, "out": 0, "total": 0},
        "bicycle": {"in": 0, "out": 0, "total": 0},
        "car": {"in": 0, "out": 0, "total": 0},
        "motorcycle": {"in": 0, "out": 0, "total": 0},
        "bus": {"in": 0, "out": 0, "total": 0},
        "truck": {"in": 0, "out": 0, "total": 0}
    }
}

# Guardar JSON
json_file = output_dir / f"results_{processing_id}.json"
with open(json_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\n✅ Resultados guardados en: {json_file}")
print("\n" + "="*60)
print("📊 RESUMEN DEL PROCESAMIENTO")
print("="*60)
print(f"📁 Video entrada: {results_data['video_entrada']}")
print(f"📁 Video salida: {results_data['video_salida']}")
print(f"🔢 Total frames: {total_frames}")
print("\n📈 CONTEO TOTAL:")
print(f"  ➡️  IN (izq→der): {results_data['conteo_total']['in_count']}")
print(f"  ⬅️  OUT (der→izq): {results_data['conteo_total']['out_count']}")
print(f"  🎯 Total: {results_data['conteo_total']['total']}")

print("\n🚗 CONTEO POR CLASE:")
emojis = {
    "person": "🚶",
    "bicycle": "🚲",
    "car": "🚗",
    "motorcycle": "🏍️",
    "bus": "🚌",
    "truck": "🚛"
}
for clase, counts in results_data['conteo_por_clase'].items():
    emoji = emojis.get(clase, "📦")
    if isinstance(counts, dict):
        total = counts.get('total', 0)
        in_count = counts.get('in', 0)
        out_count = counts.get('out', 0)
        print(f"  {emoji} {clase.capitalize()}: {total} (IN: {in_count}, OUT: {out_count})")
    else:
        print(f"  {emoji} {clase.capitalize()}: {counts}")

print(f"\n📝 Reporte JSON: {json_file}")
print("="*60)