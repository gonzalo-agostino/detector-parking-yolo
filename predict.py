# predict.py (Versión Inteligente para Imágenes y Videos)
from ultralytics import YOLO
import os
import sys # Importamos la librería 'sys' para leer argumentos de la terminal

def main():
    # --- VERIFICACIÓN INICIAL ---
    
    # Comprobamos si el usuario ha proporcionado un archivo para analizar
    if len(sys.argv) < 2:
        print("\n¡Error! Debes especificar la ruta a una imagen o video.")
        print("Uso: python predict.py ruta/al/archivo.mp4")
        print("Ejemplo de imagen: python predict.py datasets/test/images/201.jpg")
        print("Ejemplo de video: python predict.py videos_de_prueba/mi_video_dron.mp4")
        print("Para usar la cámara web: python predict.py 0\n")
        return

    # El primer argumento después del nombre del script es la ruta al archivo
    source_path = sys.argv[1]

    # --- CONFIGURACIÓN ---
    
    # 1. RUTA AL MODELO ENTRENADO
    # Asegúrate de que apunte a tu archivo 'best.pt'
    MODEL_PATH = os.path.join('runs', 'detect', 'parking_yolov8_run', 'weights', 'best.pt')
    
    # --- LÓGICA DE PREDICCIÓN ---
    
    print(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo. Verifica la ruta: {e}")
        return

    print(f"Iniciando la predicción en la fuente: {source_path}")
    
    # Realiza la predicción. Los parámetros son los mismos para ambos tipos de archivo,
    # YOLO es lo suficientemente inteligente para manejarlos.
    results = model.predict(
        source=source_path,
        show=True,        # Muestra la ventana de resultados en vivo.
        conf=0.5,         # Umbral de confianza del 50%.
        save=True         # Guarda el resultado (imagen o video).
    )
    
    # El bucle es opcional pero se recomienda para asegurar que los videos se procesen completamente.
    for result in results:
        pass

    print("\nPredicción finalizada.")
    print("El resultado se ha guardado en la carpeta 'runs/detect/predict'.")

if __name__ == '__main__':
    main()