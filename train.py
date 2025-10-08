# train.py
from ultralytics import YOLO

def main():
    # Carga un modelo YOLOv11 pre-entrenado. 'yolov11n.pt' es la versión "nano",
    # es la más ligera y rápida, ideal para empezar.
    model = YOLO('yolov8n.pt')

    # Inicia el entrenamiento del modelo
    print("Iniciando el entrenamiento del modelo de detección de parking...")
    results = model.train(
        data='parking_data.yaml',  # Tu archivo de configuración del dataset
        epochs=50,                # Número de veces que el modelo verá el dataset completo.
                                  # Puedes empezar con un número menor (ej. 25) para una prueba rápida.
        imgsz=640,                # Tamaño al que se redimensionarán las imágenes
        batch=8,                  # Cuántas imágenes procesar a la vez. Si tienes un error de memoria (CUDA out of memory),
                                  # reduce este número a 4 o incluso 2.
        name='parking_yolov11_run' # Un nombre para la carpeta de resultados
    )
    print("¡Entrenamiento finalizado exitosamente!")
    # La ruta al modelo entrenado se puede encontrar en los resultados,
    # usualmente en 'runs/detect/parking_yolov11_run/weights/best.pt'
    print(f"El mejor modelo ha sido guardado en la carpeta: {results.save_dir}")

if __name__ == '__main__':
    main()