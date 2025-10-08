# Detector de Espacios de Parking con YOLOv8

Este proyecto utiliza YOLOv8 para detectar espacios de estacionamiento libres y ocupados a partir de imágenes o videos, especialmente útil para análisis con drones.

---

## 🚀 Instalación

Sigue estos pasos para poner en marcha el proyecto en tu computadora.

1.  **Clona este repositorio de código:**
    _Abre una terminal y ejecuta el siguiente comando._

    ```bash
    git clone [https://github.com/gonzalo-agostino/detector-parking-yolo.git](https://github.com/gonzalo-agostino/detector-parking-yolo.git)
    cd detector-parking-yolo
    ```

2.  **Crea y activa un entorno virtual:**

    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instala las dependencias necesarias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Descarga el modelo entrenado:**
    - En este repositorio de GitHub, ve a la sección de **"Releases"** (en el panel de la derecha).
    - Descarga el archivo `best.pt`.
    - Crea la siguiente estructura de carpetas: `runs/detect/parking_yolov8_run/weights/`.
    - Coloca el archivo `best.pt` que descargaste dentro de esa carpeta `weights`.

---

## 🏃‍♂️ Cómo Usarlo

Usa el script `predict.py` desde la terminal, pasándole la ruta a una imagen o video que quieras analizar.

**Para analizar una imagen:**

```bash
python predict.py ruta/a/tu/imagen.jpg
```
