# Detector de Espacios de Parking con YOLOv8

Este proyecto utiliza YOLOv8 para detectar espacios de estacionamiento libres y ocupados a partir de imágenes o videos, especialmente útil para análisis con drones.

## 🚀 Instalación

1.  **Clona el repositorio:**

    ```bash
    git clone [https://www.youtube.com/watch?v=W3ARA19UB4w](https://www.youtube.com/watch?v=W3ARA19UB4w)
    cd [Nombre de tu repositorio]
    ```

2.  **Crea y activa un entorno virtual:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Descarga el modelo entrenado:**
    - Ve a la sección de "Releases" de este repositorio.
    - Descarga el archivo `best.pt`.
    - Crea la siguiente estructura de carpetas `runs/detect/parking_yolov8_run/weights/` y coloca el archivo `best.pt` dentro.

## 🏃‍♂️ Cómo Usarlo

Usa el script `predict.py` desde la terminal, pasándole la ruta a una imagen o video.

**Para una imagen:**

```bash
python predict.py ruta/a/tu/imagen.jpg
```
