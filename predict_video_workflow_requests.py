# Vídeo/Webcam → Roboflow Rapid Workflow (requests + base64)
# Ejemplos MUY cortos:
#   python predict_video_workflow_requests.py .\videos\video2.mp4
#   python predict_video_workflow_requests.py 0 -p strict90        # webcam, umbral 0.90
#   python predict_video_workflow_requests.py .\videos\v.mp4 -p recall -o out.mp4
#   python predict_video_workflow_requests.py .\videos\v.mp4 -p balanced -r 100,300,1800,1000 --classes car
import os, cv2, time, argparse, base64, csv, requests

API_URL   = "https://serverless.roboflow.com"
WORKSPACE = "parkdetectioni"
WORKFLOW  = "find-cars-2"

# -------------------- Args --------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Roboflow Rapid (workflow) sobre video/webcam con presets simples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # FUENTE POSICIONAL (más simple)
    ap.add_argument("source", help="Ruta a video o índice webcam (0,1,...)")
    # Compatibilidad con tu flag anterior (--source)
    ap.add_argument("--source", dest="source_flag", help=argparse.SUPPRESS)

    ap.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY",""), help="API key (o setear ROBOFLOW_API_KEY)")
    ap.add_argument("-p","--preset", choices=["balanced","recall","strict90"], default="balanced", help="Perfil de parámetros listo")
    ap.add_argument("--max-size", type=int, default=None, help="Lado máx. enviado (auto por preset)")
    ap.add_argument("--fps-step", type=int, default=None, help="Procesar 1 de cada N frames (auto por preset)")

    # Salidas rápidas
    ap.add_argument("-o","--out", default="salida_workflow.mp4", help="MP4 anotado")
    ap.add_argument("-c","--csv", default="", help="CSV opcional (frame,timestamp_ms,count,state)")

    # Visual
    ap.add_argument("--draw-label", action="store_true", help="Dibujar clase+confianza")

    # ROI rápida: "x1,y1,x2,y2"
    ap.add_argument("-r","--roi", default="", help="ROI rect en formato x1,y1,x2,y2")
    # ROI detallada (compatibilidad)
    ap.add_argument("--roi-rect", type=int, nargs=4, metavar=("X1","Y1","X2","Y2"), help=argparse.SUPPRESS)

    # Filtros (pueden quedar vacíos si usás preset)
    ap.add_argument("--min-conf", type=float, default=None, help="Confianza mínima [0–1] (auto por preset)")
    ap.add_argument("--min-area-frac", type=float, default=None, help="Área mínima relativa (auto por preset)")
    ap.add_argument("--ar-min", type=float, default=None, help="Relación de aspecto mínima w/h (auto por preset)")
    ap.add_argument("--ar-max", type=float, default=None, help="Relación de aspecto máxima w/h (auto por preset)")
    ap.add_argument("--k-on", type=int, default=None, help="Frames para activar estado (auto por preset)")
    ap.add_argument("--k-off", type=int, default=None, help="Frames para desactivar estado (auto por preset)")

    # Filtro por clases (opcional). Ej: --classes car  |  --classes car,auto,coche
    ap.add_argument("--classes", default="car", help="Lista de clases permitidas separadas por coma. Vacío = no filtra.")

    args = ap.parse_args()

    # Si alguien pasó --source viejo, priorizarlo
    if args.source_flag is not None:
        args.source = args.source_flag

    # Parse ROI rápida si viene
    if args.roi and (args.roi_rect is None):
        try:
            x1,y1,x2,y2 = [int(v) for v in args.roi.split(",")]
            args.roi_rect = (x1,y1,x2,y2)
        except Exception:
            raise SystemExit("[ERROR] Formato de --roi inválido. Usa x1,y1,x2,y2")

    # Clases
    args.classes = [s.strip().lower() for s in args.classes.split(",") if s.strip()]  # puede quedar []

    # Aplicar preset por defecto y permitir override con flags
    apply_preset(args)
    return args

def apply_preset(args):
    # Presets: valores base
    presets = {
        "balanced":  dict(max_size=1280, fps_step=3, min_conf=0.55, min_area_frac=0.00025, ar_min=0.4, ar_max=2.5, k_on=2, k_off=3),
        "recall":    dict(max_size=1600, fps_step=1, min_conf=0.40, min_area_frac=0.00015, ar_min=0.3, ar_max=3.0, k_on=2, k_off=3),
        "strict90":  dict(max_size=1280, fps_step=2, min_conf=0.90, min_area_frac=0.00025, ar_min=0.4, ar_max=2.5, k_on=2, k_off=3),
    }
    base = presets[args.preset]
    # Solo rellenar los que vengan en None (si el usuario pasa un flag, respétalo)
    for k, v in base.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

# -------------------- Core --------------------
def resize_keep_aspect(img, max_dim=1280):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = float(max_dim) / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def post_workflow(frame_bgr, api_key, max_dim=1280, timeout=30):
    small = resize_keep_aspect(frame_bgr, max_dim=max_dim)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return []
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    url = f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW}"
    payload = {"api_key": api_key, "inputs": {"image": {"type": "base64", "value": b64}}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    outputs = data.get("outputs", [])
    if not outputs:
        return []
    preds_block = outputs[0].get("predictions", {})
    preds = preds_block.get("predictions", []) or []

    # Reescalar a tamaño original
    sx = frame_bgr.shape[1] / float(small.shape[1])
    sy = frame_bgr.shape[0] / float(small.shape[0])
    for p in preds:
        p["x"] *= sx; p["y"] *= sy
        p["width"] *= sx; p["height"] *= sy
    return preds

# --------- Filtros / ROI / Dibujo ----------
def in_roi(xc, yc, rect):
    x1,y1,x2,y2 = rect
    return (x1 <= xc <= x2) and (y1 <= yc <= y2)

def filter_preds(preds, frame_w, frame_h, *, min_conf, min_area_frac, ar_min, ar_max, roi_rect=None, classes=None):
    min_area = frame_w * frame_h * (min_area_frac or 0.0)
    out = []
    for p in preds:
        conf = float(p.get("confidence", 0.0))
        if conf < float(min_conf):
            continue
        # Filtro por clase (si se especificó)
        if classes:
            cls = str(p.get("class","")).lower()
            if cls not in classes:
                continue

        w, h = float(p["width"]), float(p["height"])
        if (w * h) < min_area:
            continue
        if h <= 1e-6:
            continue
        ar = w / h
        if ar < float(ar_min) or ar > float(ar_max):
            continue
        if roi_rect is not None:
            xc, yc = float(p["x"]), float(p["y"])
            if not in_roi(xc, yc, roi_rect):
                continue
        out.append(p)
    return out

def draw_preds(frame, preds, draw_label=False):
    for p in preds:
        x,y,w,h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        if draw_label:
            conf = float(p.get("confidence", 0.0))
            cls  = str(p.get("class", "obj"))
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)

# -------------------- Main --------------------
if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise SystemExit("[ERROR] Falta API key. Pasá --api-key o seteá ROBOFLOW_API_KEY.")

    # Fuente (posicional)
    source = 0 if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] No se pudo abrir: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    # Estado ON/OFF (suavizado)
    streak_on = 0
    streak_off = 0
    state_active = False

    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame","timestamp_ms","count","state_active"])

    frame_idx = 0
    last_preds = []
    print(f"[INFO] preset={args.preset}  fps-step={args.fps_step}  max-size={args.max_size}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % int(args.fps_step) == 0:
            try:
                preds = post_workflow(frame, api_key=args.api_key, max_dim=int(args.max_size))
            except requests.HTTPError as e:
                print(f"[HTTP] {e.response.status_code}: {e.response.text[:200]}"); preds = []
            except Exception as e:
                print(f"[ERR] {e}"); preds = []
            last_preds = preds
        else:
            preds = last_preds

        roi_rect = tuple(args.roi_rect) if isinstance(args.roi_rect, tuple) else (tuple(args.roi_rect) if args.roi_rect else None)
        preds = filter_preds(
            preds, frame_w=W, frame_h=H,
            min_conf=args.min_conf, min_area_frac=args.min_area_frac,
            ar_min=args.ar_min, ar_max=args.ar_max,
            roi_rect=roi_rect, classes=args.classes
        )

        # Suavizado ON/OFF
        if len(preds) > 0:
            streak_on += 1; streak_off = 0
            if not state_active and streak_on >= int(args.k_on):
                state_active = True
        else:
            streak_off += 1; streak_on = 0
            if state_active and streak_off >= int(args.k_off):
                state_active = False

        # Dibujo ROI si hay
        if roi_rect:
            x1,y1,x2,y2 = roi_rect
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)

        # HUD
        cv2.putText(frame, f"count={len(preds)}  state={'ON' if state_active else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (50,220,50) if state_active else (0,0,255), 2)

        draw_preds(frame, preds, draw_label=args.draw_label)
        out.write(frame)
        cv2.imshow("Roboflow Rapid Workflow (requests)", frame)

        if csv_writer:
            ts_ms = int(1000.0 * (frame_idx / fps))
            csv_writer.writerow([frame_idx, ts_ms, len(preds), int(state_active)])

        if cv2.waitKey(1) & 0xFF == 27:
            break
        frame_idx += 1

    cap.release(); out.release(); cv2.destroyAllWindows()
    if csv_writer:
        csv_file.close()
    print(f"[OK] Video anotado: {args.out}")
    if args.csv:
        print(f"[OK] CSV: {args.csv}")
