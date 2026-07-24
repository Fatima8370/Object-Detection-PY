"""
detection_worker.py  (v5 - LRF & Model Switching)
-------------------------------------------------
Features:
  - Multi-threaded background worker for YOLO Object Detection.
  - Dynamic model reloading at runtime (e.g. switching between yolov8n.pt and best.pt).
  - Boresight Crosshair Overlay: Renders fixed reticle at frame center (w//2, h//2).
  - Laser Range Finder (LRF) Distance Overlay: Displays live distance measurement
    on the reticle HUD and attaches distance metadata to targeted objects.
  - Lag-free CaptureThread: Single-slot raw frame queue for zero camera stream delay.
"""

import threading
import queue
import time
import cv2
import numpy as np
from ultralytics import YOLO


_PALETTE = [
    (0, 200, 255), (0, 255, 128), (255, 80,  80),  (255, 200,  0),
    (180,  0, 255), (0, 180, 255), (255, 128,  0),  (50, 255, 200),
]

def _box_color(class_id: int) -> tuple:
    return _PALETTE[int(class_id) % len(_PALETTE)]


MODE_CAMERA = "camera"
MODE_IMAGE  = "image"
MODE_VIDEO  = "video"


class _CaptureThread(threading.Thread):
    def __init__(self, source, slot: queue.Queue):
        super().__init__(daemon=True)
        self._source = source
        self._slot   = slot
        self._stop   = threading.Event()

    def halt(self):
        self._stop.set()

    def run(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            return

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                if isinstance(self._source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            try:
                self._slot.get_nowait()
            except queue.Empty:
                pass
            try:
                self._slot.put_nowait(frame)
            except queue.Full:
                pass

        cap.release()


class DetectionWorker(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, status_callback,
                 fps_callback, mode: str = MODE_CAMERA, source=None,
                 model_path: str = "yolov8n.pt"):
        super().__init__(daemon=True)

        self.frame_queue  = frame_queue
        self._status_cb   = status_callback
        self._fps_cb      = fps_callback
        self._mode        = mode
        self._source      = source
        self._model_path  = model_path

        self._lock        = threading.Lock()
        self._running     = False
        self._conf        = 0.10
        self._greyscale   = True
        self._rerender    = False
        self._reload_model = False
        self._lrf_distance_str = "--- m"

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def stop(self):
        with self._lock:
            self._running = False

    def set_conf(self, value: float):
        with self._lock:
            self._conf     = float(value)
            self._rerender = True

    def set_greyscale(self, value: bool):
        with self._lock:
            self._greyscale = bool(value)
            self._rerender  = True

    def set_model_path(self, model_path: str):
        with self._lock:
            if self._model_path != model_path:
                self._model_path = model_path
                self._reload_model = True
                self._rerender = True

    def set_lrf_distance(self, distance_str: str):
        with self._lock:
            self._lrf_distance_str = distance_str
            self._rerender = True

    def _render(self, frame_bgr: np.ndarray, results,
                conf_thresh: float, apply_grey: bool, lrf_dist: str) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2

        if apply_grey:
            grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            out  = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        else:
            out  = frame_bgr.copy()

        font = cv2.FONT_HERSHEY_DUPLEX

        # ── Draw Fixed Boresight Crosshair at Frame Center ───────────────────
        ch_color = (0, 255, 255)  # Cyan/Yellow BGR
        arm = 22
        gap = 5
        
        cv2.circle(out, (cx, cy), 16, ch_color, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, (0, 255, 255), -1)

        cv2.line(out, (cx - arm, cy), (cx - gap, cy), ch_color, 2, cv2.LINE_AA)
        cv2.line(out, (cx + gap, cy), (cx + arm, cy), ch_color, 2, cv2.LINE_AA)
        cv2.line(out, (cx, cy - arm), (cx, cy - gap), ch_color, 2, cv2.LINE_AA)
        cv2.line(out, (cx, cy + gap), (cx, cy + arm), ch_color, 2, cv2.LINE_AA)

        # ── Render LRF Distance HUD Overlay near Crosshair ────────────────────
        hud_label = f"RANGE: {lrf_dist}"
        fs, th = 0.5, 1
        (tw, lh), bl = cv2.getTextSize(hud_label, font, fs, th)
        hx, hy = cx + 24, cy + 6
        cv2.rectangle(out, (hx - 4, hy - lh - 4), (hx + tw + 6, hy + bl + 2), (15, 15, 20), cv2.FILLED)
        cv2.rectangle(out, (hx - 4, hy - lh - 4), (hx + tw + 6, hy + bl + 2), ch_color, 1, cv2.LINE_AA)
        cv2.putText(out, hud_label, (hx, hy), font, fs, (0, 255, 255), th, cv2.LINE_AA)

        # ── Render Detected Objects & Crosshair Alignment ─────────────────────
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_thresh:
                    continue
                cls_id   = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Check if this object box covers the boresight center crosshair
                is_targeted = (x1 <= cx <= x2) and (y1 <= cy <= y2)

                if is_targeted:
                    color = (0, 255, 255)  # Bright cyan highlight
                    thick = 3
                    label = f"🎯 TARGET: {cls_name} {conf:.0%} [{lrf_dist}]"
                else:
                    color = _box_color(cls_id)
                    thick = 2
                    label = f"{cls_name}  {conf:.0%}"

                cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

                (tw, lh), bl = cv2.getTextSize(label, font, fs, th)
                lx, ly = x1, max(y1 - 6, lh + 6)
                cv2.rectangle(out, (lx, ly - lh - bl - 4),
                              (lx + tw + 8, ly + bl - 2), color, cv2.FILLED)
                cv2.putText(out, label, (lx + 4, ly - 2),
                            font, fs, (15, 15, 15), th, cv2.LINE_AA)

        return out

    def _push(self, frame: np.ndarray):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def run(self):
        with self._lock:
            model_path = self._model_path

        self._status_cb(f"Loading model ({model_path})…")
        try:
            model = YOLO(model_path)
        except Exception as exc:
            self._status_cb(f"Model error: {exc}")
            return

        with self._lock:
            self._running = True

        if self._mode == MODE_IMAGE:
            self._run_image(model)
        else:
            self._run_stream(model)

        self._status_cb("Stopped")
        self._fps_cb(0.0)

    def _run_image(self, model):
        frame = cv2.imread(self._source)
        if frame is None:
            self._status_cb(f"Cannot read: {self._source}")
            return

        self._status_cb("Running — Image")

        while self.running:
            with self._lock:
                conf = self._conf
                grey = self._greyscale
                lrf_dist = self._lrf_distance_str
                reload_mod = self._reload_model
                path = self._model_path
                self._rerender = False

            if reload_mod:
                with self._lock:
                    self._reload_model = False
                self._status_cb(f"Reloading model ({path})…")
                try:
                    model = YOLO(path)
                    self._status_cb(f"Running — Image ({path})")
                except Exception as exc:
                    self._status_cb(f"Model reload error: {exc}")

            results = model(frame, imgsz=640, verbose=False)
            annotated = self._render(frame, results, conf, grey, lrf_dist)
            self._push(annotated)
            time.sleep(0.05)

    def _run_stream(self, model):
        label = "Camera" if self._mode == MODE_CAMERA else "Video"
        if self._mode == MODE_CAMERA:
            try:
                src = int(self._source) if self._source is not None else 0
            except (ValueError, TypeError):
                src = 0
        else:
            src = self._source

        INFER_SZ = 640

        raw_slot   = queue.Queue(maxsize=1)
        cap_thread = _CaptureThread(src, raw_slot)
        cap_thread.start()

        try:
            _ = raw_slot.get(timeout=3.0)
            raw_slot.put_nowait(_)
        except queue.Empty:
            self._status_cb(f"{label} error: no frames received")
            cap_thread.halt()
            return

        self._status_cb(f"Running — {label}")

        fps_count = 0
        t_start   = time.perf_counter()
        last_results = None
        last_frame   = None

        while True:
            with self._lock:
                if not self._running:
                    break
                conf       = self._conf
                grey       = self._greyscale
                lrf_dist   = self._lrf_distance_str
                rerender   = self._rerender
                reload_mod = self._reload_model
                path       = self._model_path
                if rerender:
                    self._rerender = False

            if reload_mod:
                with self._lock:
                    self._reload_model = False
                self._status_cb(f"Reloading model ({path})…")
                try:
                    model = YOLO(path)
                    self._status_cb(f"Running — {label} ({path})")
                except Exception as exc:
                    self._status_cb(f"Model reload error: {exc}")

            if rerender and last_results is not None:
                annotated = self._render(last_frame, last_results, conf, grey, lrf_dist)
                self._push(annotated)

            try:
                frame = raw_slot.get_nowait()
            except queue.Empty:
                time.sleep(0.005)
                continue

            last_frame = frame
            results      = model(frame, imgsz=INFER_SZ, verbose=False)
            last_results = results

            annotated = self._render(frame, results, conf, grey, lrf_dist)
            self._push(annotated)

            fps_count += 1
            elapsed    = time.perf_counter() - t_start
            if elapsed >= 0.5:
                self._fps_cb(fps_count / elapsed)
                fps_count = 0
                t_start   = time.perf_counter()

        cap_thread.halt()
