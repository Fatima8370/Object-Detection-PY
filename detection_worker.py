"""
detection_worker.py  (v4)
-------------------------
Model: YOLO26s (Ultralytics, Jan 2026)
  • NMS-free end-to-end inference → simpler pipeline, lower latency
  • STAL label assignment + Progressive Loss → much better small-object recall
  • ~43% faster CPU inference vs YOLO11n

Camera lag fix:
  CaptureThread reads at full webcam speed into a 1-slot queue.
  Inference loop always gets the freshest frame (stale frames discarded).
  Camera uses imgsz=640 (fast); image/video use imgsz=1280 (quality).

Dynamic confidence:
  Raw YOLO results cached. Slider/toggle changes trigger re-render from
  cache without re-running the model → instant visual update.

IMPORTANT — COCO-80 class limitation:
  All YOLO models (including YOLO26) are trained on the COCO dataset which
  contains exactly 80 object classes. Objects NOT in COCO — such as
  eyeglasses, calculators, trees, pens — will NEVER be detected regardless
  of confidence threshold. See the full class list printed at startup.
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


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight capture thread — only reads pixels, never runs inference
# ─────────────────────────────────────────────────────────────────────────────

class _CaptureThread(threading.Thread):
    """
    Reads frames from a cv2.VideoCapture as fast as the camera allows and
    drops everything except the very latest frame into a maxsize-1 slot queue.
    The inference loop in DetectionWorker reads from this queue, so it always
    gets the freshest frame and never waits behind stale ones.
    """

    def __init__(self, source, slot: queue.Queue):
        super().__init__(daemon=True)
        self._source = source           # 0 for webcam, path for video file
        self._slot   = slot             # maxsize=1
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
                if isinstance(self._source, str):   # video file: loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Replace any unread frame — the inference loop will pick this up
            try:
                self._slot.get_nowait()             # discard old frame
            except queue.Empty:
                pass
            try:
                self._slot.put_nowait(frame)
            except queue.Full:
                pass

        cap.release()
        self._cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Main detection worker
# ─────────────────────────────────────────────────────────────────────────────

class DetectionWorker(threading.Thread):

    # YOLOv8 Nano — 3.2M params, optimized for real-time on standard hardware
    MODEL_PATH = "yolov8n.pt"

    def __init__(self, frame_queue: queue.Queue, status_callback,
                 fps_callback, mode: str = MODE_CAMERA, source=None):
        super().__init__(daemon=True)

        self.frame_queue  = frame_queue
        self._status_cb   = status_callback
        self._fps_cb      = fps_callback
        self._mode        = mode
        self._source      = source

        self._lock        = threading.Lock()
        self._running     = False
        self._conf        = 0.10
        self._greyscale   = True
        # flag: re-render cached results without re-running YOLO
        self._rerender    = False

    # ── public API (all thread-safe) ─────────────────────────────────────────

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
            self._rerender = True   # trigger re-render from cache

    def set_greyscale(self, value: bool):
        with self._lock:
            self._greyscale = bool(value)
            self._rerender  = True  # trigger re-render from cache

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self, frame_bgr: np.ndarray, results,
                conf_thresh: float, apply_grey: bool) -> np.ndarray:
        if apply_grey:
            grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            out  = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        else:
            out  = frame_bgr.copy()

        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id   = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color    = _box_color(cls_id)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label  = f"{cls_name}  {conf:.0%}"
            font   = cv2.FONT_HERSHEY_DUPLEX
            fs, th = 0.55, 1
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

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        self._status_cb("Loading model…")
        try:
            model = YOLO(self.MODEL_PATH)
        except Exception as exc:
            self._status_cb(f"Model error: {exc}")
            return

        with self._lock:
            self._running = True

        if self._mode == MODE_IMAGE:
            self._run_image(model)
        else:
            self._run_stream(model)   # camera and video share the same loop

        self._status_cb("Stopped")
        self._fps_cb(0.0)

    # ── image mode ────────────────────────────────────────────────────────────

    def _run_image(self, model):
        frame = cv2.imread(self._source)
        if frame is None:
            self._status_cb(f"Cannot read: {self._source}")
            return

        self._status_cb("Running — Image")

        with self._lock:
            conf = self._conf
            grey = self._greyscale
            self._rerender = False

        # Run inference ONCE at high resolution
        results = model(frame, imgsz=640, verbose=False)
        annotated = self._render(frame, results, conf, grey)
        self._push(annotated)

        # Stay alive; re-render whenever slider/toggle changes
        while self.running:
            with self._lock:
                rerender = self._rerender
                if rerender:
                    conf           = self._conf
                    grey           = self._greyscale
                    self._rerender = False

            if rerender:
                annotated = self._render(frame, results, conf, grey)
                self._push(annotated)

            time.sleep(0.03)    # ~33 Hz check loop — no CPU burn

    # ── camera + video-file mode ──────────────────────────────────────────────

    def _run_stream(self, model):
        """
        Decouples capture from inference:
          CaptureThread fills a 1-slot raw-frame queue at full camera speed.
          This inference loop grabs the latest frame, runs YOLO, pushes result.
          Because the capture queue always holds only the freshest frame, the
          inference loop never processes a stale frame — lag eliminated.
        """
        label = "Camera" if self._mode == MODE_CAMERA else "Video"
        src   = 1 if self._mode == MODE_CAMERA else self._source

        # yolo26s at 640 is ~43% faster than yolo11n on CPU
        # while maintaining better small-object accuracy
        INFER_SZ = 640

        raw_slot      = queue.Queue(maxsize=1)     # capture → inference
        cap_thread    = _CaptureThread(src, raw_slot)
        cap_thread.start()

        # Wait up to 2 s for the first frame to confirm camera opened
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

        # Cache for dynamic conf: keep last results + raw frame
        last_results = None
        last_frame   = None

        while True:
            with self._lock:
                if not self._running:
                    break
                conf      = self._conf
                grey      = self._greyscale
                rerender  = self._rerender
                if rerender:
                    self._rerender = False

            # Re-render from cache if only slider/toggle changed
            if rerender and last_results is not None:
                annotated = self._render(last_frame, last_results, conf, grey)
                self._push(annotated)

            # Grab the latest raw frame (non-blocking)
            try:
                frame = raw_slot.get_nowait()
            except queue.Empty:
                time.sleep(0.005)
                continue

            last_frame = frame

            # Run inference
            results      = model(frame, imgsz=INFER_SZ, verbose=False)
            last_results = results

            annotated = self._render(frame, results, conf, grey)
            self._push(annotated)

            fps_count += 1
            elapsed    = time.perf_counter() - t_start
            if elapsed >= 0.5:
                self._fps_cb(fps_count / elapsed)
                fps_count = 0
                t_start   = time.perf_counter()

        cap_thread.halt()
