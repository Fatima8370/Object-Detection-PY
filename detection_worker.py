"""
detection_worker.py
-------------------
Background worker thread that owns the camera and YOLO model.
It runs entirely off the main (GUI) thread so the UI never freezes.

Thread-safety contract:
  - The worker writes processed frames into `self.frame_queue` (maxsize=1).
    A maxsize of 1 acts as a back-pressure valve: if the GUI hasn't consumed
    the last frame yet, the worker discards the new frame rather than letting
    RAM grow unboundedly (no memory leak).
  - The worker reads `self.conf_threshold` and `self.running` through a
    threading.Lock so updates from the GUI slider are applied atomically.
"""

import threading
import queue
import time
import csv
import os
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# ── colour palette for bounding boxes (BGR) ──────────────────────────────────
_PALETTE = [
    (0, 200, 255), (0, 255, 128), (255, 80,  80),  (255, 200,  0),
    (180,  0, 255), (0, 180, 255), (255, 128,  0),  (50, 255, 200),
]

def _box_color(class_id: int) -> tuple:
    return _PALETTE[int(class_id) % len(_PALETTE)]


class DetectionWorker(threading.Thread):
    """
    A daemon thread that:
      1. Opens the webcam.
      2. Passes each grayscale frame through YOLO.
      3. Draws custom bounding boxes with cv2 (no .plot()).
      4. Pushes the annotated frame into a single-slot queue for the GUI.
      5. Logs high-confidence detections to CSV.
    """

    MODEL_PATH = "yolo11n.pt"   # auto-downloaded on first run
    LOG_FILE   = "detections_log.csv"

    def __init__(self, frame_queue: queue.Queue, status_callback, fps_callback):
        super().__init__(daemon=True)   # daemon=True → thread dies when GUI closes

        # ── shared state ─────────────────────────────────────────────────────
        self.frame_queue   = frame_queue   # maxsize=1; GUI reads from this
        self._lock         = threading.Lock()
        self._running      = False
        self._conf         = 0.50          # default confidence threshold

        # ── callbacks → GUI updates status label / FPS label ─────────────────
        self._status_cb = status_callback
        self._fps_cb    = fps_callback

        # ── CSV log setup ─────────────────────────────────────────────────────
        self._csv_lock = threading.Lock()
        self._init_csv()

    # ── public properties (thread-safe) ──────────────────────────────────────

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def set_conf(self, value: float):
        """Called from the GUI thread when the slider moves."""
        with self._lock:
            self._conf = float(value)

    def stop(self):
        """Signal the worker to exit its loop gracefully."""
        with self._lock:
            self._running = False

    # ── CSV helpers ───────────────────────────────────────────────────────────

    def _init_csv(self):
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Class Name", "Confidence"])

    def _log_detection(self, class_name: str, confidence: float):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._csv_lock:
            with open(self.LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow([ts, class_name, f"{confidence:.4f}"])

    # ── frame rendering (manual cv2 drawing) ─────────────────────────────────

    def _render_detections(self, frame_bgr: np.ndarray, results, conf_thresh: float) -> np.ndarray:
        """
        Manually draw bounding boxes and labels using cv2 primitives.
        Does NOT call results[0].plot() — gives full control over style.
        """
        # Work on a copy so the original is untouched
        out = frame_bgr.copy()

        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes:
            conf  = float(box.conf[0])
            if conf < conf_thresh:
                continue

            cls_id    = int(box.cls[0])
            cls_name  = results[0].names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color     = _box_color(cls_id)

            # ── bounding box ─────────────────────────────────────────────────
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

            # ── label background pill ────────────────────────────────────────
            label     = f"{cls_name}  {conf:.0%}"
            font      = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness  = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx, ly = x1, max(y1 - 6, th + 6)
            cv2.rectangle(out, (lx, ly - th - baseline - 4),
                          (lx + tw + 8, ly + baseline - 2), color, cv2.FILLED)
            cv2.putText(out, label, (lx + 4, ly - 2),
                        font, font_scale, (15, 15, 15), thickness, cv2.LINE_AA)

            # ── CSV log ──────────────────────────────────────────────────────
            self._log_detection(cls_name, conf)

        return out

    # ── main thread loop ──────────────────────────────────────────────────────

    def run(self):
        self._status_cb("Loading model…")

        try:
            model = YOLO(self.MODEL_PATH)
        except Exception as exc:
            self._status_cb(f"Model error: {exc}")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._status_cb("Camera error: cannot open device 0")
            return

        with self._lock:
            self._running = True

        fps_counter = 0
        fps_display = 0.0
        t_start     = time.perf_counter()

        self._status_cb("Running…")

        while True:
            # ── check stop signal ─────────────────────────────────────────────
            with self._lock:
                if not self._running:
                    break
                conf_thresh = self._conf

            ret, frame_bgr = cap.read()
            if not ret:
                self._status_cb("Camera read failed — stopping.")
                break

            # ── greyscale conversion (requirement) ──────────────────────────
            # Convert to greyscale, then back to BGR so cv2 drawing still works
            grey        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_grey3 = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            # ── YOLO inference (verbose=False keeps console clean) ────────────
            results = model(frame_grey3, verbose=False)

            # ── custom rendering ──────────────────────────────────────────────
            annotated = self._render_detections(frame_grey3, results, conf_thresh)

            # ── push frame to GUI queue (drop if GUI is behind) ──────────────
            # put_nowait raises queue.Full if slot occupied → we discard.
            # This prevents unbounded memory growth if GUI can't keep up.
            try:
                self.frame_queue.put_nowait(annotated)
            except queue.Full:
                pass

            # ── FPS calculation ───────────────────────────────────────────────
            fps_counter += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 0.5:                         # update every 0.5 s
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_start     = time.perf_counter()
                self._fps_cb(fps_display)

        # ── cleanup ───────────────────────────────────────────────────────────
        cap.release()
        self._status_cb("Camera Stopped")
        self._fps_cb(0.0)
