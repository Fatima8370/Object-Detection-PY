"""
app_gui.py
----------
Main-thread GUI built with CustomTkinter.

GUI ↔ Worker communication model:
  ┌──────────────┐  frame_queue (maxsize=1)  ┌──────────────────┐
  │  Main Thread  │ ◄──────────────────────── │  Worker Thread   │
  │  (GUI/CTk)    │                           │  (Camera+YOLO)   │
  │               │  conf slider  ──────────► │                  │
  └──────────────┘                           └──────────────────┘

The GUI polls `frame_queue` every ~30 ms using `after()`.  This keeps
the Tkinter event loop 100% on the main thread while the worker runs
inference in the background — no freeze, no race conditions on the widget.
"""

import queue
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2

from detection_worker import DetectionWorker


# ── visual theme ─────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_BG          = "#0f0f13"
_PANEL_BG    = "#16161e"
_ACCENT      = "#00c8ff"
_ACCENT2     = "#7c3aed"
_TEXT        = "#e2e8f0"
_MUTED       = "#64748b"
_GREEN       = "#22c55e"
_RED         = "#ef4444"
_FONT_TITLE  = ("Helvetica", 22, "bold")
_FONT_BODY   = ("Helvetica", 13)
_FONT_MONO   = ("Courier New", 12)
_FONT_SMALL  = ("Helvetica", 11)


class ObjectDetectionApp(ctk.CTk):

    POLL_MS   = 30    # GUI polls frame queue every 30 ms (~33 fps max display)
    VIDEO_W   = 800
    VIDEO_H   = 560

    def __init__(self):
        super().__init__()

        self.title("Real-Time Object Detection  ·  YOLO11n")
        self.configure(fg_color=_BG)
        self.resizable(False, False)

        # ── shared queue — maxsize=1 is intentional (back-pressure) ──────────
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker: DetectionWorker  = None
        self._after_id = None          # handle for the polling callback

        self._build_ui()
        self._set_status("Camera Stopped", _MUTED)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────── UI layout ───────────────────────────────

    def _build_ui(self):
        # ── title bar ─────────────────────────────────────────────────────────
        title_bar = ctk.CTkFrame(self, fg_color=_PANEL_BG, corner_radius=0, height=56)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        ctk.CTkLabel(
            title_bar,
            text="⬡  Object Detection Studio",
            font=_FONT_TITLE,
            text_color=_ACCENT,
        ).pack(side="left", padx=24, pady=10)

        self._fps_label = ctk.CTkLabel(
            title_bar,
            text="FPS: —",
            font=_FONT_MONO,
            text_color=_MUTED,
        )
        self._fps_label.pack(side="right", padx=24)

        # ── main body ─────────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=_BG)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # left = video; right = controls
        self._build_video_panel(body)
        self._build_control_panel(body)

        # ── status bar ────────────────────────────────────────────────────────
        self._build_status_bar()

    def _build_video_panel(self, parent):
        vframe = ctk.CTkFrame(parent, fg_color=_PANEL_BG, corner_radius=12)
        vframe.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        # placeholder gradient canvas
        self._canvas = ctk.CTkLabel(
            vframe,
            text="",
            width=self.VIDEO_W,
            height=self.VIDEO_H,
            fg_color="#0a0a12",
            corner_radius=8,
        )
        self._canvas.pack(padx=12, pady=12)
        self._draw_placeholder()

    def _draw_placeholder(self):
        """Draw a grey gradient placeholder when no camera is active."""
        ph = np.zeros((self.VIDEO_H, self.VIDEO_W, 3), dtype=np.uint8)
        for y in range(self.VIDEO_H):
            v = int(20 + 15 * y / self.VIDEO_H)
            ph[y, :] = (v, v, v + 8)
        cv2.putText(ph, "Camera Offline", (self.VIDEO_W // 2 - 130, self.VIDEO_H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 70, 80), 2, cv2.LINE_AA)
        self._update_canvas(ph)

    def _build_control_panel(self, parent):
        cframe = ctk.CTkFrame(parent, fg_color=_PANEL_BG, corner_radius=12, width=240)
        cframe.pack(side="right", fill="y", padx=(8, 16), pady=16)
        cframe.pack_propagate(False)

        pad = {"padx": 18, "pady": 8}

        # section title
        ctk.CTkLabel(cframe, text="Controls", font=("Helvetica", 15, "bold"),
                     text_color=_TEXT).pack(pady=(20, 4))
        _divider(cframe)

        # ── Start button ──────────────────────────────────────────────────────
        self._btn_start = ctk.CTkButton(
            cframe,
            text="▶  Start Camera",
            font=_FONT_BODY,
            fg_color=_GREEN,
            hover_color="#16a34a",
            text_color="#fff",
            corner_radius=8,
            height=42,
            command=self._start_camera,
        )
        self._btn_start.pack(fill="x", **pad)

        # ── Stop button ───────────────────────────────────────────────────────
        self._btn_stop = ctk.CTkButton(
            cframe,
            text="■  Stop Camera",
            font=_FONT_BODY,
            fg_color=_RED,
            hover_color="#b91c1c",
            text_color="#fff",
            corner_radius=8,
            height=42,
            state="disabled",
            command=self._stop_camera,
        )
        self._btn_stop.pack(fill="x", **pad)

        _divider(cframe)

        # ── Confidence slider ─────────────────────────────────────────────────
        ctk.CTkLabel(cframe, text="Confidence Threshold",
                     font=_FONT_SMALL, text_color=_MUTED).pack(anchor="w", padx=18, pady=(8, 0))

        self._conf_val_label = ctk.CTkLabel(
            cframe, text="0.50", font=("Courier New", 16, "bold"), text_color=_ACCENT
        )
        self._conf_val_label.pack()

        self._slider = ctk.CTkSlider(
            cframe,
            from_=0.10,
            to=1.00,
            number_of_steps=90,
            command=self._on_slider,
            button_color=_ACCENT,
            progress_color=_ACCENT2,
        )
        self._slider.set(0.50)
        self._slider.pack(fill="x", padx=18, pady=(0, 12))

        _divider(cframe)

        # ── Info block ────────────────────────────────────────────────────────
        info_text = (
            "Model :  YOLO11n\n"
            "Input  :  Greyscale\n"
            "Log    :  detections_log.csv"
        )
        ctk.CTkLabel(
            cframe,
            text=info_text,
            font=_FONT_SMALL,
            text_color=_MUTED,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(8, 0))

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=_PANEL_BG, corner_radius=0, height=36)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_dot  = ctk.CTkLabel(bar, text="●", font=("Helvetica", 14),
                                          text_color=_MUTED)
        self._status_dot.pack(side="left", padx=(16, 4), pady=8)

        self._status_label = ctk.CTkLabel(bar, text="Camera Stopped",
                                           font=_FONT_SMALL, text_color=_MUTED)
        self._status_label.pack(side="left", pady=8)

        ctk.CTkLabel(bar, text="Fatima · RIMS Internship 2026",
                      font=_FONT_SMALL, text_color="#2d3748").pack(side="right", padx=16)

    # ──────────────────────────── button handlers ────────────────────────────

    def _start_camera(self):
        if self._worker and self._worker.is_alive():
            return  # already running

        # clear the queue before creating a new worker
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._worker = DetectionWorker(
            frame_queue     = self._frame_queue,
            status_callback = self._safe_set_status,
            fps_callback    = self._safe_set_fps,
        )
        self._worker.set_conf(self._slider.get())
        self._worker.start()

        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")

        # begin polling the queue
        self._poll_frame_queue()

    def _stop_camera(self):
        if self._worker:
            self._worker.stop()
            self._worker = None

        # cancel the polling loop
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._draw_placeholder()

    def _on_close(self):
        """Ensure the worker thread is stopped before the window closes."""
        self._stop_camera()
        self.destroy()

    # ──────────────────────────── GUI polling loop ───────────────────────────

    def _poll_frame_queue(self):
        """
        Called repeatedly via `after()` — runs 100% on the main thread.
        Retrieves the latest frame the worker pushed, converts it to a
        Tkinter-compatible PhotoImage, and updates the canvas label.
        This is the ONLY safe way to update Tkinter widgets from data
        produced by another thread.
        """
        try:
            frame_bgr = self._frame_queue.get_nowait()
            self._update_canvas(frame_bgr)
        except queue.Empty:
            pass  # no new frame yet — skip this tick, do NOT block

        # re-schedule next poll (keeps the event loop free)
        self._after_id = self.after(self.POLL_MS, self._poll_frame_queue)

    def _update_canvas(self, frame_bgr: np.ndarray):
        """Convert a BGR numpy array to CTkImage and push it to the label widget."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb).resize(
            (self.VIDEO_W, self.VIDEO_H), Image.LANCZOS
        )
        ctk_img   = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                  size=(self.VIDEO_W, self.VIDEO_H))
        # keep a reference — critical! Without this Python GC would delete
        # the image object and Tkinter would display a blank / broken image.
        self._canvas.configure(image=ctk_img)
        self._canvas._image = ctk_img   # explicit ref to prevent GC

    # ──────────────────────────── slider & labels ────────────────────────────

    def _on_slider(self, value: float):
        self._conf_val_label.configure(text=f"{value:.2f}")
        if self._worker:
            self._worker.set_conf(value)  # thread-safe setter

    def _set_status(self, text: str, color: str = _TEXT):
        self._status_label.configure(text=text, text_color=color)
        dot_color = _GREEN if "Running" in text else (_RED if "error" in text.lower() else _MUTED)
        self._status_dot.configure(text_color=dot_color)

    def _set_fps(self, fps: float):
        if fps > 0:
            self._fps_label.configure(text=f"FPS: {fps:.1f}", text_color=_GREEN)
            self._set_status(f"Running — {fps:.1f} FPS", _GREEN)
        else:
            self._fps_label.configure(text="FPS: —", text_color=_MUTED)

    # thread-safe wrappers: worker thread calls these; they schedule the
    # actual widget update on the main thread using after()
    def _safe_set_status(self, text: str):
        self.after(0, lambda: self._set_status(text))

    def _safe_set_fps(self, fps: float):
        self.after(0, lambda: self._set_fps(fps))


# ── utility ───────────────────────────────────────────────────────────────────

def _divider(parent):
    ctk.CTkFrame(parent, fg_color="#2d3748", height=1).pack(
        fill="x", padx=14, pady=6
    )
