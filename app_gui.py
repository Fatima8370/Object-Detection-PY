"""
app_gui.py  (v3)
----------------
Scalable window, fixed-ratio video canvas.
  • Start Camera / Upload Image / Upload Video input modes.
  • Confidence slider dynamically re-renders images & videos from cached
    YOLO results — no re-inference needed.
  • Camera mode uses imgsz=640 (fast); image/video use imgsz=1280 (quality).
  • Greyscale / RGB toggle updates the worker in real-time.
"""

import queue
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2

from detection_worker import DetectionWorker, MODE_CAMERA, MODE_IMAGE, MODE_VIDEO


# ── visual theme ─────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_BG        = "#0f0f13"
_PANEL_BG  = "#16161e"
_ACCENT    = "#00c8ff"
_ACCENT2   = "#7c3aed"
_TEXT      = "#e2e8f0"
_MUTED     = "#64748b"
_GREEN     = "#22c55e"
_RED       = "#ef4444"
_ORANGE    = "#f97316"
_PURPLE    = "#a855f7"

_F_TITLE   = ("Helvetica", 22, "bold")
_F_BODY    = ("Helvetica", 13)
_F_MONO    = ("Courier New", 12)
_F_SMALL   = ("Helvetica", 11)

VIDEO_W, VIDEO_H = 800, 560     # fixed canvas dimensions


class ObjectDetectionApp(ctk.CTk):

    POLL_MS = 30    # ms between GUI queue polls (~33 fps display max)

    def __init__(self):
        super().__init__()

        self.title("Object Detection")
        self.configure(fg_color=_BG)
        # Scalable window — minimum size keeps the layout intact
        self.resizable(True, True)
        self.minsize(1080, 680)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker: DetectionWorker  = None
        self._after_id = None
        self._greyscale_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._set_status("Idle — choose an input source", _MUTED)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────── UI layout ───────────────────────────────

    def _build_ui(self):
        # title bar
        bar = ctk.CTkFrame(self, fg_color=_PANEL_BG, corner_radius=0, height=56)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="⬡  Object Detection Studio",
                     font=_F_TITLE, text_color=_ACCENT).pack(side="left", padx=24)

        self._fps_label = ctk.CTkLabel(bar, text="FPS: —",
                                        font=_F_MONO, text_color=_MUTED)
        self._fps_label.pack(side="right", padx=24)

        # body (video | controls)
        body = ctk.CTkFrame(self, fg_color=_BG)
        body.pack(fill="both", expand=True)

        self._build_video_panel(body)
        self._build_control_panel(body)

        # status bar
        self._build_status_bar()

    def _build_video_panel(self, parent):
        vframe = ctk.CTkFrame(parent, fg_color=_PANEL_BG, corner_radius=12)
        vframe.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        # Fixed-size canvas inside the expanding frame
        self._canvas = ctk.CTkLabel(
            vframe, text="", width=VIDEO_W, height=VIDEO_H,
            fg_color="#0a0a12", corner_radius=8,
        )
        self._canvas.pack(padx=12, pady=12, anchor="center")
        self._draw_placeholder()

    def _draw_placeholder(self):
        ph = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        for y in range(VIDEO_H):
            v = int(18 + 14 * y / VIDEO_H)
            ph[y, :] = (v, v, v + 10)
        cv2.putText(ph, "No Source Active",
                    (VIDEO_W // 2 - 145, VIDEO_H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (55, 65, 80), 2, cv2.LINE_AA)
        cv2.putText(ph, "Use the controls panel to start",
                    (VIDEO_W // 2 - 175, VIDEO_H // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 50, 65), 1, cv2.LINE_AA)
        self._update_canvas(ph)

    def _build_control_panel(self, parent):
        cframe = ctk.CTkFrame(parent, fg_color=_PANEL_BG, corner_radius=12, width=260)
        cframe.pack(side="right", fill="y", padx=(8, 16), pady=16)
        cframe.pack_propagate(False)

        pad = {"padx": 18, "pady": 6}

        ctk.CTkLabel(cframe, text="Controls",
                     font=("Helvetica", 15, "bold"), text_color=_TEXT).pack(pady=(20, 4))
        _divider(cframe)

        # ── Live Camera ───────────────────────────────────────────────────────
        ctk.CTkLabel(cframe, text="LIVE SOURCE",
                     font=("Helvetica", 10, "bold"), text_color=_MUTED).pack(
            anchor="w", padx=18, pady=(10, 2))

        self._btn_cam = ctk.CTkButton(
            cframe, text="▶  Start Camera", font=_F_BODY,
            fg_color=_GREEN, hover_color="#16a34a", text_color="#fff",
            corner_radius=8, height=40, command=self._start_camera,
        )
        self._btn_cam.pack(fill="x", **pad)

        self._btn_stop = ctk.CTkButton(
            cframe, text="■  Stop", font=_F_BODY,
            fg_color=_RED, hover_color="#b91c1c", text_color="#fff",
            corner_radius=8, height=40, state="disabled",
            command=self._stop,
        )
        self._btn_stop.pack(fill="x", **pad)

        _divider(cframe)

        # ── File Upload ───────────────────────────────────────────────────────
        ctk.CTkLabel(cframe, text="FILE UPLOAD",
                     font=("Helvetica", 10, "bold"), text_color=_MUTED).pack(
            anchor="w", padx=18, pady=(10, 2))

        self._btn_img = ctk.CTkButton(
            cframe, text="🖼  Upload Image", font=_F_BODY,
            fg_color=_ORANGE, hover_color="#c2410c", text_color="#fff",
            corner_radius=8, height=40, command=self._upload_image,
        )
        self._btn_img.pack(fill="x", **pad)

        self._btn_vid = ctk.CTkButton(
            cframe, text="🎬  Upload Video", font=_F_BODY,
            fg_color=_PURPLE, hover_color="#7e22ce", text_color="#fff",
            corner_radius=8, height=40, command=self._upload_video,
        )
        self._btn_vid.pack(fill="x", **pad)

        _divider(cframe)

        # ── Confidence slider ─────────────────────────────────────────────────
        ctk.CTkLabel(cframe, text="Confidence Threshold",
                     font=_F_SMALL, text_color=_MUTED).pack(
            anchor="w", padx=18, pady=(10, 0))

        self._conf_lbl = ctk.CTkLabel(
            cframe, text="0.30", font=("Courier New", 16, "bold"),
            text_color=_ACCENT,
        )
        self._conf_lbl.pack()

        self._slider = ctk.CTkSlider(
            cframe, from_=0.10, to=1.00, number_of_steps=90,
            command=self._on_slider,
            button_color=_ACCENT, progress_color=_ACCENT2,
        )
        self._slider.set(0.30)
        self._slider.pack(fill="x", padx=18, pady=(0, 10))

        _divider(cframe)

        # ── Greyscale / RGB toggle ────────────────────────────────────────────
        ctk.CTkLabel(cframe, text="COLOUR MODE",
                     font=("Helvetica", 10, "bold"), text_color=_MUTED).pack(
            anchor="w", padx=18, pady=(10, 4))

        toggle_row = ctk.CTkFrame(cframe, fg_color="transparent")
        toggle_row.pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(toggle_row, text="RGB", font=_F_SMALL,
                     text_color=_MUTED).pack(side="left")

        self._grey_switch = ctk.CTkSwitch(
            toggle_row, text="", variable=self._greyscale_var,
            onvalue=True, offvalue=False,
            command=self._on_grey_toggle,
            button_color=_ACCENT, progress_color=_ACCENT2,
            width=52,
        )
        self._grey_switch.pack(side="left", padx=8)

        ctk.CTkLabel(toggle_row, text="Grey", font=_F_SMALL,
                     text_color=_MUTED).pack(side="left")

        _divider(cframe)

        # ── Model info ────────────────────────────────────────────────────────
        ctk.CTkLabel(
            cframe,
            text="Model    :  YOLO26s\nCamera  :  640 px (fast)\nImg/Vid  :  1280 px\nClasses  :  80 (COCO)",
            font=_F_SMALL, text_color=_MUTED, justify="left",
        ).pack(anchor="w", padx=18, pady=(10, 0))

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=_PANEL_BG, corner_radius=0, height=36)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_dot = ctk.CTkLabel(bar, text="●",
                                         font=("Helvetica", 14), text_color=_MUTED)
        self._status_dot.pack(side="left", padx=(16, 4))

        self._status_lbl = ctk.CTkLabel(bar, text="Idle",
                                         font=_F_SMALL, text_color=_MUTED)
        self._status_lbl.pack(side="left")

        ctk.CTkLabel(bar, text="Fatima · RIMS Internship 2026",
                      font=_F_SMALL, text_color="#2d3748").pack(side="right", padx=16)

    # ──────────────────────────── source controls ────────────────────────────

    def _start_camera(self):
        self._launch_worker(MODE_CAMERA, source=None)

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if path:
            self._launch_worker(MODE_IMAGE, source=path)

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
                       ("All files", "*.*")],
        )
        if path:
            self._launch_worker(MODE_VIDEO, source=path)

    def _launch_worker(self, mode: str, source):
        # Stop any running worker first
        self._stop(keep_ui_state=True)

        # Drain the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        self._worker = DetectionWorker(
            frame_queue     = self._frame_queue,
            status_callback = self._safe_status,
            fps_callback    = self._safe_fps,
            mode            = mode,
            source          = source,
        )
        # Set conf/grey BEFORE start() so the worker reads correct values
        # from the very first frame; _rerender=False at this point (initial run)
        self._worker.set_conf(self._slider.get())
        self._worker.set_greyscale(self._greyscale_var.get())
        # Clear the rerender flag that set_conf/set_greyscale above may have set —
        # the worker will do a fresh render on the first frame anyway
        self._worker._rerender = False
        self._worker.start()

        self._btn_cam.configure(state="disabled")
        self._btn_img.configure(state="disabled")
        self._btn_vid.configure(state="disabled")
        self._btn_stop.configure(state="normal")

        self._poll_frame_queue()

    def _stop(self, keep_ui_state: bool = False):
        if self._worker:
            self._worker.stop()
            self._worker = None

        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

        if not keep_ui_state:
            self._btn_cam.configure(state="normal")
            self._btn_img.configure(state="normal")
            self._btn_vid.configure(state="normal")
            self._btn_stop.configure(state="disabled")
            self._draw_placeholder()
            self._set_status("Idle — choose an input source", _MUTED)
            self._fps_label.configure(text="FPS: —", text_color=_MUTED)

    def _on_close(self):
        self._stop()
        self.destroy()

    # ──────────────────────────── GUI polling loop ───────────────────────────

    def _poll_frame_queue(self):
        """
        Runs exclusively on the main thread via after().
        Pulls the latest frame from the queue and updates the canvas label.
        Never blocks — if the queue is empty we simply skip this tick.
        """
        try:
            frame_bgr = self._frame_queue.get_nowait()
            self._update_canvas(frame_bgr)
        except queue.Empty:
            pass

        self._after_id = self.after(self.POLL_MS, self._poll_frame_queue)

    def _update_canvas(self, frame_bgr: np.ndarray):
        """
        Convert BGR numpy array → CTkImage → push to label widget.
        The explicit `self._canvas._image = ctk_img` reference prevents
        Python's GC from collecting the image object between frames,
        which would result in a blank/corrupted display.
        """
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                                size=(VIDEO_W, VIDEO_H))
        self._canvas.configure(image=ctk_img)
        self._canvas._image = ctk_img  # keep strong reference → no GC blank

    # ──────────────────────────── callbacks ──────────────────────────────────

    def _on_slider(self, value: float):
        self._conf_lbl.configure(text=f"{value:.2f}")
        if self._worker:
            self._worker.set_conf(value)

    def _on_grey_toggle(self):
        if self._worker:
            self._worker.set_greyscale(self._greyscale_var.get())

    def _set_status(self, text: str, color: str = _TEXT):
        self._status_lbl.configure(text=text, text_color=color)
        dot = _GREEN if "Running" in text else (_RED if "error" in text.lower() else _MUTED)
        self._status_dot.configure(text_color=dot)

    def _set_fps(self, fps: float):
        if fps > 0:
            self._fps_label.configure(text=f"FPS: {fps:.1f}", text_color=_GREEN)
            # update the status text's FPS component without overwriting mode
            cur = self._status_lbl.cget("text")
            if "Running" in cur:
                prefix = cur.split("—")[0].rstrip() if "—" in cur else cur
                self._set_status(f"{prefix} — {fps:.1f} FPS", _GREEN)
        else:
            self._fps_label.configure(text="FPS: —", text_color=_MUTED)

    # thread-safe wrappers (worker calls these from its thread)
    def _safe_status(self, text: str):
        self.after(0, lambda: self._set_status(text))

    def _safe_fps(self, fps: float):
        self.after(0, lambda: self._set_fps(fps))


# ── helpers ───────────────────────────────────────────────────────────────────

def _divider(parent):
    ctk.CTkFrame(parent, fg_color="#2d3748", height=1).pack(
        fill="x", padx=14, pady=6
    )
