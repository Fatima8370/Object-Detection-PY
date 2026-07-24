"""
app_gui.py  (v5 - LRF Integration & Model Switcher)
---------------------------------------------------
Features:
  - CustomTkinter Desktop GUI with fixed-ratio Video Canvas & reticle crosshair.
  - Model Switcher: Select between yolov8n.pt and best.pt dynamically at runtime.
  - LRF Control Interface: Real-time digital readout & serial control buttons
    ([Read Once <MAonce>], [Read Continuous <MAcont>], [Stop Read <MAStop>]).
  - Robust Error Dialogs: Port connection warnings and simulation fallback mode.
  - Main thread polling via frame queue for 100% thread-safe UI updates.
"""

import sys
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2

from detection_worker import DetectionWorker, MODE_CAMERA, MODE_IMAGE, MODE_VIDEO
from lrf_manager import LRFManager


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

_F_TITLE   = ("Helvetica", 20, "bold")
_F_HEADER  = ("Helvetica", 12, "bold")
_F_BODY    = ("Helvetica", 13)
_F_MONO    = ("Courier New", 12)
_F_SMALL   = ("Helvetica", 11)

VIDEO_W, VIDEO_H = 800, 560


def get_available_cameras(max_tested=4):
    available = []
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for i in range(max_tested):
        try:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(f"Camera {i}")
                cap.release()
        except Exception:
            pass
    if not available:
        available = ["Camera 0"]
    return available


class ObjectDetectionApp(ctk.CTk):

    POLL_MS = 30  # ~33 FPS display poll

    def __init__(self):
        super().__init__()

        self.title("Object Detection & Laser Range Finder Studio")
        self.configure(fg_color=_BG)
        self.resizable(True, True)
        self.minsize(1120, 720)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker: DetectionWorker = None
        self._after_id = None
        
        self._greyscale_var = tk.BooleanVar(value=False)
        self._model_var = tk.StringVar(value="yolov8n.pt")
        self._camera_var = tk.StringVar(value="Camera 0")
        self._port_var = tk.StringVar(value="COM5")
        self._sim_var = tk.BooleanVar(value=False)

        # LRF Hardware Manager
        self._lrf = LRFManager(
            port=self._port_var.get(),
            baudrate=115200,
            distance_callback=self._safe_lrf_distance,
            status_callback=self._safe_lrf_status,
        )

        self._build_ui()
        self._set_status("Idle — Select input source or LRF mode", _MUTED)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────── UI Layout ───────────────────────────────

    def _build_ui(self):
        # Title Bar
        bar = ctk.CTkFrame(self, fg_color=_PANEL_BG, corner_radius=0, height=56)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text="⬡  YOLO + LRF Ranging Studio",
            font=_F_TITLE, text_color=_ACCENT
        ).pack(side="left", padx=20)

        self._fps_label = ctk.CTkLabel(bar, text="FPS: —", font=_F_MONO, text_color=_MUTED)
        self._fps_label.pack(side="right", padx=20)

        # Main Body
        body = ctk.CTkFrame(self, fg_color=_BG)
        body.pack(fill="both", expand=True)

        self._build_video_panel(body)
        self._build_control_panel(body)

        # Status Bar
        self._build_status_bar()

    def _build_video_panel(self, parent):
        vframe = ctk.CTkFrame(parent, fg_color=_PANEL_BG, corner_radius=12)
        vframe.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        self._canvas = ctk.CTkLabel(
            vframe, text="", width=VIDEO_W, height=VIDEO_H,
            fg_color="#0a0a12", corner_radius=8
        )
        self._canvas.pack(padx=12, pady=12, anchor="center")
        self._draw_placeholder()

    def _draw_placeholder(self):
        ph = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        for y in range(VIDEO_H):
            v = int(18 + 14 * y / VIDEO_H)
            ph[y, :] = (v, v, v + 10)

        cx, cy = VIDEO_W // 2, VIDEO_H // 2
        cv2.circle(ph, (cx, cy), 18, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(ph, (cx - 25, cy), (cx + 25, cy), (0, 200, 255), 1)
        cv2.line(ph, (cx, cy - 25), (cx, cy + 25), (0, 200, 255), 1)

        cv2.putText(ph, "Target Boresight Standby",
                    (cx - 150, cy - 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(ph, "No Video Source Active",
                    (cx - 140, cy + 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 95, 115), 1, cv2.LINE_AA)
        self._update_canvas(ph)

    def _build_control_panel(self, parent):
        cframe = ctk.CTkFrame(parent, fg_color=_PANEL_BG, corner_radius=12, width=310)
        cframe.pack(side="right", fill="y", padx=(8, 16), pady=16)
        cframe.pack_propagate(False)

        pad = {"padx": 14, "pady": 4}

        # Scrollable container inside control panel
        scroll = ctk.CTkScrollableFrame(cframe, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text="Controls & Hardware", font=("Helvetica", 15, "bold"), text_color=_TEXT).pack(pady=(12, 4))
        _divider(scroll)

        # ── 1. Model Selection ───────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="YOLO MODEL SELECTOR", font=_F_HEADER, text_color=_MUTED).pack(anchor="w", padx=14, pady=(6, 2))
        
        self._model_combo = ctk.CTkOptionMenu(
            scroll,
            values=["yolov8n.pt", "best.pt"],
            variable=self._model_var,
            command=self._on_model_change,
            button_color=_ACCENT2,
            fg_color="#242438",
            height=34
        )
        self._model_combo.pack(fill="x", **pad)

        _divider(scroll)

        # ── 2. Laser Range Finder (D1200) Panel ──────────────────────────────
        ctk.CTkLabel(scroll, text="LASER RANGE FINDER (D1200)", font=_F_HEADER, text_color=_MUTED).pack(anchor="w", padx=14, pady=(6, 2))

        # Port & Simulation Config Row
        port_row = ctk.CTkFrame(scroll, fg_color="transparent")
        port_row.pack(fill="x", padx=14, pady=(2, 6))

        ctk.CTkLabel(port_row, text="Port:", font=_F_SMALL, text_color=_TEXT).pack(side="left")
        
        ports = LRFManager.get_available_ports()
        self._port_combo = ctk.CTkOptionMenu(
            port_row, values=ports, variable=self._port_var,
            command=self._on_port_change, width=95, height=28, fg_color="#242438"
        )
        self._port_combo.pack(side="left", padx=6)

        self._sim_switch = ctk.CTkSwitch(
            port_row, text="Sim", variable=self._sim_var, command=self._on_sim_toggle,
            width=40, font=_F_SMALL
        )
        self._sim_switch.pack(side="right")

        # Digital Readout Display Card
        readout_card = ctk.CTkFrame(scroll, fg_color="#0a0a14", corner_radius=8, border_width=1, border_color="#1e293b")
        readout_card.pack(fill="x", padx=14, pady=6)

        ctk.CTkLabel(readout_card, text="TARGET DISTANCE", font=("Helvetica", 9, "bold"), text_color=_MUTED).pack(pady=(6, 0))
        self._dist_display = ctk.CTkLabel(
            readout_card, text="--- m", font=("Courier New", 22, "bold"), text_color=_ACCENT
        )
        self._dist_display.pack(pady=(2, 6))

        # LRF Control Buttons
        self._btn_lrf_once = ctk.CTkButton(
            scroll, text="🎯  Read Once (<MAonce>)", font=_F_BODY,
            fg_color="#1e293b", hover_color="#334155", text_color="#fff",
            corner_radius=6, height=34, command=self._lrf_read_once
        )
        self._btn_lrf_once.pack(fill="x", **pad)

        self._btn_lrf_cont = ctk.CTkButton(
            scroll, text="⚡  Read Continuous (<MAcont>)", font=_F_BODY,
            fg_color="#0284c7", hover_color="#0369a1", text_color="#fff",
            corner_radius=6, height=34, command=self._lrf_start_cont
        )
        self._btn_lrf_cont.pack(fill="x", **pad)

        self._btn_lrf_stop = ctk.CTkButton(
            scroll, text="⏹  Stop Read (<MAStop>)", font=_F_BODY,
            fg_color=_RED, hover_color="#b91c1c", text_color="#fff",
            corner_radius=6, height=34, command=self._lrf_stop_read
        )
        self._btn_lrf_stop.pack(fill="x", **pad)

        _divider(scroll)

        # ── 3. Video Source Controls ──────────────────────────────────────────
        ctk.CTkLabel(scroll, text="CAMERA & MEDIA STREAM", font=_F_HEADER, text_color=_MUTED).pack(anchor="w", padx=14, pady=(6, 2))

        # Camera Selection Dropdown & Refresh
        cam_row = ctk.CTkFrame(scroll, fg_color="transparent")
        cam_row.pack(fill="x", padx=14, pady=(2, 6))

        cams = get_available_cameras()
        self._cam_combo = ctk.CTkOptionMenu(
            cam_row, values=cams, variable=self._camera_var,
            command=self._on_camera_select, height=32, fg_color="#242438"
        )
        self._cam_combo.pack(side="left", fill="x", expand=True)

        self._btn_refresh_cams = ctk.CTkButton(
            cam_row, text="🔄", width=36, height=32,
            fg_color="#1e293b", hover_color="#334155",
            command=self._refresh_cameras
        )
        self._btn_refresh_cams.pack(side="right", padx=(6, 0))

        self._btn_cam = ctk.CTkButton(
            scroll, text="▶  Start Selected Camera", font=_F_BODY,
            fg_color=_GREEN, hover_color="#16a34a", text_color="#fff",
            corner_radius=6, height=36, command=self._start_camera
        )
        self._btn_cam.pack(fill="x", **pad)

        self._btn_stop = ctk.CTkButton(
            scroll, text="■  Stop Video Stream", font=_F_BODY,
            fg_color=_RED, hover_color="#b91c1c", text_color="#fff",
            corner_radius=6, height=36, state="disabled", command=self._stop
        )
        self._btn_stop.pack(fill="x", **pad)

        self._btn_img = ctk.CTkButton(
            scroll, text="🖼  Upload Image", font=_F_BODY,
            fg_color=_ORANGE, hover_color="#c2410c", text_color="#fff",
            corner_radius=6, height=34, command=self._upload_image
        )
        self._btn_img.pack(fill="x", **pad)

        self._btn_vid = ctk.CTkButton(
            scroll, text="🎬  Upload Video", font=_F_BODY,
            fg_color=_PURPLE, hover_color="#7e22ce", text_color="#fff",
            corner_radius=6, height=34, command=self._upload_video
        )
        self._btn_vid.pack(fill="x", **pad)

        _divider(scroll)

        # ── 4. Detection Parameters ───────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Confidence Threshold", font=_F_SMALL, text_color=_MUTED).pack(anchor="w", padx=14, pady=(6, 0))

        self._conf_lbl = ctk.CTkLabel(scroll, text="0.10", font=("Courier New", 15, "bold"), text_color=_ACCENT)
        self._conf_lbl.pack()

        self._slider = ctk.CTkSlider(
            scroll, from_=0.01, to=1.00, number_of_steps=99,
            command=self._on_slider, button_color=_ACCENT, progress_color=_ACCENT2
        )
        self._slider.set(0.10)
        self._slider.pack(fill="x", padx=14, pady=(0, 6))

        toggle_row = ctk.CTkFrame(scroll, fg_color="transparent")
        toggle_row.pack(fill="x", padx=14, pady=(0, 8))

        ctk.CTkLabel(toggle_row, text="RGB", font=_F_SMALL, text_color=_MUTED).pack(side="left")
        self._grey_switch = ctk.CTkSwitch(
            toggle_row, text="", variable=self._greyscale_var,
            onvalue=True, offvalue=False, command=self._on_grey_toggle,
            button_color=_ACCENT, progress_color=_ACCENT2, width=44
        )
        self._grey_switch.pack(side="left", padx=8)
        ctk.CTkLabel(toggle_row, text="Greyscale", font=_F_SMALL, text_color=_MUTED).pack(side="left")

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=_PANEL_BG, corner_radius=0, height=36)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_dot = ctk.CTkLabel(bar, text="●", font=("Helvetica", 14), text_color=_MUTED)
        self._status_dot.pack(side="left", padx=(16, 4))

        self._status_lbl = ctk.CTkLabel(bar, text="Idle", font=_F_SMALL, text_color=_MUTED)
        self._status_lbl.pack(side="left")

        self._lrf_status_lbl = ctk.CTkLabel(bar, text="LRF: Standby", font=_F_SMALL, text_color=_MUTED)
        self._lrf_status_lbl.pack(side="right", padx=16)

    # ──────────────────────────── Camera Selection ───────────────────────────

    def _refresh_cameras(self):
        cams = get_available_cameras()
        self._cam_combo.configure(values=cams)
        if self._camera_var.get() not in cams and cams:
            self._camera_var.set(cams[0])

    def _on_camera_select(self, new_cam: str):
        if self._worker and self._worker._mode == MODE_CAMERA:
            self._start_camera()

    # ──────────────────────────── LRF Actions ────────────────────────────

    def _on_port_change(self, new_port: str):
        self._lrf.set_port(new_port)

    def _on_sim_toggle(self):
        self._lrf.simulation_mode = self._sim_var.get()
        state = "Simulation" if self._sim_var.get() else "Hardware"
        self._lrf_status_lbl.configure(text=f"LRF: {state} Mode", text_color=_ACCENT)

    def _lrf_read_once(self):
        if not self._lrf.simulation_mode and not self._lrf.connect():
            self._show_lrf_error_dialog()
            return
        self._lrf.read_once()

    def _lrf_start_cont(self):
        if not self._lrf.simulation_mode and not self._lrf.connect():
            self._show_lrf_error_dialog()
            return
        self._lrf.start_continuous()

    def _lrf_stop_read(self):
        self._lrf.stop_reading()
        self._dist_display.configure(text="--- m", text_color=_MUTED)
        if self._worker:
            self._worker.set_lrf_distance("--- m")

    def _show_lrf_error_dialog(self):
        messagebox.showwarning(
            "LRF Connection Warning",
            f"Could not connect to Laser Range Finder on {self._port_var.get()}.\n\n"
            "Please check:\n"
            "1. LRF Sensor is plugged into USB/Serial COM port.\n"
            "2. Correct COM Port is selected.\n"
            "3. Or enable 'Sim' switch for simulation mode."
        )

    # ──────────────────────────── Source Actions ────────────────────────────

    def _on_model_change(self, new_model: str):
        if self._worker:
            self._worker.set_model_path(new_model)

    def _start_camera(self):
        cam_str = self._camera_var.get()
        try:
            cam_idx = int(cam_str.replace("Camera", "").strip())
        except ValueError:
            cam_idx = 0
        self._launch_worker(MODE_CAMERA, source=cam_idx)

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if path:
            self._launch_worker(MODE_IMAGE, source=path)

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        )
        if path:
            self._launch_worker(MODE_VIDEO, source=path)

    def _launch_worker(self, mode: str, source):
        self._stop(keep_ui_state=True)

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
            model_path      = self._model_var.get()
        )
        self._worker.set_conf(self._slider.get())
        self._worker.set_greyscale(self._greyscale_var.get())
        self._worker.set_lrf_distance(self._dist_display.cget("text"))
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
            self._set_status("Idle — Choose input source", _MUTED)
            self._fps_label.configure(text="FPS: —", text_color=_MUTED)

    def _on_close(self):
        self._lrf.disconnect()
        self._stop()
        self.destroy()

    # ──────────────────────────── Thread Polling & Canvas ────────────────────

    def _poll_frame_queue(self):
        try:
            frame_bgr = self._frame_queue.get_nowait()
            self._update_canvas(frame_bgr)
        except queue.Empty:
            pass
        self._after_id = self.after(self.POLL_MS, self._poll_frame_queue)

    def _update_canvas(self, frame_bgr: np.ndarray):
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil     = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(VIDEO_W, VIDEO_H))
        self._canvas.configure(image=ctk_img)
        self._canvas._image = ctk_img

    # ──────────────────────────── Callbacks ──────────────────────────────────

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
        else:
            self._fps_label.configure(text="FPS: —", text_color=_MUTED)

    def _safe_status(self, text: str):
        self.after(0, lambda: self._set_status(text))

    def _safe_fps(self, fps: float):
        self.after(0, lambda: self._set_fps(fps))

    def _safe_lrf_distance(self, dist_val: float, dist_str: str):
        def update():
            self._dist_display.configure(text=dist_str, text_color=_GREEN if dist_val > 0 else _ORANGE)
            if self._worker:
                self._worker.set_lrf_distance(dist_str)
        self.after(0, update)

    def _safe_lrf_status(self, msg: str, is_error: bool):
        def update():
            color = _RED if is_error else _GREEN
            self._lrf_status_lbl.configure(text=f"LRF: {msg}", text_color=color)
        self.after(0, update)


def _divider(parent):
    ctk.CTkFrame(parent, fg_color="#2d3748", height=1).pack(fill="x", padx=14, pady=6)
