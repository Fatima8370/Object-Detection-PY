"""
app.py
------
Unified Application Entry Point for YOLO Object Detection & Laser Range Finder (LRF) System.

Features:
  - Multi-threaded YOLOv8 Object Detection (webcam, image, video).
  - Dynamic Model Switcher (yolov8n.pt vs best.pt).
  - Hardware Serial Interface for D1200 Laser Range Finder (COM5, 115200 8N1).
  - Non-blocking ASCII buffer parsing regex for real-time distance measurements.
  - Boresight reticle crosshair & target alignment distance overlay.
  - Graceful connection handling & simulation fallback.

Usage:
  python app.py
"""

import sys
from app_gui import ObjectDetectionApp


def main():
    app = ObjectDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
