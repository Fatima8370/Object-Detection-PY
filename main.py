"""
main.py
-------
Entry point for the Real-Time Object Detection application.

Run:
    python main.py

Dependencies (install once):
    pip install ultralytics opencv-python customtkinter Pillow numpy
"""

from app_gui import ObjectDetectionApp


def main():
    app = ObjectDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
