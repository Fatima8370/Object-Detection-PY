"""
lrf_manager.py
--------------
Hardware interface for D1200 6-Pin Laser Ranging Sensor over Serial (RS232/TTL).
Pin Config: 1: GND, 2: VCC(5V), 4: TXD->RXD, 5: RXD->TXD, 6: EN(GND)
Baud Rate: 115200 8N1

Serial Commands (ASCII):
  - Single Measurement:   <MAonce>
  - Continuous Ranging:   <MAcont>
  - Stop Ranging:         <MAStop>

Return Data Format: ASCII stream containing numbers followed by 'm', e.g. "2.46m\0" or "0.73m\r\n".
Buffer Parsing: Uses non-blocking regex matching to parse incoming chunks without blocking UI.
"""

import serial
import serial.tools.list_ports
import threading
import time
import re
from typing import Callable, Optional


class LRFManager:
    def __init__(
        self,
        port: str = "COM5",
        baudrate: int = 115200,
        distance_callback: Optional[Callable[[float, str], None]] = None,
        status_callback: Optional[Callable[[str, bool], None]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.distance_callback = distance_callback
        self.status_callback = status_callback

        self.ser: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.mode = "stop"  # "stop", "once", "cont"
        self.simulation_mode = False
        
        self.last_distance: Optional[float] = None
        self.last_distance_str: str = "--- m"

    @staticmethod
    def get_available_ports() -> list[str]:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if "COM5" not in ports:
            ports.append("COM5")
        return sorted(list(set(ports)))

    def set_port(self, port: str):
        if self._running:
            self.stop_reading()
        self.port = port

    def connect(self) -> bool:
        if self.simulation_mode:
            if self.status_callback:
                self.status_callback("LRF Simulation Active", False)
            return True

        with self._lock:
            if self.ser and self.ser.is_open:
                return True
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                if self.status_callback:
                    self.status_callback(f"LRF Connected ({self.port})", False)
                return True
            except Exception as exc:
                self.ser = None
                if self.status_callback:
                    self.status_callback(f"LRF Port Error ({self.port}): {exc}", True)
                return False

    def disconnect(self):
        self.stop_reading()
        with self._lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None
        if self.status_callback:
            self.status_callback("LRF Disconnected", False)

    def send_command(self, cmd: str) -> bool:
        if self.simulation_mode:
            return True

        if not self.ser or not self.ser.is_open:
            if not self.connect():
                return False
        try:
            with self._lock:
                self.ser.write(cmd.encode("ascii"))
            return True
        except Exception as exc:
            if self.status_callback:
                self.status_callback(f"LRF Write Error: {exc}", True)
            return False

    def read_once(self) -> bool:
        """Triggers a single laser distance measurement (<MAonce>)."""
        self.stop_reading()
        if not self.connect():
            return False
        
        self.mode = "once"
        self._running = True
        self.send_command("<MAonce>")
        
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def start_continuous(self) -> bool:
        """Starts continuous laser distance measurement loop (<MAcont>)."""
        self.stop_reading()
        if not self.connect():
            return False

        self.mode = "cont"
        self._running = True
        self.send_command("<MAcont>")

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def stop_reading(self):
        """Halts the reading thread and sends <MAStop>."""
        self._running = False
        if not self.simulation_mode and self.ser and self.ser.is_open:
            try:
                with self._lock:
                    self.ser.write(b"<MAStop>")
            except Exception:
                pass

        self.mode = "stop"
        if self._thread and self._thread.is_alive() and threading.current_thread() != self._thread:
            self._thread.join(timeout=0.5)
        self._thread = None

    def _reader_loop(self):
        buffer = ""
        pattern = re.compile(r'(\d+\.\d+)m')
        start_time = time.time()
        sim_val = 2.45

        while self._running:
            if self.simulation_mode:
                time.sleep(0.1)
                sim_val = round(2.0 + (time.time() % 3.5), 2)
                self.last_distance = sim_val
                dist_str = f"{sim_val:.2f} m"
                self.last_distance_str = dist_str
                if self.distance_callback:
                    self.distance_callback(sim_val, dist_str)

                if self.mode == "once":
                    break
                continue

            try:
                if not self.ser or not self.ser.is_open:
                    break

                with self._lock:
                    n_bytes = self.ser.in_waiting if self.ser else 0
                    chunk = self.ser.read(n_bytes).decode("ascii", errors="ignore") if n_bytes > 0 else ""

                if chunk:
                    buffer += chunk
                    if len(buffer) > 2048:
                        buffer = buffer[-1024:]

                    matches = pattern.findall(buffer)
                    if matches:
                        latest_val = float(matches[-1])
                        self.last_distance = latest_val
                        if latest_val == 0.0:
                            dist_str = "Out of Range"
                        else:
                            dist_str = f"{latest_val:.2f} m"
                        
                        self.last_distance_str = dist_str
                        if self.distance_callback:
                            self.distance_callback(latest_val, dist_str)

                        if self.mode == "once":
                            break
                else:
                    time.sleep(0.01)

                if self.mode == "once" and (time.time() - start_time > 3.0):
                    if self.status_callback:
                        self.status_callback("LRF Timeout: No data received", True)
                    break

            except Exception as exc:
                if self.status_callback:
                    self.status_callback(f"LRF Ingestion Error: {exc}", True)
                break

        if self.mode == "once":
            self.send_command("<MAStop>")
            self._running = False
            self.mode = "stop"
