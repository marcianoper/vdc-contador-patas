import json
import time
import threading
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class State:
    running: bool = False
    lot_active: bool = False
    lot_type: str = "blanca"
    count: int = 0
    last_count_ts: float = 0.0
    started_ts: float = 0.0
    stopped_ts: float = 0.0
    last_event: str = ""
    stream_ok: bool = False

state = State()
lock = threading.Lock()
stop_event = threading.Event()

app = Flask(__name__)

def now():
    return time.time()

def set_state(**kwargs):
    with lock:
        for k, v in kwargs.items():
            setattr(state, k, v)

def get_state():
    with lock:
        return asdict(state)

def counter_worker():
    cfg = load_config()
    cap = cv2.VideoCapture(cfg["stream_url"])

    if not cap.isOpened():
        set_state(stream_ok=False, last_event="❌ No se pudo abrir el stream")
        return

    set_state(stream_ok=True, last_event="📡 Stream conectado")

    bg = cv2.createBackgroundSubtractorMOG2()
    prev_centroid = None

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        roi = cfg["roi"]
        crop = frame[
            roi["y"]:roi["y"]+roi["h"],
            roi["x"]:roi["x"]+roi["w"]
        ]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        fg = bg.apply(gray)

        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area and area > cfg["min_area"]:
                max_area = area
                largest = c

        centroid = None
        if largest is not None:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                centroid = (
                    int(M["m10"]/M["m00"]),
                    int(M["m01"]/M["m00"])
                )

        st = get_state()
        if st["running"] and st["lot_active"] and centroid:
            if prev_centroid:
                if prev_centroid[1] < cfg["line_y"] <= centroid[1]:
                    if now() - st["last_count_ts"] > cfg["cooldown_sec"]:
                        set_state(
                            count=st["count"] + 1,
                            last_count_ts=now(),
                            last_event=f"✅ Conteo: {st['count'] + 1}"
                        )
            prev_centroid = centroid

        time.sleep(0.02)

@app.route("/")
def home():
    return Response(open("ui.html").read(), mimetype="text/html")

@app.route("/api/status")
def status():
    return jsonify(get_state())

@app.route("/api/start", methods=["POST"])
def start():
    data = request.json
    set_state(
        running=True,
        lot_active=True,
        lot_type=data.get("lot_type", "blanca"),
        started_ts=now(),
        last_event="▶️ Lote iniciado"
    )
    return jsonify(ok=True)

@app.route("/api/stop", methods=["POST"])
def stop():
    set_state(
        running=False,
        lot_active=False,
        stopped_ts=now(),
        last_event="⏹️ Lote cerrado"
    )
    return jsonify(ok=True)

if __name__ == "__main__":
    threading.Thread(target=counter_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
