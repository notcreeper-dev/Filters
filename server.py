import time
import base64

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request

from geometry import ClosingGestureDetector, portal_width, render_portal
from hand_tracking import INDEX_TIP, THUMB_TIP
from filters import FILTROS

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")

    allowed_origins = {
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    }

    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

    return response

# One MediaPipe tracker and gesture detector for this backend.
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

closing_detector = ClosingGestureDetector()
filter_index = 0


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Filters Python backend"})


@app.post("/api/process-frame")
def process_frame():
    request_started = time.perf_counter()
    global filter_index

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not isinstance(image_data, str) or "," not in image_data:
        return jsonify({"error": "Expected a base64 data URL in 'image'."}), 400

    decode_started = time.perf_counter()

    try:
        encoded = image_data.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        frame_array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid image data."}), 400

    if frame is None:
        return jsonify({"error": "Could not decode image."}), 400

    decode_ms = (time.perf_counter() - decode_started) * 1000

    # Match the original application's mirrored camera view.
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mediapipe_started = time.perf_counter()
    results = hands.process(rgb)
    mediapipe_ms = (time.perf_counter() - mediapipe_started) * 1000

    portal_started = time.perf_counter()

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            raw_label = handedness.classification[0].label
            label = "Right" if raw_label == "Left" else "Left"

            if label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

    if left_hand is not None and right_hand is not None:
        lm_left = left_hand.landmark
        lm_right = right_hand.landmark

        p1 = (lm_left[INDEX_TIP].x * width, lm_left[INDEX_TIP].y * height)
        p2 = (lm_left[THUMB_TIP].x * width, lm_left[THUMB_TIP].y * height)
        p3 = (lm_right[INDEX_TIP].x * width, lm_right[INDEX_TIP].y * height)
        p4 = (lm_right[THUMB_TIP].x * width, lm_right[THUMB_TIP].y * height)

        distance = portal_width(p1, p2, p3, p4)

        if closing_detector.update(distance, width):
            filter_index = (filter_index + 1) % len(FILTROS)

        frame = render_portal(frame, p1, p2, p3, p4, FILTROS[filter_index])

    portal_ms = (time.perf_counter() - portal_started) * 1000
    output_started = time.perf_counter()
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return jsonify({"error": "Could not encode processed frame."}), 500

    result = base64.b64encode(buffer).decode("ascii")
    output_encode_ms = (time.perf_counter() - output_started) * 1000
    total_ms = (time.perf_counter() - request_started) * 1000
    return jsonify({
        "image": "data:image/jpeg;base64," + result,
        "filter_index": filter_index,
        "hands_detected": left_hand is not None and right_hand is not None,
        "timings_ms": {
            "decode": round(decode_ms, 2),
            "mediapipe": round(mediapipe_ms, 2),
            "portal": round(portal_ms, 2),
            "output_encode": round(output_encode_ms, 2),
            "total": round(total_ms, 2),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
