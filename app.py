import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CONFIDENCE_THRESH = 0.5
PICK_QUEUE_TARGET = 4

MODEL_PATH = "best.pt"

PX_PER_MM_FALLBACK = 3.2

st.set_page_config(
    page_title="SCARA Vision System",
    layout="wide"
)

st.title("SCARA Conveyor Vision System")

# ─────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class PacketInfo:
    x_px: float
    y_px: float
    x_mm: float
    y_mm: float
    angle: float
    confidence: float
    corners: np.ndarray


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"ไม่พบโมเดล: {MODEL_PATH}")
        return None

    return YOLO(MODEL_PATH)

model = load_model()


# ─────────────────────────────────────────────────────────────
# TRANSFORMER
# ─────────────────────────────────────────────────────────────

class CoordTransformer:

    def __init__(self):
        self.px_per_mm = PX_PER_MM_FALLBACK

    def px_to_mm(self, x_px, y_px):
        return (
            x_px / self.px_per_mm,
            y_px / self.px_per_mm
        )


transformer = CoordTransformer()


# ─────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────

def detect_packets(frame):

    if model is None:
        return [], frame

    results = model(frame, verbose=False)

    if not results:
        return [], frame

    result = results[0]

    packets = []

    if result.obb is not None:

        for obb in result.obb:

            conf = float(obb.conf)

            if conf < CONFIDENCE_THRESH:
                continue

            cx, cy, w, h, r = map(float, obb.xywhr[0])

            angle_deg = float(np.degrees(r))

            corners = (
                obb.xyxyxyxy[0]
                .cpu()
                .numpy()
                .reshape(4, 2)
                .astype(int)
            )

            x_mm, y_mm = transformer.px_to_mm(cx, cy)

            packets.append(
                PacketInfo(
                    x_px=cx,
                    y_px=cy,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    angle=angle_deg,
                    confidence=conf,
                    corners=corners
                )
            )

    packets.sort(key=lambda p: p.x_px)

    annotated = frame.copy()

    pick_group = packets[:PICK_QUEUE_TARGET]

    for p in packets:

        will_pick = p in pick_group

        color = (0, 255, 0) if will_pick else (150, 150, 150)

        thickness = 2 if will_pick else 1

        cv2.polylines(
            annotated,
            [p.corners],
            isClosed=True,
            color=color,
            thickness=thickness
        )

        cv2.circle(
            annotated,
            (int(p.x_px), int(p.y_px)),
            4,
            color,
            -1
        )

        label = (
            f"X:{p.x_mm:.1f}mm "
            f"Y:{p.y_mm:.1f}mm "
            f"A:{p.angle:.1f}"
        )

        cv2.putText(
            annotated,
            label,
            (int(p.x_px), int(p.y_px) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    status = (
        f"Detected: {len(packets)} | "
        f"Pick Ready: {len(pick_group) >= PICK_QUEUE_TARGET}"
    )

    cv2.putText(
        annotated,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return packets, annotated


# ─────────────────────────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        packets, annotated = detect_packets(img)

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ─────────────────────────────────────────────────────────────
# STREAMLIT
# ─────────────────────────────────────────────────────────────

st.write("เปิดกล้องเพื่อตรวจจับ Packet")

webrtc_streamer(
    key="scara-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
