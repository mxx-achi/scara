import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.set_page_config(page_title="SCARA Vision", layout="wide")

@st.cache_resource
def load_model():
    # โค้ดจะพยายามหาไฟล์ในโฟลเดอร์ปัจจุบันก่อน
    model_name = "yolo26s-obb.pt"
    if os.path.exists(model_name):
        return YOLO(model_name)
    return None

model = load_model()

class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if model:
            results = model(img, conf=0.5, verbose=False)
            if results and len(results[0]) > 0:
                img = results[0].plot()
        return img

st.title("🚀 SCARA Conveyor Detection")

if model is None:
    st.error("❌ หาไฟล์ 'yolo26s-obb.pt' ไม่เจอ! ตรวจสอบว่าอัปโหลดเข้า GitHub หรือยัง")
    # แสดงรายการไฟล์ที่มีเพื่อให้คุณตรวจสอบได้ง่ายขึ้น
    st.write("ไฟล์ที่หาเจอในเครื่องตอนนี้:", os.listdir(".")) 
else:
    st.success("✅ โหลดโมเดลสำเร็จ!")
    webrtc_streamer(
        key="yolo-obb",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
