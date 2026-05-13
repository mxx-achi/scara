import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SCARA Vision (YOLOv8-OBB)", layout="wide")

@st.cache_resource
def load_yolo_model():
    # เปลี่ยนชื่อไฟล์ตรงนี้ให้ตรงกับใน GitHub ของคุณ
    model_path = "yolo26s-obb.pt" 
    
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_yolo_model()

class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if model is not None:
            # รันการตรวจจับแบบ OBB
            results = model(img, conf=0.5, verbose=False)
            
            # วาดผลลัพธ์ลงบนภาพ
            if results and len(results[0]) > 0:
                img = results[0].plot()
        
        return img

st.title("🚀 SCARA Conveyor - YOLOv8-OBB")

if model is None:
    st.error("❌ ไม่พบไฟล์ 'yolo26s-obb.pt' ใน GitHub กรุณาตรวจสอบชื่อไฟล์อีกครั้งครับ")
else:
    st.success("✅ โหลดโมเดล yolo26s-obb สำเร็จ!")
    
    webrtc_streamer(
        key="yolo-obb",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )
