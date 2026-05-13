import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SCARA Vision System", layout="wide")

# โหลดโมเดล AI (Cache ไว้เพื่อไม่ให้โหลดใหม่ทุกรอบที่ Refresh)
@st.cache_resource
def load_yolo_model():
    model_path = "best.pt" # ตรวจสอบชื่อไฟล์ให้ตรงกับใน GitHub
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_yolo_model()

class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        # แปลงเฟรมภาพจาก Browser เป็นรูปแบบที่ OpenCV ใช้งานได้
        img = frame.to_ndarray(format="bgr24")
        
        if model is not None:
            # รันการตรวจจับ OBB
            results = model(img, conf=0.5, verbose=False)
            
            # ถ้าเจอวัตถุ ให้วาดเส้นขอบ (Annotated Frame)
            if results and len(results[0]) > 0:
                img = results[0].plot()
        
        return img

# ส่วนแสดงผลบนหน้าเว็บ
st.title("🚀 SCARA Conveyor Real-time Detection")
st.subheader("YOLOv8-OBB บน Streamlit Cloud")

if model is None:
    st.error("❌ ไม่พบไฟล์โมเดล 'best.pt' ใน Repository กรุณาอัปโหลดไฟล์ก่อนครับ")
else:
    st.success("✅ โหลดโมเดลสำเร็จ พร้อมใช้งาน")
    
    # ตัวเปิดกล้อง WebRTC
    webrtc_streamer(
        key="yolo-detection",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

st.info("คำแนะนำ: หากกล้องไม่ขึ้น ให้ตรวจสอบสิทธิ์การเข้าถึงกล้องของ Browser ครับ")
