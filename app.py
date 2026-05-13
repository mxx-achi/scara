import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import os

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SCARA Vision System", layout="wide")

# 2. ฟังก์ชันโหลดโมเดล (ใช้ Cache เพื่อไม่ให้โหลดใหม่ทุกครั้งที่ขยับหน้าจอ)
@st.cache_resource
def load_yolo_model():
    # ตรวจสอบชื่อไฟล์ให้ตรงกับใน GitHub เป๊ะๆ
    model_path = "yolo26s-obb.pt" 
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = YOLO("yolo26s-obb.pt")

# 3. ส่วนประมวลผลวิดีโอ
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        # แปลงเฟรมภาพจากกล้องเป็น Array ที่ OpenCV ใช้งานได้
        img = frame.to_ndarray(format="bgr24")
        
        if model is not None:
            # รันการตรวจจับวัตถุ (OBB)
            results = model(img, conf=0.5, verbose=False)
            
            # ถ้าตรวจพบวัตถุ ให้วาดเส้นขอบและผลลัพธ์ลงบนภาพ
            if results and len(results[0]) > 0:
                img = results[0].plot()
        
        return img

# 4. ส่วนแสดงผลบนหน้าเว็บ Streamlit
st.title("🚀 SCARA Conveyor - Realtime Detection")
st.write("ระบบตรวจจับวัตถุด้วย YOLOv8-OBB ผ่านกล้องเว็บแคม")

if model is None:
    st.error("❌ ไม่พบไฟล์ 'yolo26s-obb.pt' ใน Repository กรุณาอัปโหลดไฟล์โมเดลก่อนครับ")
else:
    st.success("✅ โหลดโมเดล yolo26s-obb สำเร็จ พร้อมใช้งาน")
    
    # ส่วนเรียกใช้งานกล้องผ่าน WebRTC
    webrtc_streamer(
        key="yolo-obb",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

st.info("คำแนะนำ: หากกล้องไม่ขึ้น ให้ตรวจสอบการอนุญาตเข้าถึงกล้องใน Browser ของคุณ")
