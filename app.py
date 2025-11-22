import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image

st.title("Car Plate Detection App 🚗")

# โหลดโมเดล
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])

if uploaded_file:
    # บันทึกภาพชั่วคราว
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())

    # พยากรณ์
    results = model.predict(temp.name)

    # แสดงผล
    st.image(results[0].plot(), caption="Detected Plate", use_column_width=True)
