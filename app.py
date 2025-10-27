import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def load_model():
    return YOLO("yolov8n.pt")

def detect_objects(model, image):
    results = model(image)
    return results

def draw_boxes(image, results, model):
    image = np.array(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    st.title("YOLOv8X Object Detection")
    st.write("Upload an image to detect objects using YOLOv8X.")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Detecting objects...")
        results = detect_objects(model, image)
        
        output_image = draw_boxes(image, results, model)
        st.image(output_image, caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()