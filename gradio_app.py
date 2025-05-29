# This is a gradio app that displays the user's live webcam feed

import gradio as gr
import cv2
import numpy as np
import os

# Load Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def process_frame(frame):
    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Simple hand detection using skin color segmentation (very basic)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # Filter small areas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

with gr.Blocks(title="Live Webcam Feed with Hand and Face Tracking") as demo:
    gr.Markdown("# Live Webcam Feed with Hand and Face Tracking\nDisplays the user's live webcam feed with hand and face tracking overlays.")
    webcam = gr.Image(source="webcam", streaming=True, label="Webcam Feed")
    output = gr.Image(label="Processed Feed")
    
    def gr_process(image):
        if image is None:
            return None
        # Convert from RGB to BGR for OpenCV
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed = process_frame(frame)
        # Convert back to RGB for display
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    webcam.change(fn=gr_process, inputs=webcam, outputs=output)

demo.launch()

