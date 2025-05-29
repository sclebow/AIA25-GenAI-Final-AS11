# This is a gradio app that displays the user's live webcam feed

import gradio as gr
import cv2
import numpy as np

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to process video frames for hand and face detection
def detect_hands_and_faces(frame):
    # Ensure frame is writable
    frame = frame.copy()

    # Crop and resize the frame to speed up processing
    frame = cv2.resize(frame, (320, 240))  # Resize to a standard size for processing

    # Convert frame to numpy array if needed
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    # Convert to BGR if RGBA
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Simple hand detection using color thresholding (skin color in HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Hand', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Convert back to RGB for Gradio
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Define Gradio interface for live webcam processing
with gr.Blocks(title="Live Webcam Feed with Hand and Face Tracking") as demo:
    gr.Markdown("# Live Webcam Feed\nHand and Face Tracking using OpenCV.")
    
    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"], 
            streaming=True, 
            label="Webcam Input", 
            # interactive=False,
            width=300,
            height=300,
            webcam_constraints={"width": 240, "height": 240, "fps": 15}  # Adjusted constraints for webcam
        )
        output = gr.Image(label="Processed Output")  # Removed live=True for compatibility
    
    webcam.stream(fn=detect_hands_and_faces, inputs=webcam, outputs=output)

demo.launch()

