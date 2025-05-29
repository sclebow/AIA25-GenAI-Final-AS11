# This is a gradio app that displays the user's live webcam feed

import gradio as gr
import cv2
import numpy as np

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to process video frames for face detection
def detect_faces(frame):
    # Ensure frame is writable
    process_frame = frame.copy()
    original_frame = frame.copy()

    # Crop and resize the frame to speed up processing
    max_size = 320

    # Resize frame so that the larger dimension is max_size and maintain aspect ratio
    h, w = process_frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)
        process_frame = cv2.resize(process_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert frame to numpy array if needed
    if not isinstance(process_frame, np.ndarray):
        process_frame = np.array(process_frame)
    # Convert to BGR if RGBA
    if process_frame.shape[2] == 4:
        process_frame = cv2.cvtColor(process_frame, cv2.COLOR_RGBA2BGR)
    elif process_frame.shape[2] == 1:
        process_frame = cv2.cvtColor(process_frame, cv2.COLOR_GRAY2BGR)
    # Face detection
    gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(process_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(process_frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Convert back to RGB for Gradio
    process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    return process_frame

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
    
    webcam.stream(fn=detect_faces, inputs=webcam, outputs=output)

demo.launch()

