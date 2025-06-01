# This is a gradio app that displays the user's live webcam feed

import gradio as gr
import cv2
import numpy as np

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global list to store face regions
face_regions = []
# Global number of faces to capture
num_faces_to_capture = 10

# Define a function to process video frames for face detection
def detect_faces(frame):
    global face_regions, num_faces_to_capture
    # Only capture more faces if we haven't reached the limit
    if len(face_regions) >= num_faces_to_capture:
        return None
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
    for i, (x, y, w, h) in enumerate(faces):
        if len(face_regions) >= num_faces_to_capture:
            break
        scale_x = original_frame.shape[1] / float(process_frame.shape[1])
        scale_y = original_frame.shape[0] / float(process_frame.shape[0])
        face_region = original_frame[int(y * scale_y):int((y + h) * scale_y), 
                                     int(x * scale_x):int((x + w) * scale_x)]
        face_regions.append(face_region)
        print(f"Captured face {len(face_regions)}: {face_region.shape}")
        print(f"len(face_regions): {len(face_regions)}")
    # If we've captured enough faces, return None to stop streaming
    if len(face_regions) >= num_faces_to_capture:
        return None
    # If no face was found, return the original frame (or None)
    if len(face_regions) == 0:
        return frame  # or None if you prefer a blank output
    # Otherwise, return the last detected face
    return face_regions[-1]

# Custom function to display captured faces after streaming stops
with gr.Blocks(title="Live Webcam Feed with Hand and Face Tracking") as demo:
    gr.Markdown("# Live Webcam Feed\nHand and Face Tracking using OpenCV.")
    with gr.Row() as webcam_row:
        webcam = gr.Image(
            sources=["webcam"], 
            streaming=True, 
            label="Webcam Input", 
            width=300,
            height=300,
            webcam_constraints={"width": 240, "height": 240, "fps": 15}
        )
        output = gr.Image(label="Processed Output")
    faces_gallery = gr.Gallery(label="Captured Faces", visible=False, columns=num_faces_to_capture, height="auto")

    def stream_callback(frame):
        result = detect_faces(frame)
        if result is None:
            return [None, gr.update(visible=True, value=face_regions)]

        return [result, gr.update(visible=False)]

    webcam.stream(fn=stream_callback, inputs=webcam, outputs=[output, faces_gallery])

demo.launch(share=True)

