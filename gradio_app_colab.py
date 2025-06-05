# gradio_app_colab.py
# Gradio app for face detection in Colab (image upload instead of webcam)

import gradio as gr
import cv2
import numpy as np
import imageio
import tempfile

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

num_faces_to_capture = 10

# Detect faces in a single image and return cropped face regions
def detect_faces_in_image(image):
    if image is None:
        return []
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # Convert to BGR if RGBA
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face_regions = []
    for (x, y, w, h) in faces[:num_faces_to_capture]:
        face_region = image[y:y+h, x:x+w]
        face_regions.append(face_region)
    return face_regions

def faces_to_gif(faces, size=(128, 128)):
    if not faces:
        return None
    resized_faces = [cv2.resize(f, size) for f in faces]
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        imageio.mimsave(tmpfile.name, resized_faces, format='GIF', duration=0.3, loop=0)
        return tmpfile.name

def process_faces(faces):
    # Placeholder for face processing logic
    return faces

def gradio_interface(images):
    # Accepts a single image or a list of images
    if not isinstance(images, list):
        images = [images]
    all_faces = []
    for img in images:
        faces = detect_faces_in_image(img)
        all_faces.extend(faces)
        if len(all_faces) >= num_faces_to_capture:
            break
    all_faces = all_faces[:num_faces_to_capture]
    faces_gif_path = faces_to_gif(all_faces)
    processed = process_faces(all_faces)
    processed_gif_path = faces_to_gif(processed)
    faces_count_value = f"# **Faces captured:** {len(all_faces)}/{num_faces_to_capture}"
    return [all_faces, faces_gif_path, faces_count_value, processed, processed_gif_path]

with gr.Blocks(title="Face Detection from Uploaded Images (Colab)") as demo:
    gr.Markdown("# Face Detection using OpenCV in Colab.\nUpload images to detect and extract faces.\n\n**Note:** Webcam input is not available in Colab.")
    image_input = gr.Image(type="numpy", label="Upload Image(s)", tool=None, shape=None, sources=["upload"], image_mode="RGB", multiple=True)
    faces_gallery = gr.Gallery(label="Captured Faces", columns=num_faces_to_capture, scale=1, height=300)
    faces_gif = gr.Image(label="Faces GIF", width=200, height=300)
    faces_count = gr.Markdown(f"# **Faces captured:** 0/{num_faces_to_capture}")
    processed_faces_gallery = gr.Gallery(label="Processed Faces", columns=num_faces_to_capture, scale=1, height=300)
    processed_gif = gr.Image(label="Processed GIF", width=200, height=300)

    image_input.change(
        fn=gradio_interface,
        inputs=image_input,
        outputs=[faces_gallery, faces_gif, faces_count, processed_faces_gallery, processed_gif]
    )

demo.launch()
