# This is a gradio app that displays the user's live webcam feed

import gradio as gr

with gr.Blocks(title="Live Webcam Feed") as demo:
    gr.Markdown("# Live Webcam Feed\nDisplays the user's live webcam feed.")
    webcam = gr.Video(label="Webcam Feed", streaming=True, autoplay=True, show_label=False, sources=["webcam"])

demo.launch()

