import gradio as gr
from PIL import Image


available_languages = ['English', 'French', 'German', 'Russian', 'Japanese', 'Chinese', 'Korean']


def process_image(img, source_langauge, target_language):
    # Open the image
    image = Image.open(img)
    processed_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return processed_image


demo = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="filepath"),
            gr.Dropdown(choices=available_languages, label="Select source language"),
            gr.Dropdown(choices=available_languages, label="Select target language")],
    outputs=gr.Image(type="pil"),
    title="Translate Any Comics and Manga",
)


if __name__ == "__main__":
    demo.launch()
