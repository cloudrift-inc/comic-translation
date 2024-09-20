import gradio as gr
from PIL import Image
import numpy as np

from comic.translate import ComicTranslator

translator = ComicTranslator()


def process_image(img, source_langauge, target_language):
    img = Image.open(img)
    return translator.translate(np.asarray(img), source_langauge, target_language)


demo = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="filepath"),
            gr.Dropdown(choices=[lang[0] for lang in translator.available_languages], label="Select source language", value="English"),
            gr.Dropdown(choices=[lang[0] for lang in translator.available_languages], label="Select target language", value="Russian")],
    outputs=[gr.Image(type="pil"), gr.Textbox(max_lines=100)],
    title="Translate Any Comics and Manga",
)


if __name__ == "__main__":
    demo.launch()
