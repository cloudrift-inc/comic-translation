import numpy as np

from .text_translator import TextTranslator
from .text_detector import OCRProcessor


class ComicTranslator:
    def __init__(self):
        self.text_translator = TextTranslator()
        self.text_detector = OCRProcessor()

    @property
    def available_languages(self):
        return self.text_translator.available_languages

    def translate(self, img, source_language, target_language):
        """
        Find text on the image, translate it to the target language, and return image with translated text.
        """
        result = self.text_detector.perform_ocr(img)
        text = ""
        if result:
            for line in result:
                text += line[1][0] + "\n"
        processed_img = self.text_detector.draw_ocr_results(img, result)

        print(text)
        translated_text = self.text_translator.translate_text(text, source_language=source_language, target_language=target_language)

        return processed_img, translated_text
