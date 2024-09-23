import numpy as np
from PIL import Image

from comic import DATA_DIR
from comic.text_translator import TextTranslator
from comic.text_detection import Detector
from comic.text_detector import OCRProcessor
from comic.inpainting import Inpainter


class ComicTranslator:
    def __init__(self):
        self.text_translator = TextTranslator()
        self.text_detector = OCRProcessor()
        self.detector = Detector()
        self.inpainter = Inpainter()

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

        boxes = [line[0] for line in result]
        bboxes = [(min([x for x, y in b]),
                   min([y for x, y in b]),
                   max([x for x, y in b]),
                   max([y for x, y in b])) for b in boxes]
        img = Image.fromarray(img)
        boxes = self.detector.detect(img)
        clean_image = self.inpainter.process_image(img, boxes)
        #processed_img = self.text_detector.draw_ocr_results(img, result)
        print("Input text:\n", text)


        translated_text = self.text_translator.translate_text(text, source_language=source_language, target_language=target_language)
        print("Translated text:\n", translated_text)

        return clean_image, translated_text


if __name__ == '__main__':
    translator = ComicTranslator()
    img = Image.open("/home/red-haired/Projects/comic-translation/data/aligned_jap/shokugeki_no_soma/31/2.png")#DATA_DIR / 'the_werewolf_stalks.jpg')
    processed_img, translated_text = translator.translate(np.asarray(img), "Japanese", "Russian")
    processed_img.save('result.jpg')
