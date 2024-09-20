import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

from comic import DATA_DIR


class OCRProcessor:
    def __init__(self, lang='en', use_angle_cls=True, font_path=None, font_size=20):
        """
        Initialize the OCRProcessor with the specified language and font.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        try:
            self.ocr_engine = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
        except Exception as e:
            print(f"Error initializing OCR engine: {e}")
            raise
        self.font = self._load_font(font_path, font_size)

    def _load_font(self, font_path, font_size):
        """
        Attempt to load a custom font, and fall back to default if it fails.
        Returns:
            PIL.ImageFont.FreeTypeFont: Loaded font object.
        """
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            print(f"Error loading font '{font_path}': {e}")
            print("Using default font.")
            font = ImageFont.load_default()
        return font

    def perform_ocr(self, img):
        """
        Perform OCR on the specified image.
        Returns:
            list: OCR results for the image.
        """
        result = self.ocr_engine.ocr(img, cls=True)
        if result:
            return result[0]  # Return the result for the single image
        else:
            print("No text found in image.")
            return []

    def draw_ocr_results(self, image, result):
        """
        Draw OCR results onto the image

        """
        if not result:
            print("No OCR results to draw.")
            return
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        for box, text in zip(boxes, txts):
            draw.polygon([tuple(point) for point in box], outline="red")
            draw.text((box[0][0], box[0][1]), text, font=self.font, fill="blue")
        return image

    def show_image(self, image_path):
        try:
            image = Image.open(image_path)
            image.show()
        except Exception as e:
            print(f"Error displaying image '{image_path}': {e}")


# Example usage
if __name__ == "__main__":
    ocr_processor = OCRProcessor(lang='en', font_path=DATA_DIR / 'simfang.ttf')
    img_path = '../data/the_werewolf_stalks.jpg'
    image = Image.open(img_path)
    result = ocr_processor.perform_ocr(np.array(image))
    if result:
        for line in result:
            print(line)

    image = ocr_processor.draw_ocr_results(image, result)
    image.save('result.jpg')
    ocr_processor.show_image('result.jpg')
