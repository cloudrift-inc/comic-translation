import csv
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TextTranslator:
    TOKEN = "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"

    def __init__(self):
        self.available_languages = []
        self._load_language_list()

        self.src_lang_code = None
        self.tgt_lang_code = None

    def _load_language_list(self):
        with open('./data/FLORES-200_code.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip header
            for row in reader:
                self.available_languages.append(row)

    def _load_languages(self, source_language, target_language):
        src_lang_code = [lang[1] for lang in self.available_languages if lang[0] == source_language][0]
        if self.src_lang_code != src_lang_code:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", token=self.TOKEN, src_lang=source_language)
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=self.TOKEN)
            self.src_lang_code = src_lang_code

        self.tgt_lang_code = [lang[1] for lang in self.available_languages if lang[0] == target_language][0]

    def translate(self, img, source_language, target_language):
        """
        Find text on the image, translate it to the target language, and return image with translated text.
        """
        text = "天気次第"

        translated_text = self.translate_text(text, source_language=source_language, target_language=target_language)

        # dummy for now
        processed_image = np.array(img)[:, ::-1, :]

        return processed_image, translated_text

    def translate_text(self, text, source_language, target_language):
        """
        Translate text from source language to target language.
        """
        self._load_languages(source_language, target_language)

        inputs = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang_code),
            max_length=300
        )

        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
