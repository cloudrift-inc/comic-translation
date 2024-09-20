import csv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from comic import DATA_DIR


class TextTranslator:
    TOKEN = "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"

    def __init__(self):
        self.available_languages = []
        self._load_language_list()

        self.pipeline = None

        self.src_lang_code = None
        self.tgt_lang_code = None

    def _load_language_list(self):
        with open(DATA_DIR / 'FLORES-200_code.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip header
            for row in reader:
                self.available_languages.append(row)

    def _load_languages(self, source_language, target_language):
        src_lang = [lang[1] for lang in self.available_languages if lang[0] == source_language][0]
        tgt_lang = [lang[1] for lang in self.available_languages if lang[0] == target_language][0]

        if self.pipeline is None or src_lang != self.pipeline.src_lang:
            tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", token=self.TOKEN, src_lang=src_lang)
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token=self.TOKEN)
        else:
            tokenizer = self.pipeline.tokenizer
            model = self.pipeline.model

        if self.pipeline is None or tokenizer != self.pipeline.tokenizer or model != self.pipeline.model or tgt_lang != self.pipeline.tgt_lang:
            self.pipeline = pipeline("translation", model=model, tokenizer=tokenizer,
                                     src_lang=src_lang, tgt_lang=tgt_lang, max_length=400, device=0)

    def translate_text(self, text, source_language, target_language):
        """
        Translate text from source language to target language.
        """
        self._load_languages(source_language, target_language)
        result = self.pipeline(text)
        return result[0]['translation_text']


if __name__ == '__main__':
    translator = TextTranslator()
    text = "Hello, how are you? Today is a beautiful day, don't you think? I'm going to the park."
    translated_text = translator.translate_text(text, "English", "Russian")
    print(translated_text)
