from transformers import pipeline
from PIL import Image


class TextBlockDetector:
    def __init__(self):
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")

    def detect(self, image):
        predictions = self.detector(
            image,
            candidate_labels=["text block", "text"],
        )
        #print(predictions)
        boxes = [[p["box"]["xmin"] + 5, p["box"]["ymin"] + 5, p["box"]["xmax"] - 5, p["box"]["ymax"] - 5] for p in predictions]
        return boxes


if __name__ == "__main__":

    img = Image.open("/home/red-haired/Projects/comic-translation/data/aligned_jap/shokugeki_no_soma/31/2.png")#DATA_DIR / 'the_werewolf_stalks.jpg')
    detector = TextBlockDetector()
    res = detector.detect(img)
    print(res)
    #processed_img.save('result.jpg')
