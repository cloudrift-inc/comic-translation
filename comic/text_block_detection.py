from transformers import pipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import torch


class TextBlockDetector:
    def __init__(self):
        self.device = "cuda"
        #checkpoint = "google/owlv2-base-patch16-ensemble"
        #self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect(self, image):
        inputs = self.processor(images=image, text="a text.", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # predictions = self.detector(
        #     image,
        #     candidate_labels=["text"],
        # )
        #print(predictions)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        print("results", results)

        #boxes = [[p["box"]["xmin"] + 5, p["box"]["ymin"] + 5, p["box"]["xmax"] - 5, p["box"]["ymax"] - 5] for p in predictions]
        return results[0]["boxes"].tolist()


if __name__ == "__main__":

    img = Image.open("/home/red-haired/Projects/comic-translation/data/aligned_jap/shokugeki_no_soma/31/2.png")#DATA_DIR / 'the_werewolf_stalks.jpg')
    detector = TextBlockDetector()
    res = detector.detect(img)
    print(res)
    #processed_img.save('result.jpg')
