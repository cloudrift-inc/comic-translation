import PIL.Image
import numpy as np
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
import torch
from argparse import ArgumentParser
from comic.text_generator import TextGenerator
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image, ImageChops

class Inpainter:
    def __init__(self):
        repo_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        pipe = AutoPipelineForInpainting.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")
        self.text_generator = TextGenerator()

    def boxes_to_mask(self, image, bounding_boxes):
        """
        Convert a list of bounding boxes to an image mask.

        Parameters:
        - image_shape: Tuple (height, width) representing the size of the mask to create.
        - bounding_boxes: List of bounding boxes where each bounding box is a tuple
                          (x_min, y_min, x_max, y_max).

        Returns:
        - mask: A binary mask where the regions within the bounding boxes are set to 1, and the rest is 0.
        """
        # Create an empty mask
        mask = np.zeros_like(image)[:, :, 0]

        # Loop over each bounding box and set the corresponding region in the mask to 1
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = [int(b) for b in bbox]

            mask[y_min:y_max, x_min:x_max] = 255


        return mask

    def inpaint_boxes(self, image, boxes, texts):
        mask = self.boxes_to_mask(image, boxes)
        #mask = PIL.Image.fromarray(mask, mode="1")
        mask = np.asarray(mask)
        for box, text in zip(boxes, texts):
            image, m = self.text_generator.generate(image, box, text, fancy=False, generate_mask=True)

            m = np.asarray(m)
            print(mask.dtype, m.dtype)
            mask = np.logical_xor(mask, m)

        #mask.save(f"mask_{boxes[0]}.png")
        Image.fromarray(mask).save(f"mask_{boxes[0]}.png")
        mask_img = Image.fromarray(~mask)
        compose = Image.composite(image, Image.fromarray(np.zeros_like(image)), mask_img)
        compose.save(f"img_{boxes[0]}.png")

        #plt.imsave(f"compose_{boxes[0]}.png", mask, cmap=cm.gray)
        #mask = ~mask
        mask = mask.astype(dtype=np.uint8)
        return self.inpaint(image, mask)

    def inpaint(self, image, mask):
        print(type(image))
        print(image.size)
        print(image.mode)
        print(type(mask))
        print(mask.shape)
        print(mask.dtype)

        # to floating point
        image = np.array(image)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        mask = (mask != 0).astype(np.float32)

        # check shapes
        assert image.ndim == 3 and image.shape[2] == 3
        assert mask.ndim == 2 and mask.shape == image.shape[:2]

        # pad to 512x512
        h, w = image.shape[:2]
        if image.shape[0] < 512 or image.shape[1] < 512:
            temp_img = np.zeros((512, 512, 3), dtype=np.float32)
            temp_img[:h, :w, :] = image.astype(np.float32)
            image = temp_img
            temp_mask = np.zeros((512, 512), dtype=np.float32)
            temp_mask[:h, :w] = mask.astype(np.float32)
            mask = temp_mask

        # inpaint
        prompt = ""
        image = self.pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=25).images[0]

        return image.crop((0, 0, w, h))

    def process_image(self, image, bounding_boxes, texts):
        """
        Crops 512x512 pixel regions centered on each bounding box and applies a function to each crop.

        :param image_path: Path to the input image.
        :param bounding_boxes: List of bounding boxes, each defined as a tuple (left, upper, right, lower).
        :param function: Function to apply to each cropped image and its corresponding bounding box.
        """
        # Open the image
        d = 256
        out_image = image.copy()
        i = 0
        for box, text in zip(bounding_boxes, texts):
            left, upper, right, lower = box
            # Calculate the center coordinates of the bounding box
            center_x = (left + right) / 2
            center_y = (upper + lower) / 2
            # Define the crop area
            crop_left = max(0, center_x - d)
            crop_upper = max(0, center_y - d)
            crop_right = min(image.width, center_x + d)
            crop_lower = min(image.height, center_y + d)
            # Crop the image
            cropped_img = image.crop((crop_left, crop_upper, crop_right, crop_lower))
            # Apply the function to the cropped image and the bounding box
            scaled_box = (left if center_x < d else d - (right - left)/2,
                          upper if center_y < d else d - (lower - upper)/2,
                          right if center_x < d else d + (right - left)/2,
                          lower if center_y < d else d + (lower - upper)/2)

            filled = self.inpaint_boxes(cropped_img, [scaled_box], [text])
            image.crop(box).save(f"crop_{i}.png")
            filled.save("filled.png")
            print(cropped_img.size)
            print(filled.size)
            #filled.crop(scaled_box).save(f"crop_{i}.png")
            out_image.paste(filled.crop(scaled_box), (int(left), int(upper)))
            i += 1
        return out_image


if __name__ == "__main__":
    img_path = "data/manga_no_text.png"
    bbox = [373, 176, 373 + 87, 176 + 111]
    text_japanese = "きれーな もんだなあ"
    text_english = "It's beautiful, isn't it?"

    parser = ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-b", "--boxes", nargs="*", type=int)
    args = parser.parse_args()

    inpainter = Inpainter()
    image = Image.open(img_path).convert("RGB")
    result = inpainter.inpaint_boxes(image, [bbox], [text_english])
    result.save("result.png")
