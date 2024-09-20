import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from argparse import ArgumentParser

from PIL import Image

class Inpainter:
    def __init__(self):
        repo_id = "stabilityai/stable-diffusion-2-inpainting"
        pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

    def boxes_to_mask(self, image_shape, bounding_boxes):
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
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Loop over each bounding box and set the corresponding region in the mask to 1
        for bbox in bounding_boxes:
            x_min, y_min, x_max, y_max = [int(b) for b in bbox]

            mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def inpaint_boxes(self, image, boxes):
        mask = self.boxes_to_mask(image.size[::-1], boxes)
        return self.inpaint(image, mask)

    def inpaint(self, image, mask):
        prompt = ""
        w = image.width
        h = image.height
        if image.width < 512 or image.height < 512:
            temp_img = Image.new('RGB', (512, 512), (0, 0, 0))
            temp_img.paste(image, (0, 0))
            image = temp_img

        image = self.pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=25).images[0]
        return image.crop((0, 0, w, h))

    def process_image(self, image, bounding_boxes):
        """
        Crops 512x512 pixel regions centered on each bounding box and applies a function to each crop.

        :param image_path: Path to the input image.
        :param bounding_boxes: List of bounding boxes, each defined as a tuple (left, upper, right, lower).
        :param function: Function to apply to each cropped image and its corresponding bounding box.
        """
        # Open the image
        d = 256
        out_image = image.copy()
        for box in bounding_boxes:
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
                          d + (right - left)/2,
                          d + (lower - upper)/2)

            filled = self.inpaint_boxes(cropped_img, [scaled_box])
            cropped_img.save("cropped.png")
            filled.save("filled.png")
            print(cropped_img.size)
            print(filled.size)
            out_image.paste(filled.crop(scaled_box), (int(left), int(upper)))
        return out_image



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-b", "--boxes", nargs="*", type=int)
    args = parser.parse_args()

    inpainter = Inpainter()
    image = Image.open(args.image).convert("RGB")
    out_image = inpainter.process_image(image, [args.boxes])
    out_image.save("out.png")
