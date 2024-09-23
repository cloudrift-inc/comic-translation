from pathlib import Path
from numpy.random import Generator, PCG64
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import string
import numpy as np



def clip(val, low, high):
    return int(max(low, min(high, val)))

def get_avg_color_wh(img, box):
    colors = []
    dx = 5
    box = [clip(box[0] + dx, 0, img.width - 1),
           clip(box[1] + dx, 0, img.height - 1),
           clip(box[2] - dx, 0, img.width - 1),
           clip(box[3] - dx, 0, img.height - 1)]
    colors.append(img.getpixel((box[0], box[1])))
    colors.append(img.getpixel((box[2], box[3])))
    colors.append(img.getpixel((box[0], box[3])))
    colors.append(img.getpixel((box[2], box[1])))
    color = np.median(colors, axis=0).astype(int)
    return tuple(color)

def multiline_text(draw, pos, text, box_width, box_height, font, color=0, place='left',
                   contour=0, contour_color=255, spacing=0):
    justify_last_line = False
    x, y = pos
    if contour > 0:
        mask = Image.new("L", draw.im.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        multiline_text(mask_draw, pos, text, box_width=box_width, box_height=box_height, font=font,
                       color=255, place=place, contour=0, spacing=spacing)
        mask = mask.filter(ImageFilter.MaxFilter(contour*2 + 1))
        draw.bitmap((0, 0), mask, fill=contour_color)

    h, w = 0, 0
    lines = []
    line = []
    words = text.split()
    line_spacing = font.getsize(string.ascii_letters)[1] + spacing
    for word in words:
        new_line = ' '.join(line + [word])
        size = font.getsize(new_line)
        if size[0] <= box_width:
            line.append(word)
            w = max(w, size[0])
        else:
            h += line_spacing
            lines.append(line)
            line = [word]
    if line:
        lines.append(line)
        h += line_spacing
        size = font.getsize(line[0])
        w = max(w, size[0])
    lines = [' '.join(line) for line in lines if line]
    height = y + box_height // 2 - h // 2
    y = height
    x_min = x + w
    for index, line in enumerate(lines):
        if place == 'left':
            write_text(draw, (x, height), line, font=font, color=color)
        elif place == 'right':
            total_size = font.getsize(line)
            x_left = x + box_width - total_size[0]
            if x_left < x_min:
                x_min = x_left
            write_text(draw, (x_left, height), line, font=font, color=color)
        elif place == 'center':
            total_size = font.getsize(line)
            x_left = int(x + ((box_width - total_size[0]) / 2))
            if x_left < x_min:
                x_min = x_left
            write_text(draw, (x_left, height), line, font=font, color=color)
        elif place == 'justify':
            words = line.split()
            if (index == len(lines) - 1 and not justify_last_line) or \
                    len(words) == 1:
                write_text(draw, (x, height), line, font=font, color=color)
                continue
            line_without_spaces = ''.join(words)
            total_size = font.getsize(line_without_spaces)
            space_width = (box_width - total_size[0]) / (len(words) - 1.0)
            start_x = x
            for word in words[:-1]:
                write_text(draw, (start_x, height), word, font=font, color=color)
                word_size = font.getsize(word)
                start_x += word_size[0] + space_width
            last_word_size = font.getsize(words[-1])
            last_word_x = x + box_width - last_word_size[0]
            write_text(draw, (last_word_x, height), words[-1], font=font, color=color)
        height += line_spacing
    return x_min, y, w, h


def write_text(draw, pos, text, font, color=0, contour=0, contour_color=255):
    x, y = pos

    if contour > 0:
        mask = Image.new("L", draw.im.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        write_text(mask_draw, pos, text, font=font, color=255, contour=0)
        mask = mask.filter(ImageFilter.MaxFilter(contour))
        draw.bitmap((0, 0), mask, contour_color)

    text_size = font.getsize(text)
    draw.text((x, y), text, font=font, fill=color)
    return text_size


def write_contour(draw, pos, text, font, color=0, contour_color=255):
    mask = Image.new("L", draw.im.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    write_text(mask_draw, pos, text, font, 255)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    draw.bitmap((0, 0), mask, fill=contour_color)
    write_text(draw, pos, text, font, color)

def box_from_mask(mask):
    pos = np.nonzero(np.array(mask) > 20)
    if len(pos[0]) == 0:
        return None
    xmin = int(np.min(pos[1]))
    xmax = int(np.max(pos[1]))
    ymin = int(np.min(pos[0]))
    ymax = int(np.max(pos[0]))
    pos = [xmin, ymin, xmax - xmin, ymax - ymin]
    return pos
class TextGenerator:
    def __init__(self):
        #self.font_files = []
        self.min_font_size = 14
        self.max_font_size = 60
        self.rg = Generator(PCG64())
        self.font = Path(__file__).parent.parent/"fonts/plain_fonts/mangat.ttf"
        #self.fancy_fonts = list((config.fonts_dir / 'fancy_fonts').iterdir())
        #self.plain_fonts = list((config.fonts_dir / 'fancy_fonts').iterdir())

    def find_font_size(self, text, box, font_file, spacing):
        box_area = box[2]*box[3]
        font = ImageFont.truetype(str(font_file), size=self.min_font_size)
        font_size = font.getsize(text)
        font_area = font_size[0]*(font_size[1] + spacing + 4)
        multiplier = math.sqrt(box_area*0.4 / font_area)
        return int(self.min_font_size*multiplier)

    def generate(self, image, box, text, fancy=False, generate_mask=True, mask=None):
        if fancy:
            #font_file = self.rg.choice(self.fancy_fonts)
            spacing = 4
        else:
            #font_file = self.rg.choice(self.plain_fonts)
            spacing = 2
        font_file = self.font

        font_size = self.find_font_size(text, box, font_file, spacing=spacing)

        font = ImageFont.truetype(str(font_file), size=font_size)
        image = image.copy()
        draw = ImageDraw.Draw(image)
        x = box[0]
        y = box[1]
        box_width = box[2]
        if fancy:
            contour = self.rg.choice([1, 2, 3])
        else:
            contour = 0
        if self.rg.random() > 0.5:
            text = text.upper()
        bg_color = get_avg_color_wh(image, (box[0], box[1], box[0] + box[2], box[1] + box[3]))
        if np.mean(bg_color) > 128:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        contour_color = (255 - color[0], 255 - color[0], 255 - color[0])
        pos = multiline_text(draw, (x, y), text, box_width, box_height=box[3], font=font, place='center', color=color,
                             contour=contour, contour_color=contour_color, spacing=spacing)
        if generate_mask:
            mask_color = 255
            if mask is None:
                mask = Image.new("L", draw.im.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            multiline_text(mask_draw, (x, y), text, box_width, box_height=box[3], font=font, place='center',
                           color=mask_color, contour=contour, contour_color=mask_color, spacing=spacing)
            return image, mask
        return image
