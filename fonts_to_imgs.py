
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import matplotlib.pyplot as plt
import numpy as np
import os




def get_all_paths():
    font_paths = []
    main_path = "/System/Library/Fonts/"
    supplemental_path = "/System/Library/Fonts/Supplemental/"
    heb_path = "Heb_fonts/"

    # for path in os.listdir(main_path):
    #     if path.find('.') > 0 and 'LastResort.otf' not in path:
    #         new_path = main_path + path
    #         font_paths.append(new_path)

    # for path in os.listdir(supplemental_path):
    #     if path.find('.') > 0:
    #         new_path = supplemental_path + path
    #         font_paths.append(new_path)

    for path in os.listdir(heb_path):
        if path.find('.') > 0:
            new_path = heb_path + path
            font_paths.append(new_path)
    
    return font_paths

def has_hebrew(font_ft):
    for tb in font_ft['cmap'].tables:
        if ord("א") in tb.cmap.keys():
            return True
    return False

def get_font_name(font_path):
    start_i = font_path.rfind('/')+1
    end_i = font_path.find('.')
    return font_path[start_i:end_i]


def save_glyphs(font_path):

    point_size = 28
    font = ImageFont.truetype(font_path, point_size)
    for i in range(ord("א"), ord("ת")+1): # iterate hebrew alphabet
        # print(i)
        char_img = Image.new('RGB', (28, 28), (255, 255, 255))

        draw = ImageDraw.Draw(char_img)
        draw.text((14,14), chr(i), anchor="mm", font = font, fill = "#000000")
    #     plt.imshow(canvas)
    #     plt.show(canvas)
    #     canvas.show()
        font_name = get_font_name(font_path)
        char_img.save(f"imgs/{i}_{font_name}.png")

def gen_font_data_set():
    
    font_paths = get_all_paths()
    
    for font_path in font_paths:

        try:
            font_ft = TTFont(font_path)
            if has_hebrew(font_ft):
                save_glyphs(font_path)
                print("Saving font:", get_font_name(font_path) )

        except Exception as e:
            print(e)



if __name__=="__main__":

    gen_font_data_set()


