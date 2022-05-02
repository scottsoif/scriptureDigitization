# python3
# -*- coding: utf-8 -*-
#
# ===============================================
# Author: Scott A. Soifer
# Email: sas2412@columbia.edu
# Email: soifer00@gmail.com
# Date Created: Sun May 1 6:26:34 EST 2022
# ======================================================


from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

import matplotlib.pyplot as plt
import numpy as np
import os



def get_all_paths():
    font_paths = []
    main_path = "/System/Library/Fonts/"
    supplemental_path = "/System/Library/Fonts/Supplemental/"
    heb_path = "data/Heb_fonts/"

    for path in os.listdir(main_path):
        if path.find('.') > 0 and 'LastResort.otf' not in path:
            new_path = main_path + path
            font_paths.append(new_path)

    for path in os.listdir(supplemental_path):
        if path.find('.') > 0:
            new_path = supplemental_path + path
            font_paths.append(new_path)

    for path in os.listdir(heb_path):
        if path.find('.') > 0:
            new_path = heb_path + path
            font_paths.append(new_path)
    
    print(font_paths)
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

def remove_border_and_resize(char_img):
    # remove white border from canvas region
    char_img_np = np.array(char_img)
    mask = np.where(char_img_np!=255)
    bnd_rect = [np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])]

    # make square bounding box
    if (bnd_rect[3]-bnd_rect[2]) > (bnd_rect[1]-bnd_rect[0]):
        h = (bnd_rect[3]-bnd_rect[2])
        center = (bnd_rect[1]+bnd_rect[0])//2
        bnd_rect[1] = center + h//2
        bnd_rect[0] = center - h//2
    else:
        w = (bnd_rect[1]-bnd_rect[0])
        center = (bnd_rect[3]+bnd_rect[2])//2
        bnd_rect[3] = center + w//2
        bnd_rect[2] = center - w//2 

    char_img_np = char_img_np[bnd_rect[0]:bnd_rect[1], bnd_rect[2]:bnd_rect[3], :]
    char_img = Image.fromarray(char_img_np)
    char_img = char_img.resize((28,28))

    return char_img

def save_glyphs(font_path):

    point_size = 100
    font = ImageFont.truetype(font_path, point_size)
    for i in range(ord("א"), ord("ת")+1): # iterate hebrew alphabet
        # print(i)
        char_img = Image.new('RGB', (125, 125), (255, 255, 255))

        draw = ImageDraw.Draw(char_img)
        draw.text((62,62), chr(i), anchor="mm", font = font, fill = "#000000")
        font_name = get_font_name(font_path)
        # char_img.show()
        # print(chr(i), i)
        char_img = remove_border_and_resize(char_img)

        # char_img.show()

        # break
        char_img.save(f"data/imgs/{i}_{font_name}.png")

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


