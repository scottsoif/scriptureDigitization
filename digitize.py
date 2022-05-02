# python3
# -*- coding: utf-8 -*-
#
# ===============================================
# Author: Scott A. Soifer
# Email: sas2412@columbia.edu
# Email: soifer00@gmail.com
# Date Created: Sun May 1 6:26:34 EST 2022
# ======================================================


import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import datetime


import cv2
import imutils

import numpy as np
import pathlib
import os
import argparse


idx_to_letter = {(i-ord("א")):chr(i) for i in range(ord('א'), ord("ת")+1)}


def get_words_by_line(cv2_img, english=True):
  canny_ratio = 4
  canny_threshold = 50
  upper_canny_threshold = canny_threshold*canny_ratio
  img_canny = cv2.Canny(cv2_img, canny_threshold, upper_canny_threshold)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
  connected = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)


  cntrs, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  cntrs, median_h = remove_non_words_or_chars(cntrs)

  cntrs_by_line = sort_contrs_by_line(cntrs, english=english)
  return cntrs_by_line

def get_chars_from_word_comps(cv2_word_img, english=True):

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
  connected = cv2.dilate(cv2_word_img, kernel, iterations=2)
 
  canny_ratio = 4
  canny_threshold = 50
  upper_canny_threshold = canny_threshold*canny_ratio
  img_canny = cv2.Canny(cv2_word_img, canny_threshold, upper_canny_threshold)
  
  cntrs, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cntrs, median_h = remove_non_words_or_chars(cntrs, .4)
  bound_rects = np.array([cv2.boundingRect(cntr) for cntr in cntrs])

  bound_rects = sort_contrs_lr(bound_rects, english)

  return bound_rects, cntrs

def get_chars_from_word(bound_rect_of_word, img_blur_bw, im_src_bw, english=True, show_bound_rects_chars=False):
  x, y, w, h = bound_rect_of_word
  word_img = img_blur_bw[y:y+h, x:x+w]
  img_h, img_w = img_blur_bw.shape
  char_imgs_out = [word_img]

  bound_rects_chars_word, cntrs = get_chars_from_word_comps(word_img, english=english)

  if show_bound_rects_chars:
    show_bounds_rects_by_char(word_img, bound_rects_chars_word)

  for i in range(len(bound_rects_chars_word)):
    bound_rects_chars_word[i][0] += x
    bound_rects_chars_word[i][1] += y
    bound_rects_chars_word[i] = pad_char(bound_rects_chars_word[i], img_h, img_w, 1)

  for x1,y1,w1,h1 in bound_rects_chars_word:
    char_img = im_src_bw[y1:y1+h1, x1:x1+w1]
    char_img = resize_char(char_img)
    char_img = reshape_char_square(char_img)
    char_imgs_out.append(char_img)

  return char_imgs_out, bound_rects_chars_word

def pad_char(bound_rects_by_word, img_h, img_w, pad_amnt=3):
  x, y, w, h = bound_rects_by_word
  x = max(0, x-pad_amnt)
  y = max(0, y-pad_amnt)
  w = min(img_w, w+pad_amnt)
  h = min(img_h, h+pad_amnt)
  return x, y, w, h

def preprocess_img_cv2(img_path, bw=False, thresh_level=55, no_blur=False):
  # read, gray and blur image
  cv2_img = cv2.imread(img_path)
  img_out = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) # gray

  if bw:
    _ , img_out = cv2.threshold(img_out, thresh_level, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  if not no_blur:
    img_out = cv2.GaussianBlur(img_out, (5, 5), 1)

  return img_out
  
def resize_char(char_img):
  char_h, char_w = char_img.shape
  if char_h > char_w:
    char_img = imutils.resize(char_img, height=28)
  else:
    char_img = imutils.resize(char_img, width=28)

  return char_img

def reshape_char_square(char_img):

  char_h, char_w = char_img.shape
  char_img = char_img.astype('float64')
  char_img /= 255.0  # normalize pixels to 0,1
  char_img = 1-char_img # invert colors
  rw_start = (28-char_h)//2
  cl_start = (28-char_w)//2

  square_char_img = np.zeros((28, 28))
  square_char_img[rw_start:(rw_start+char_h), cl_start:(cl_start+char_w)] =  char_img
  return square_char_img

def remove_non_words_or_chars(cntrs, h_multiple=.3):
  if len(cntrs)==0:
    return [], 0
  # removes small cntrs
  bound_rects = np.array([cv2.boundingRect(cntr) for cntr in cntrs])
  # w_thresh = .3*np.median(bound_rects[:,2])
  med_h = np.median(bound_rects[:,3])
  h_thresh = h_multiple*med_h
  h_max = 10*med_h
  filter_cntrs = []

  for i in range(len(bound_rects)):
    x, y, w, h = bound_rects[i]
    if w > 5  and h > h_thresh and h < h_max:
      filter_cntrs.append(cntrs[i])

  return filter_cntrs, med_h

def sort_contrs_lr(bound_rects, english=True):

  if len(bound_rects)<2:
    return bound_rects

  bound_rects = sorted(bound_rects, key=lambda x: x[0], reverse=(not english))

  return bound_rects


def sort_contrs_by_line(cntrs, english=True):
  bound_rects = np.array([cv2.boundingRect(cntr) for cntr in cntrs])
  median_h = np.median(bound_rects[:,3])
  line_h_offset = 2*median_h
  bound_rects = sorted(bound_rects, key=lambda x: x[1])
  curr_line_h = line_h_offset
  curr_line = 0
  bound_rects_by_line = [[]]

  for i in range(len(bound_rects)):
    if bound_rects[i][1] < curr_line_h: # bound_rects[i][1] == y

      bound_rects_by_line[curr_line].append(bound_rects[i])

    else: 
      # create new line
      bound_rects_by_line.append([])
      curr_line_h += line_h_offset
      curr_line += 1
      bound_rects_by_line[curr_line].append(bound_rects[i])


  for i in range(len(bound_rects_by_line)):
    bound_rects_by_line[i] = sort_contrs_lr(bound_rects_by_line[i], english)

  return  bound_rects_by_line


def show_bounds_rects_by_char(word_img, bound_rects_chars_word):
  # warning: need to fix function to account for padding
  # print("warning: need to fix function to account for padding")
  bound_box_img = word_img.copy()

  for i in range(len(bound_rects_chars_word)):

    x2, y2, w2, h2 =  bound_rects_chars_word[i]
    # cv2.drawContours(out_img, cntrs[i], -1, (255, 0, 0), 2)
    cv2.rectangle(bound_box_img, (x2,y2),(x2+w2,y2+h2),(0,255, 0),2)

  plt.figure(figsize=(10,5))
  plt.imshow(bound_box_img)

def show_char_imgs(char_imgs, hide_word_img=False):
  for i in range(len(char_imgs_out)):
    if i == 0:
      if hide_word_img:
        continue
      plt.figure(figsize=(10,5))
    else:
      plt.figure(figsize=(3,3))
    plt.imshow(char_imgs_out[i])

def show_bounds_rects_by_word(bound_rects_by_line,img_blur):
  # bound_box_img = np.zeros((*img_blur.shape,3))
  bound_box_img = img_blur.copy()


  i = 1
  for row in range(len(bound_rects_by_line)):
    for word_i in range(len(bound_rects_by_line[row])):

      x, y, w, h =  bound_rects_by_line[row][word_i]

      cv2.putText(bound_box_img, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                  1, (0, 255, 255), 2)
      cv2.rectangle(bound_box_img,(x,y),(x+w,y+h),(0,255,0),2)
      i+=1

  plt.figure(figsize=(15,15))
  plt.imshow(bound_box_img)

##################
## new section
##################

def predict_char(model, idx_to_letter, img,  show_img=False):  

  img = tf.expand_dims(img, axis=0) # add 1
  img = 1-img
  pred = model.predict(img)
  pred_idx = np.argmax(pred, axis=1)[0]
  char_letter = idx_to_letter[pred_idx]
  
  if show_img:
    plt.imshow(tf.squeeze(img, axis=0), plt.get_cmap('gray'))
    plt.show()
    print("Prediction:  ", pred_idx , char_letter, np.max(pred))

  return char_letter

def get_page_out(model, bound_rects_by_line, img_blur_bw, img_src_bw):
    page_output = []
    i = 0
    print("Processing line status:")
    for line_i, line in enumerate(bound_rects_by_line):
        # if i > 10:
        #     break
        print(f"{line_i+1}/{len(bound_rects_by_line)}")
        predicted_line  = []

        for bound_rect_words in line:
            i+= 1 
            pred_word = []

            char_imgs_out, bound_rects_chars_word = get_chars_from_word(bound_rect_words,
                                                            img_blur_bw, 
                                                            img_blur_bw, 
                                                            english=False, 
                                                            show_bound_rects_chars=False
                                                            )
            # plt.show()
            # show_char_imgs(char_imgs_out)
            for img in char_imgs_out[1:]:
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                # img = cv2.erode(img, kernel, iterations=2)
                pred_letter = predict_char(model, idx_to_letter, img, show_img=False)

                pred_word.append(pred_letter)

            pred_word = "".join(pred_word)
            predicted_line.append(pred_word)
            # print(pred_word)

        predicted_line = " ".join(predicted_line)
        page_output.append(predicted_line)
    
    return page_output

def digitize_page(img_path):
        
    # img_path = "data/ancient_img2.png"


    img_blur = preprocess_img_cv2(img_path) # for lines and words
    img_blur_bw = preprocess_img_cv2(img_path, bw=True, thresh_level=55) # for chars
    img_src_bw = preprocess_img_cv2(img_path, no_blur=True) # for char_img
    img_raw = cv2.imread(img_path)

    bound_rects_by_line =  get_words_by_line(img_blur, english=False)
    # show_bounds_rects_by_word(bound_rects_by_line, img_blur)
    model = keras.models.load_model("outputs/ocr_model_hebrew.h5")

    page_output = get_page_out(model, bound_rects_by_line, img_blur_bw, img_src_bw)
    for line in page_output:
        print(line)

def main():
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--i', required=True, dest="filepath", help="input filepath of image to digitize")

    args = parser.parse_args()
    digitize_page(args.filepath)

if __name__=="__main__":
    main()


    