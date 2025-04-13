import cv2
import numpy as np
import math
from extract_state import padding
import os
import random
import pandas as pd

def split_char_individual(img, output):
    arr = cv2.imread(img)
    arr = arr[arr.shape[0] - arr.shape[0]//10:, :]#cropped

    for i in range (arr.shape[1]):
        column = arr[:, i, :]

        if 255 in column:
            break
    mult = arr.shape[1] / 128
    char_1 = arr[:, i: math.ceil(i+4*mult),:]
    char_1 = padding(char_1)

    cv2.imwrite(output, char_1)



for f in range(0, len(os.listdir("data"))):
    image = os.listdir("data")[f]

    for l in range(10):
        new_name = image.strip(".png") + " " + str(l) + ".png"
        char = cv2.imread("data/" + image)
        for i in range(char.shape[0]):
            for j in range(char.shape[1]):
                prob = random.random()
                if prob < 0.01:
                    for k in range(3):
                        char[i, j, k] = 255 - char[i, j, k]

        cv2.imwrite("Aug Data/"+new_name, char)



