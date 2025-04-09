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
    for i in range(char_1.shape[0]):
        for j in range(char_1.shape[1]):
            for k in range(3):
                prob = random.random()
                if prob < 0.1:
                    char_1[i, j,k ] += random.randint(-50, 50)

    cv2.imwrite(output, char_1)


y = []

for i in range(0, len(os.listdir("Dataset"))):
    for k in range(100):
        split_char_individual(img="Dataset/" + os.listdir("Dataset")[i],
                              output="chars data/" + str(len(y)) +".png")
        y.append(os.listdir("Dataset")[i][0])


df = pd.DataFrame()
df["y"] = y
df.to_csv('out.csv', index=False)