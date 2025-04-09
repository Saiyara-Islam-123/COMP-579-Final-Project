import cv2
import numpy as np
import math

def padding(arr):
    # padding
    row, column, layer = (arr.shape)
    padding = np.zeros((row, row - column, layer), dtype=np.uint8)
    arr = np.hstack((padding, arr))

    return arr

def scale_up(arr_dim):

    #raw pixel values, convert to my screenshot's resolution
    mult = arr_dim/128
    x_upscaled = (0, math.ceil(16 * mult))
    y_upscaled = (math.ceil(16 * mult), math.ceil(36 * mult))
    vx_upscaled = (math.ceil(51 * mult), math.ceil(79 * mult))
    vy_upscaled = (math.ceil(79 * mult), math.ceil(93 * mult))
    d_upscaled = (math.ceil(92 * mult), math.ceil(127 * mult))

    return x_upscaled, y_upscaled, vx_upscaled, vy_upscaled, d_upscaled

def crop(img):
    frame = cv2.imread(img)
    arr = np.array(frame)
    black_pixels = 1

    for i in range(arr.shape[1]): #iterate through first row
        b = arr[0, i, 0]
        g = arr[0, i, 1]
        r = arr[0, i, 2]
        if b == 0 and g == 0 and r == 0:
            black_pixels += 1

        else:
            break


    new_arr = arr[:, black_pixels: arr.shape[1] - black_pixels, :]

    return new_arr

def extract_state(img, r_target, b_target, g_target):
    arr = crop(img)

    x, y, vx, vy, d = scale_up(arr.shape[0]) #all ranges

    ranges = [x, y, vx, vy, d]
    count = 0
    for r in ranges:
        start = r[0]
        end = r[1]
        new_arr = np.zeros((arr.shape[0], arr.shape[1],1), dtype=np.uint8)
        for i in range(arr.shape[0]):
            for j in range(start, end+1):
                b = arr[i, j, 0]
                g = arr[i, j, 1]
                r = arr[i, j, 2]

                if b == b_target and r == r_target and g == g_target:
                    new_arr[i, j, 0] = 255


        if count == 0:
            cv2.imwrite( "x.png", new_arr)

        elif count == 1:
            cv2.imwrite( "y.png", new_arr)

        elif count == 2:
            cv2.imwrite( "vx.png", new_arr)
        elif count == 3:
            cv2.imwrite( "vy.png", new_arr)
        else:
            cv2.imwrite( "d.png", new_arr)
        count += 1

def split_char(img):
    arr = cv2.imread(img)
    arr = arr[arr.shape[0] - arr.shape[0]//10:, :]#cropped
    print(arr.shape)

    for i in range (arr.shape[1]):
        column = arr[:, i, :]

        if 255 in column:
            break
    mult = arr.shape[1] / 128
    char_1 = arr[:, i: math.ceil(i+3*mult),:]
    char_2 = arr[:, math.ceil(i+3*mult): math.ceil(i+7*mult), :]
    char_3 = arr[:, math.ceil(i + 7 * mult): math.ceil(i + 11 * mult), :]
    char_4 = arr[:, math.ceil(i + 11 * mult): math.ceil(i + 15 * mult), :]

    char_1 = padding(char_1)
    char_2 = padding(char_2)
    char_3 = padding(char_3)
    char_4 = padding(char_4)


    cv2.imwrite("char1.png", (char_1))
    return char_1, char_2, char_3, char_4

split_char("vx.png")
#extract_state(img="SS/5.png", r_target=131, b_target=156, g_target=118)