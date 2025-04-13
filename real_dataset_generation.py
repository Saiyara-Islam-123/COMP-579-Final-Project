from extract_state import *

def create_dataset(img, SS_name):
    extract_state(img=img, r_target=131, b_target=156, g_target=118)

    state_vars = ["x.png", "y.png", "vx.png", "vy.png", "d.png"]

    for im in state_vars:
        for index in range(5):
            split_char(im, index, SS_name=SS_name)

for i in range (117, 203):
    create_dataset(img="SS/" + str(i) + ".png", SS_name=str(i))