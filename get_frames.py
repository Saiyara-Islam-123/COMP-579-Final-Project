import mss
import cv2
import numpy as np
import win32gui
import time

window_handle = win32gui.FindWindow(None, "CELESTE.P8 (PICO-8)")

frame_count = 0

frame_count = 0
with mss.mss() as sct:
    time.sleep(10)

    monitor = sct.monitors[0]
    img = sct.grab(monitor)
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Save frame
        cv2.imwrite("SS/" + str(frame_count)+'.png', frame)
        frame_count += 1

cv2.destroyAllWindows()