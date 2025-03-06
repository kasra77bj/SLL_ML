"""
    Align the laser lines with the X and Y directions

    First the camera should be calibrated
"""


import cv2
import os
import pandas as pd

# make the folder to save images
#output_folder = "captured_images_4MP"

cameraMatrix = pd.read_csv('config/calibration_matrix.csv').values
distCoeffs = pd.read_csv('config/distortion_coeff.csv').values

'''
By default, OpenCV has an image resolution of 640x480 so we must set the frame width and height acocordingly
'''
cap = cv2.VideoCapture(0) # the number of output for the camera

# Set capture properties to 8MP resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1520)

# image file names
img_counter = 0


# main loop for capturing image
while True:
    ret, frame = cap.read()
    frame = cv2.undistort(frame,cameraMatrix, distCoeffs)
    frame = frame[0:2448, 0:2500]
    cv2.line(frame, (1560, 1), (1560, 2448), (255, 0, 0), 5)
    cv2.line(frame, (1050, 1), (1050, 2448), (255, 255, 0), 5)
    cv2.line(frame, (1800, 1), (1800, 2448), (255, 0, 0), 5)
    cv2.line(frame, (2200, 1), (2200, 2448), (255, 0, 0), 5)
    cv2.line(frame, (1, 1630), (3264, 1630), (255, 0, 255), 5)
    cv2.line(frame, (1, 400), (3264, 400), (255, 0, 255), 5)
    # frame = frame[0:1788, 700:1788]
    # cv2.line(frame, (800, 1), (800, 2000), (255, 0, 0), 5)
    # diplay image
    # display_frame = cv2.resize(frame, (800, 600))  # Adjust the dimensions as needed
    display_frame = cv2.resize(frame, (1080, 720))  # Adjust the dimensions as needed

    cv2.imshow('Capture Images: (q for quit)', display_frame)

    key = cv2.waitKey(1) & 0xFF



    if key == ord('q'):
        break
    




cap.release()
cv2.destroyAllWindows()