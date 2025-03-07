"""
    Capturing test images for checking the code and ...
"""


import cv2
import os
import pandas as pd

# make the folder to save images
# output_folder = 'test_images_2DCV_8MP_6_17_24'
output_folder = "SLL_ML/25mm_2_layer_wet_3_7_25"
os.makedirs(output_folder, exist_ok=True)
cameraMatrix = pd.read_csv('/home/kasra/Kasra/SLL/SLL_ML/config/calibration_matrix.csv').values
distCoeffs = pd.read_csv('/home/kasra/Kasra/SLL/SLL_ML/config/distortion_coeff.csv').values


cap = cv2.VideoCapture(0) # the number of output for the camera
# Set capture properties to 8MP resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)


# Set capture properties to 4MP resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1520)




# image file names
img_counter = 10


# main loop for capturing image
while True:
    ret, frame = cap.read()
    frame = cv2.undistort(frame,cameraMatrix, distCoeffs)
    frame = frame[0:2448, 0:2500]

    # Resize the frame for display purposes
    display_frame = cv2.resize(frame, (800, 600))  # Adjust the dimensions as needed

    # display image
    cv2.imshow('Capture Images: (S for save, q for quit)', display_frame)
    # diplay image
    #cv2.imshow('Capture Images: (S for save, q for quit)', frame)


    key = cv2.waitKey(1) & 0xFF

    # save the image with s and quit the program with q
    if key == ord('s'):
        img_name = "{}/test_image_{}.png".format(output_folder,img_counter)
        cv2.imwrite(img_name, frame)
        print("Image {} saved as {}".format(img_counter, img_name))
        img_counter += 1


    elif key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()