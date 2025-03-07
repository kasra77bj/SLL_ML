"""
    First run this code to get 20 photos for calibration. 
    Use at least 20 images!
    Capture the images from different angles and distances.
"""


import cv2
import os

# make the folder to save images
#output_folder = "captured_images_4MP"
output_folder = "SLL_ML/calibrate_images_8MP_3_6_25"
# output_folder = "2DCV_images_8MP_6_17_24"
# output_folder = "Michael"

os.makedirs(output_folder, exist_ok=True)

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
img_counter = 1


# main loop for capturing image
while True:
    ret, frame = cap.read()

    # diplay image
    display_frame = cv2.resize(frame, (800, 800))  # Adjust the dimensions as needed
    cv2.imshow('Capture Images: (S for save, q for quit)', display_frame)

    key = cv2.waitKey(1) & 0xFF

    # save the image with s and quit the program with q
    if key == ord('s'):
        img_name = "{}/image_{}.png".format(output_folder,img_counter) #==== change it to / instead of \
        cv2.imwrite(img_name, frame)
        print("Image {} saved as {}".format(img_counter, img_name))
        img_counter += 1


    elif key == ord('q'):
        break
    


cap.release()
cv2.destroyAllWindows()