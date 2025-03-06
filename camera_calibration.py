"""
    First run the other code to generate the pictures
    and then use this code for the calibration.

    !!!  Use at least 20 images for the calibration  !!!   
"""

import numpy as np
import cv2
import os
import pandas as pd

# load the saved images from the other code for calibration
image_folder = 'calibrate_images_8MP_6_4_24'
images = [cv2.imread(os.path.join(image_folder, img)) for img in os.listdir(image_folder) if img.endswith(".png")] # change the format if needed!

# cheess board pattern and size
pattern_size = (18, 13)  # no. of intersections in x, y directions
# square_size = 25.0  # in mm (for scaling)

# the points of the object and image
obj_points = []
img_points = []

# generate coordinates for the chess board corners in real world
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# scale the object points based on the measured size
# objp = objp * square_size

# use all images for calibration
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscale the image

    # find the chessboard corners
    ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    #print(corners)


    # after finding corners: add object points and image points
    if ret_corners:

        # Draw the corners on the image
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, ret_corners)

        # Display the image with corners
        cv2.imshow('Image with Corners', img_with_corners)
        cv2.waitKey(500)  # Show the image for 500 milliseconds

        # Close the window with 'Esc' key
        k = cv2.waitKey(1)
        if k == 27:
            break
    

        # add the detected points to the list
        obj_points.append(objp)
        img_points.append(corners)

    else:
        print("No Corners detected in ", img)
# camera calibration function
ret, Kmtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)


 
# Save outputs to config
pd.DataFrame(Kmtx).to_csv('config/calibration_matrix.csv', index=False)
pd.DataFrame(dist).to_csv('config/distortion_coeff.csv', index=False)

# print calibration results

print("focal point x: {}".format(Kmtx[0,0]))
print("focal point y: {}".format(Kmtx[1,1]))
print("principal point x (cx): {}".format(Kmtx[0,2]))
print("principal point y (cy): {} \n".format(Kmtx[1,2]))
print("Calibration matrix: \n {}".format(Kmtx))
print("\nDistortion coefficients:\n")
print(dist)

print("\nRotation Vector: \n {}".format(rvecs))
print("\nTranslation Vector: \n {}".format(tvecs))


print("\nret: \n {}".format(ret))
