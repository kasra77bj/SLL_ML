import numpy as np
import cv2
import os
import pandas as pd
from functions import show_lines, get_lines, measurements, get_ppm
from time import time, sleep 

cameraMatrix = pd.read_csv('/home/kasra/Kasra/SLL/SLL_ML/config/calibration_matrix.csv').values
distCoeffs = pd.read_csv('/home/kasra/Kasra/SLL/SLL_ML/config/distortion_coeff.csv').values
# ppm = pd.read_csv('config/ppm.csv').values[0][0]
# Angle in degrees
theta = 90 - 38
# theta = 48.15
dist_lasers = (30 / np.sin(np.radians(theta)))

# lists for the transition part
transition_list = []
width_list = []
ppm_list = []
height_list = []
time_list = []

# make the folder to save images
# output_folder = "test"
# output_folder = "test_results_8MP_5_21_24_moving_6cms_100mm_test_1_normal"
output_folder = "Phase2_results_5_27_24_deformed_6cm_test_4"
os.makedirs(output_folder, exist_ok=True)
# debug_folder = output_folder + "\debug"
# os.makedirs(debug_folder, exist_ok=True)


'''
By default, OpenCV has an image resolution of 640x480 so we must set the frame width and height acocordingly
'''
cap = cv2.VideoCapture(0) # the number of output for the camera

# Set capture properties to 8MP resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)

# 4MP Resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1520)

# image file names
img_counter = 1
start_number = 0
output_counter = 0
# Initialize an empty DataFrame to store the results
# results_df = pd.DataFrame(columns=['Image_Name', 'Time' , 'Width', 'Height'])
results_df = pd.DataFrame(columns=['Image Number','Time' , 'Width', 'Height', 'PPM'])

# cv2.namedWindow('Live Feed',cv2.WINDOW_NORMAL)
cv2.namedWindow('Debug Image',cv2.WINDOW_NORMAL)

# main loop for capturing image
while True:
    ret, frame = cap.read()

    t_start = time()

    # Load the image

    # Dimensions are  1520 high x  2668 wide
    frame = cv2.undistort(frame,cameraMatrix, distCoeffs)
    frame = frame[0:2448, 0:2500] #==== changed this one
    
    display_frame = cv2.resize(frame, (1080,720))  # Adjust the dimensions as needed
    # cv2.imshow('Live Feed', display_frame)
    # diplay image
    key = cv2.waitKey(1) & 0xFF
    # Quit the program with q
    if key == ord('q'):
        break

    # Get the two lines that correlate with the layer and the platform
    best_pair, ppm, calibration_lines = get_lines(frame, dist_lasers)
    print("best pair: \n {}".format(best_pair))
    
    # finishing the one round of scan and override the results to the dataframe
    if best_pair == None and len(width_list) > 3:
        length = len(width_list)
        print("length of the width list: ", length)
        for i in range(1, length-1):
            ppm_value = ppm_list[i]
            width_value = width_list[i]
            height_value = height_list[i]
            time_value = time_list[i]
            output_counter = output_counter + 1
            results_df = results_df.append({'Image Number': i,'Time': time_value, 'Width': width_value, 'Height': height_value,'PPM': ppm_value}, ignore_index=True)
        # removing the first and last two measurements
        # width_list = width_list[2:-2]
        # ppm_list = ppm_list[2:-2]
        # height_list = height_list[2:-2]
        # time_list = time_list[2:-2]
        start_number = start_number + 3
        numbers = list(range(start_number, start_number + len(width_list)))
        start_number = start_number + len(width_list) + 1
        
        # appending the list to the dataframe
        # for          
        # results_df = results_df.append({'Image Number': numbers,'Time': time_value, 'Width': width_value, 'Height': height_value,'PPM': ppm_value}, ignore_index=True)

        # clearing the lists
        ppm_list.clear()
        width_list.clear()
        height_list.clear()
        time_list.clear()
        
        
    # Lines on the platform
    if best_pair == None:
        print("No Lines found.")
        ppm_list.clear()
        width_list.clear()
        height_list.clear()
        time_list.clear()
        continue
    
    transition_list.append(best_pair)
    width, height = measurements(best_pair,ppm,theta)
    width_list.append(width)
    height_list.append(height)
    ppm_list.append(ppm)
    
    debug_img = show_lines(frame,best_pair)
    t_process = time() - t_start
    time_list.append(t_process)
    
    # print("Time: ", time() - t_start)
    # print('PPM: ',ppm)
    # print("Best pair: ",best_pair)
    # print(calibration_lines)
    print('Width: ',width)
    print('Height: ', height)
    print("Counter: ", output_counter)
    cv2.imshow('Debug Image', debug_img)
    # debug_img_name = "{}\image_{}.png".format(output_folder,img_counter)
    # cv2.imwrite(debug_img_name, debug_img)
    # img_name = "{}\image_{}.png".format(output_folder,img_counter)
    # cv2.imwrite(img_name, frame)
    # print("Image {} saved as {}".format(img_counter, debug_img_name))


    # Append the results to the DataFrame
    # results_df = results_df.append({'Image_Name': debug_img_name, 'Time': t_process, 'Width': width, 'Height': height,'PPM': ppm}, ignore_index=True)
    img_counter += 1


    

# Save the DataFrame to an Excel file
results_df.to_excel(output_folder+'.xlsx', index=False)

    

cap.release()
cv2.destroyAllWindows()