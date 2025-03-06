import numpy as np
import cv2


def dist_pts(pt1,pt2):
    return np.sqrt(((pt1[0] - pt2[0]) **2) + ((pt1[1] - pt2[1]) **2))


def distance_points(pts):
    return np.sqrt(((pts[0][0] - pts[1][0]) **2) + ((pts[0][1] - pts[1][1]) **2))


def dist_x(pts):
    return pts[0][0] - pts[1][0]


def calc_PPMM(reference_pixel_size, reference_physical_size):
    """
    Calculate the pixel-per-inch (PPI) ratio based on a reference object.
    
    Args:
    - reference_pixel_size: Size of the reference object in pixels.
    - reference_physical_size: Size of the reference object in inches.
    
    Returns:
    - ppi: Pixel-per-inch ratio.
    """
    ppmm = reference_pixel_size / reference_physical_size
    print('PPM: ',ppmm)
    return ppmm


def calc_PPI(reference_pixel_size, reference_physical_size):
    """
    Calculate the pixel-per-inch (PPI) ratio based on a reference object.
    
    Args:
    - reference_pixel_size: Size of the reference object in pixels.
    - reference_physical_size: Size of the reference object in inches.
    
    Returns:
    - ppi: Pixel-per-inch ratio.
    """
    ppi = reference_pixel_size / reference_physical_size
    return ppi


def measure_object_width(object_pixel_size,unit_ratio):
    """
    Measure the size of an object in inches based on its size in pixels and the PPI ratio.
    
    Args:
    - object_pixel_size: Size of the object in pixels.
    - ratio: Pixel-per-unit ratio.
    
    Returns:
    - object_physical_size: Size of the object in inches.
    """
    object_physical_size = object_pixel_size / unit_ratio
    return object_physical_size


def measurements(best_pair,ppm,theta):
    deltaY = abs(best_pair[0][2] - best_pair[1][0])
    deltaY_mm = deltaY/ppm
    height = deltaY_mm * np.tan(np.radians(theta))
    deltaX = abs(best_pair[0][1] - best_pair[0][3])
    width = deltaX/ppm
    return width, height
    
    
def get_lines(img,dist_lasers):
    img_cropped = img[0:2448, 800:2500]
    img_cropped = mask_image(img_cropped)
    ppm, x_thresh, calibration_lines = get_ppm(img,dist_lasers)
    if ppm is None:
        return None, None, None
    contours = detect_contours(img_cropped)
    best_pair = extract_lines(contours,x_thresh)
    best_pair[0][0] += 800
    best_pair[0][2] += 800
    best_pair[1][0] += 800
    best_pair[1][2] += 800
    return best_pair, ppm, calibration_lines
    
    
def get_ppm(img,dist_lasers):
    img = mask_image(img)
    contours = detect_contours(img)
    lowest_x_contour = None
    highest_x_contour = None
    highest_y_contour = None
    lowest_x = float('inf')
    prev_low_x = None
    highest_x = float('-inf')
    prev_high_x = None
    # Iterate through each contour
    for contour in contours:
        if cv2.contourArea(contour) < 4000:
            continue
        # Calculate the bounding rectangle of the contour
        x, y, _, h = cv2.boundingRect(contour)
        # Update the contour with the lowest x value
        if h < 40:
            continue
        if h < 40 or y < 400 or y+h > 1850: 
            continue
        if x < 1150 and y > 400 and y+h < 1950: #depends on the layer
            if x < lowest_x:
                lowest_x = x
                prev_low_x = lowest_x_contour
                lowest_x_contour = contour
        if x > 1150 and y > 400 and y+h < 1950: #x is middle of the lines on layer
        # Update the contour with the highest y value
            if x > highest_x:
                highest_x = x
                prev_high_x = highest_x_contour
                highest_x_contour = contour
    if lowest_x == float('-inf'):
        lowest_x_contour = prev_low_x
    if highest_x == float('inf'):
        highest_x_contour == prev_high_x
    x1, y1, w1, h1 = cv2.boundingRect(lowest_x_contour)
    x2, y2, w2, h2 = cv2.boundingRect(highest_x_contour)
    x3 = 0
    #x3, _, _, _ = cv2.boundingRect(highest_y_contour)
    
    # change the x*w_1 
    calibration_lines = [[x1+int(0.95*w1), y1, x1+int(0.95*w1), (y1+h1)],[x2, y2, x2,y2+h2]]
    delta_lines = abs(calibration_lines[0][0] - calibration_lines[1][0])
    # print('Calibration Lines:',calibration_lines)
    if (x1 == x2):
        return None, None, None
    return calc_PPMM(delta_lines,dist_lasers), x3, calibration_lines


def mask_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the lower and upper ranges for red
    red_lower1 = np.array([0, 80, 90], np.uint8)
    red_upper1 = np.array([18, 255, 255], np.uint8)
    red_lower2 = np.array([160, 100,100], np.uint8)
    red_upper2 = np.array([180, 255, 255], np.uint8)
    # Define the lower and upper boundaries for white in the HSV color space
    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 255, 255], np.uint8)

    # Create a mask for the white color
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    # Create masks for the red color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    # Combine the masks for red
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_combined = cv2.bitwise_or(mask_white, mask_red)
    # Apply the mask to the image
    res = cv2.bitwise_and(img, img, mask=mask_combined)
    return res
    

def detect_contours(img):
    blurred_image = cv2.GaussianBlur(img, (7, 7), 0)
    # Convert the image to grayscale
    kg = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(kg.copy(), cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_NONE)
    return contours


def extract_lines(contours,x_thresh):
    # Initialize variables to keep track of the contours
    best_pair = None
    lowest_x_contour = None
    highest_y_contour = None
    lowest_x = float('inf')
    highest_y = float('-inf')

    if contours is None:
        return None
    # Iterate through each contour
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        if h < 50:
            continue
        # Update the contour with the lowest x value
        if x > 500 and x < 980 and y > 200 and y+h < 1900:
            if x < lowest_x:
                lowest_x = x
                lowest_x_contour = contour
        # if (x > 750) and (y+h > 1850):
        if (x > 1100) and (y+h > 1900):
            # Update the contour with the highest y value
            if y + h > highest_y:
                highest_y = y + h
                highest_y_contour = contour
        
    x1, y1, w1, h1 = cv2.boundingRect(lowest_x_contour)
    x2, y2, w2, h2 = cv2.boundingRect(highest_y_contour)
    best_pair = [[x1+int(0.001*w1)-15, y1+55, x1+int(0.001*w1)-15, (y1+h1-55)],[x2+int(w2/2)+25, y2, x2+int(w2/2)+25,y2+h2]]
    return best_pair


def show_lines(img, best_pair):
    # Create a blank image to draw lines on
    select_line_image = np.zeros_like(img)
    img2 = img.copy()
    cv2.line(select_line_image, (best_pair[0][0], best_pair[0][1]), (best_pair[0][2], best_pair[0][3]), (0, 255, 0), 10)
    cv2.line(select_line_image, (best_pair[1][0]-10, best_pair[1][1]), (best_pair[1][2]-10, best_pair[1][3]), (255,0, 0), 10)
    cv2.circle(select_line_image, (best_pair[0][0], best_pair[0][1]), 5, (0, 0, 255), -1)
    cv2.putText(select_line_image, f"{(best_pair[0][0], best_pair[0][1])}", (best_pair[0][0] + 10, best_pair[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(select_line_image, (best_pair[0][2], best_pair[0][3]), 5, (0, 0, 255), -1)
    cv2.putText(select_line_image, f"{(best_pair[0][2], best_pair[0][3])}", (best_pair[0][2] + 10, best_pair[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(select_line_image, (best_pair[1][0], best_pair[1][1]), 5, (0, 255, 0), -1)
    cv2.putText(select_line_image, f"{(best_pair[1][0], best_pair[1][1])}", (best_pair[1][0] + 10, best_pair[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(select_line_image, (best_pair[1][2], best_pair[1][3]), 5, (0, 255, 0), -1)
    cv2.putText(select_line_image, f"{(best_pair[1][2], best_pair[1][3])}", (best_pair[1][2] + 10, best_pair[1][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)    
    overlay = cv2.addWeighted(img2, 0.7, select_line_image, 0.3, 0)
    return overlay


def detect_edges(img):
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Sobel operator in the X/Y direction
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)    
    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert magnitude to 8-bit image
    magnitude = np.uint8(magnitude)
    # Threshold the magnitude image to isolate the line
    _, thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    return thresh


# Function to calculate the slope of a line given two points
def slope(p1, p2):
    # Avoid division by zero
    if p2[0] == p1[0]:
        return float('inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def find_lines(img):
    # Use HoughLinesP to detect lines in the thresholded image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(gray, 1.75,(np.pi/180)/25, 50, minLineLength=250, maxLineGap=50)

    # Select the best pair of lines (Straightest + longest)
    selected_lines = []
    used_lines = set()
    if lines is None:
        return None
    for i in range(len(lines)):
        if i in used_lines:
            continue
        for j in range(i+1, len(lines)):
            if j in used_lines:
                continue
            # Extract the start and end points of the lines
            x1_1, y1_1, x2_1, y2_1 = lines[i][0]
            x1_2, y1_2, x2_2, y2_2 = lines[j][0]
            # Check if the y-coordinates match with a tolerance
            tolerance = 50  # You can adjust the tolerance value as needed
            # if abs(y1_1 - y2_2) <= tolerance and x1_1 - x2_2 > 200:
            if y2_2 - y2_1 <= tolerance and x1_1 - x2_2 > 200:
                selected_lines.append((lines[i][0], lines[j][0]))
                used_lines.update([i, j])
    # Initialize variables to keep track of the best pair
    max_distance = 0
    best_pair = None
    highest_y = float('-inf')
    lowest_y = float('inf')
    min_x = float('inf')
    x_thresh = 600

    # Iterate through each pair of lines
    for line_pair in selected_lines:
        line1, line2 = line_pair
        # Calculate the distance between the midpoints of the lines
        midpoint1 = ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2)
        midpoint2 = ((line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2)
        current_distance = dist_pts(midpoint1, midpoint2)
        
        # Check the x-coordinates of the points in line1
        for x, y in [(line1[0], line1[1]), (line1[2], line1[3])]:
            if x < x_thresh:
                highest_y = max(highest_y, y)
                lowest_y = min(lowest_y, y)
                min_x = min(min_x, x)
                
        # # Check the x-coordinates of the points in line2
        for x, y in [(line2[0], line2[1]), (line2[2], line2[3])]:
            if x < x_thresh:
                highest_y = max(highest_y, y)
                lowest_y = min(lowest_y, y)
                min_x = min(min_x, x)
        
        # Update the best pair if this pair is further apart and has a smaller slope
        if current_distance > max_distance:
            max_distance = current_distance
            best_pair = line_pair
            best_pair[1][0] = min_x
            best_pair[1][2] = min_x
            best_pair[0][1] = highest_y
            best_pair[1][1] = highest_y
            # best_pair[1][0] = best_pair[1][2] 
            best_pair[1][3] = lowest_y 

    return best_pair





class ROIHandler:
    def __init__(self):
        self.clickedPoints = []
        self.clickedCoordinates = []
        self.count = 0

    
    def get_ROI(self,image):
    
        pts, ROI_coordinates = self.get_ROI_points(image)
        # print("clickedPoints:\n",pts)
        # print("clickedCoordinates: \n",ROI_coordinates)
        # print("\nclickedcoordinate pair: ",ROI_coordinates[0])
        # print("\nclickedcoordinate pair y: ",ROI_coordinates[0][1])
        
        # region of interest (ROI)
        pt_A = pts[0]
        # print("\n pt_A:", pt_A)
        # print("\n pt_A x:", pts[0][0])
        # print("\n pt_A y:", pts[0][1])

        # finding the maximum width of the ROI
        width_AD = np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2))
        width_BC = np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        # finding the maximum height of the ROI
        height_AB = np.sqrt(((pts[0][0] - pts[1][0]) ** 2) + ((pts[0][1] - pts[1][1]) ** 2))
        height_CD = np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        roiPoints = np.array(pts)
        perspectivePoints = np.array([[0,           0],
                                    [0,           maxHeight-1],
                                    [maxWidth-1,  maxHeight-1],
                                    [maxWidth-1,  0]], dtype= np.float32)
        perspectiveMatrix = cv2.getPerspectiveTransform(np.float32(roiPoints), perspectivePoints)    
        
        return perspectiveMatrix, maxWidth, maxHeight
    
    
    def get_ROI_points(self,image):

        """
        This function gets an image and we can click on the points on the image
        and get the coordinates of the clicked points (uses the get_points_coordinates function).
        """

        # display the image for clicking
        cv2.imshow('Click points on the image', image)
        
        # Set mouse callback function
        cv2.setMouseCallback('Click points on the image', self.get_ROI_coordinates)
        
        # quit the program after pressing 'q' 
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.count == 4:
                break

        # close all windows
        cv2.destroyAllWindows()

        # return clicked points and their coordinates
        return self.clickedPoints, self.clickedCoordinates


    def get_ROI_coordinates(self,event, x, y, flags, param):
        # If left mouse button is clicked, record the point and its coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.count += 1
            self.clickedPoints.append((x, y))
            self.clickedCoordinates.append((x, y))
            print("Clicked point:", (x, y))


class x_clicks:
    def __init__(self):
        self.clickedPoints = []
        self.clickedCoordinates = []
        self.count = 0
        
    def get_clicked_points(self,image,num_clicks):
        """
        This function gets an image and we can click on the points on the image
        and get the coordinates of the clicked points (uses the get_points_coordinates function).
        """
        # display the image for clicking
        cv2.imshow('Click points on the image', image)
        
        # Set mouse callback function
        cv2.setMouseCallback('Click points on the image', self.get_points_coordinates)
        
        # quit the program after pressing 'q' 
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.count >= num_clicks:
                break

        # close all windows
        cv2.destroyAllWindows()

        # return clicked points and their coordinates
        return self.clickedPoints, self.clickedCoordinates




    def get_points_coordinates(self,event, x, y, flags, param):
        
        # If left mouse button is clicked, record the point and its coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.count +=1
            self.clickedPoints.append((x, y))
            self.clickedCoordinates.append((x, y))
            print("Clicked point:", (x, y))



class Clicker:
    def __init__(self):
        self.clickedPoints = []
        self.clickedCoordinates = []
        
    def get_clicked_points(self,image):
        """
        This function gets an image and we can click on the points on the image
        and get the coordinates of the clicked points (uses the get_points_coordinates function).
        """
        # display the image for clicking
        cv2.imshow('Click points on the image', image)
        
        # Set mouse callback function
        cv2.setMouseCallback('Click points on the image', self.get_points_coordinates)
        
        # quit the program after pressing 'q' 
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # close all windows
        cv2.destroyAllWindows()

        # return clicked points and their coordinates
        return self.clickedPoints, self.clickedCoordinates




    def get_points_coordinates(self,event, x, y, flags, param):
        
        # If left mouse button is clicked, record the point and its coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clickedPoints.append((x, y))
            self.clickedCoordinates.append((x, y))
            print("Clicked point:", (x, y))