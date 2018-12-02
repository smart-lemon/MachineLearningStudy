import cv2
import numpy as np
import matplotlib.pyplot as plt

# Prerequistes
# pip3 install opencv-contrib-python

def perform_canny(test_image):

    # Make a copy and work with it - turn it gray 
    gray_scale = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurry_image = cv2.GaussianBlur(gray_scale, (5, 5), 0)

    # Note: Canny does Gaussian blur by itself
    canny_image = cv2.Canny(blurry_image, 50, 105)
    return canny_image

def show_image_ocv(disp_image):

    # Window name - result
    cv2.imshow('result', disp_image)

    # Show image and wait for key
    cv2.waitKey(0)

def show_image_matplot(disp_image):
    plt.imshow(disp_image)
    plt.show()

def display_lines(test_img, lines):
    line_image = np.zeros_like(test_img)
    if lines is not None: 
        for line in lines: 
            x1, y1, x2, y2 = line.reshape(4) 
            cv2.line( line_image, (x1, y1), (x2, y2),
                      (255, 0, 0) )
    return line_image
                  
def region_of_interest(test_image):
    height = test_image.shape[0]

    # Create a triangle
    polygons = np.array([[ (200, height), 
                           (1100, height), 
                           (550, 250) ]])
    mask = np.zeros_like(test_image)
    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(test_image, mask)
    return masked_image

image = cv2.imread('./test_image.jpg')

lane_image = np.copy(image)

# Perform canny
canny_image = perform_canny(lane_image)

# Crop your image - after identifying the region of interest
cropped_image = region_of_interest(canny_image)
show_image_matplot(cropped_image)

# Hough detection in polar 
# where the bin size is of precision of 2 pixels by 1 degree in radians (pi/180)
# and the threshold is minimum no of bin votes needed to detect a line 
lines = cv2.HoughLinesP( cropped_image, 2, np.pi/180,
                         100, # threshold 
                         np.array([]), # placeholder 
                         minLineLength = 40, 
                         maxLineGap = 5 )

# Draw the lines on the image
line_image = display_lines(lane_image, lines)

# Combine the images with the lines - the weight is multiplied by all the pixel
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 
                              1) # the gamma
show_image_matplot(combo_image)

