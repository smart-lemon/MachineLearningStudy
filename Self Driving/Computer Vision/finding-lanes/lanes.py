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
                      (25, 255, 0) )
    return line_image

def display_lines_avg(test_img, lines):
    line_image = np.zeros_like(test_img)
    if lines is not None: 
        for line in lines: 
            x1, y1, x2, y2 = line 
            cv2.line( line_image, (x1, y1), (x2, y2),
                      (25, 255, 0) )
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

def make_points(test_image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(test_image.shape[0])   # bottom of the image
    y2 = int(y1 * 3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    # Contains intercept of line on left and right
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)

            # Detemine the slope and intercept
            slope = fit[0]
            intercept = fit[1]

            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # Add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def detect_lanes_in_image():
    # Open the image and make copy
    image = cv2.imread('./test_image.jpg')
    lane_image = np.copy(image)

    # Perform canny
    canny_image = perform_canny(lane_image)

    # Crop your image - after identifying the region of interest
    cropped_image = region_of_interest(canny_image)

    # For debugging
    show_image_matplot(cropped_image)

    # Hough detection in polar 
    # where the bin size is of precision of 2 pixels by 1 degree in radians (pi/180)
    # and the threshold is minimum no of bin votes needed to detect a line 
    lines = cv2.HoughLinesP( cropped_image, 2, np.pi/180,
                            50, # threshold 
                            np.array([]), # placeholder 
                            minLineLength = 40, 
                            maxLineGap = 5 )

    # Draw the lines on the image
    line_image = display_lines(lane_image, lines)

    # Combine the images with the lines - the weight is multiplied by all the pixel
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 
                                1) # the gamma
    show_image_matplot(combo_image)

    averaged_lines = average_slope_intercept(lane_image, lines)

    # Add the average lines and mix it into the image
    line_image = display_lines(lane_image, np.array(averaged_lines))
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    show_image_matplot(combo_image)

# Detect lane lines in an image
# detect_lanes_in_image()

# Detect the lane lines in a video 
def detect_lanes_in_video():

    cap = cv2.VideoCapture("/Users/renoReno/Documents/DataSet/SdVideo.mp4")
    while(cap.isOpened()):
        _, frame = cap.read()
        canny_image = perform_canny(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP( cropped_canny, 2, np.pi/180, 100, np.array([]), 
                                minLineLength = 40, maxLineGap = 5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, np.array(averaged_lines))
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("Lane detection on video", combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

detect_lanes_in_video() 