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


image = cv2.imread('./test_image.jpg')

lane_image = np.copy(image)

# Perform canny
canny_image = perform_canny(lane_image)

show_image_matplot(canny_image)