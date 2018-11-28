import cv2

image = cv2.imread('./test_image.jpg')

# Window name - result
cv2.imshow('result', image)

# Show image and wait for key
cv2.waitKey(0)

# Make a copy and work with it - turn it gray 
lane_image = np.copy(image) 
gray_scale = cv2.cvtColor(lane_image. cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
blurry_iamge = cv2.GaussianBlur(gray_scale, (5, 5), 0)

# Note: Canny does Gaussian blur by itself






# Prerequistes
# pip3 install opencv-contrib-python