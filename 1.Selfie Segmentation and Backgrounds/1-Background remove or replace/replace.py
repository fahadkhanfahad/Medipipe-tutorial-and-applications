import cv2
import numpy as np
import mediapipe as mp

# Initializing MediaPipe
selfie_segmentation = mp.solutions.selfie_segmentation
# Setting segmentation function
segment = selfie_segmentation.SelfieSegmentation()

# Loading image
image = cv2.imread('fahad.jpg')
image = cv2.resize(image, (450, 400))
background=cv2.imread('background.jpg')
background=cv2.resize(background, (450, 400))

# Converting color
in_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Performing segmentation
result = segment.process(in_RGB)
binary_mask = result.segmentation_mask > 0.9

# Convert the result to an OpenCV image
segmented_image = cv2.cvtColor(cv2.UMat(result.segmentation_mask), cv2.COLOR_BGR2RGB)

binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))

# Create the output image to have white background where ever black is present in the mask.
output_image = np.where(binary_mask_3,image, 255)  

# Now instead of having a white background if you need to add another background image, 
# you just need to replace `255` with a background image in `np.where` function

new_background= np.where(binary_mask_3,image,background )  





cv2.imshow("Original image", image)
cv2.imshow("Performing segmentation", segmented_image)
cv2.imshow("output image",output_image)
cv2.imshow("New background",new_background)

cv2.waitKey(0)
cv2.destroyAllWindows()
