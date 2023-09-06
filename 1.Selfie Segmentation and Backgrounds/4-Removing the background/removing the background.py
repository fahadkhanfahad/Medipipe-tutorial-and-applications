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
background = cv2.imread('background.jpg')
background = cv2.resize(background, (450, 400))

# Converting color
in_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Performing segmentation
result = segment.process(in_RGB)
binary_mask = result.segmentation_mask > 0.9

# Convert the result to an OpenCV image
segmented_image = cv2.cvtColor(cv2.UMat(result.segmentation_mask), cv2.COLOR_BGR2RGB)

# Convert binary mask to uint8
binary_mask_uint8 = binary_mask.astype(np.uint8) * 255

# Stack the original image with the binary mask
output_image = cv2.bitwise_and(image, image, mask=binary_mask_uint8)

cv2.imshow("Original image", image)
cv2.imshow("Performing segmentation", segmented_image)
cv2.imshow("Output image", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
