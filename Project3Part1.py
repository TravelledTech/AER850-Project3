# ===== Part 1. Masking =====
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Img (5792, 4344)
img = cv.imread("motherboard_image.JPEG")
greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(greyImg, (5,5), 0)

# #Adaptive Gaussian
# thresh = cv.adaptiveThreshold(
#     blur, 
#     255, 
#     cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     cv.THRESH_BINARY_INV, 
#     71, 
#     10
# )

ret, thresh = cv.threshold(blur, 120, 255, cv.THRESH_BINARY_INV)
edges = cv.Canny(thresh,50,200)

contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cnt = max(contours, key=cv.contourArea)
contours = [cnt]

mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)

final = cv.bitwise_and(img, img, mask = mask)

# #Prints image
# cv.namedWindow("Threshold", cv.WINDOW_NORMAL)
# cv.imshow("Threshold", thresh)

# cv.namedWindow("Edges", cv.WINDOW_NORMAL)
# cv.imshow("Edges", edges)

# cv.namedWindow("Mask", cv.WINDOW_NORMAL)
# cv.imshow("Mask", mask)

# cv.namedWindow("Final", cv.WINDOW_NORMAL)
# cv.imshow("Final", final)
cv.imwrite("Threshold.JPEG", thresh)
cv.imwrite("Mask.JPEG", mask)
cv.imwrite("Final.JPEG", final)