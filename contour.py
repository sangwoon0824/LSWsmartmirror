import cv2
import numpy as np

src = cv2.imread("t.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)


height, width, channel = src.shape


img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=5
)

contours = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
cv2.drawContours(temp_result,contours[0],contourIdx=-1,color=(255,255,255))



print(ret)
#cv2.imshow("test1",dst)
cv2.imshow("test2",temp_result)
cv2.waitKey(0)
cv2.destroyAllWindows()