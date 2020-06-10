import cv2 # opencv 사용
import numpy as np
face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def face_extractor(img):
    #흑백처리 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #얼굴 찾기 
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #찾은 얼굴이 없으면 None으로 리턴 
    if faces is():
        return None
    #얼굴들이 있으면 
    for(x,y,w,h) in faces:
        #해당 얼굴 크기만큼 cropped_face에 잘라 넣기 
        #근데... 얼굴이 2개 이상 감지되면??
        #가장 마지막의 얼굴만 남을 듯
        cropped_face = img[y:y+h, x:x+w]
    #cropped_face 리턴 
    return cropped_face
    
src = cv2.imread('./faces/user3.jpg',cv2.IMREAD_COLOR) # 이미지 읽기


#height, width = src.shape[:2]
#M = np.float32([[1, 0, 80], [0, 1, 0]]) # 이미지를 오른쪽으로 100, 아래로 25 이동시킵니다.
#img_translation = cv2.warpAffine(src, M, (width,height))
roi = src[100:200,50:150]


#cv2.imshow("src", roi)

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##image = cv2.resize(image,(200,200))




#cv2.imshow('white',image) # 흰색 추출 이미지 출력
#cv2.imshow('result',image) # 이미지 출력

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

#faces = face_cascade.detectMultiScale(gray, 1.3,5)
#for (x,y,w,h) in faces:
#    cv2.rectangle(roi, (x,y), (x+w, y+h), (255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = roi[y:y+h, x:x+w]





mark = np.copy(roi) # image 복사
gray = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
image_enhanced = gray


#  BGR 제한 값 설정
blue_threshold = 170
green_threshold = 170
red_threshold = 170
bgr_threshold = [blue_threshold, green_threshold, red_threshold]

# BGR 제한 값보다 작으면 검은색으로
thresholds = (roi[:,:,0] < bgr_threshold[0]) \
            | (roi[:,:,1] < bgr_threshold[1]) \
            | (roi[:,:,2] < bgr_threshold[2])
mark[thresholds] = [0,0,0]



img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)


for i in range (0, 100):
    for j in range (0, 100):
        if img_blurred[i,j] > 155:
            img_blurred[i,j] = 255
            continue
        img_blurred[i,j] += 100



img_thresh = cv2.adaptiveThreshold(
    img_blurred, 
    maxValue=255.0, 
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, 
    C=6
)
cv2.imshow('Image view', img_blurred)
cv2.imshow('Image view2', img_thresh)
cv2.waitKey(0)
cv2.waitKey(0)