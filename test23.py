import cv2
face_cascade = cv2.CascadeClassifier('haarcascades/myhaar.xml')

img = cv2.imread('./model/5.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.01,7)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x+w/2, y+h/2), (255,0,0),2)

cv2.imshow('Image view', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
