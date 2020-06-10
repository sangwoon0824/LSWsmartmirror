import cv2

capture = cv2.VideoCapture("./model/a25.mp4")
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("./model/a25.mp4")

    ret, frame = capture.read()
    height, width, channel = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,7)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        
        roi = frame[y:y+h, x:x+w]
    cv2.imshow("VideoFrame",frame)
    cv2.imshow("VideoFrame2",roi)

    

    if cv2.waitKey(33) > 0: break

capture.release()
cv2.destroyAllWindows()