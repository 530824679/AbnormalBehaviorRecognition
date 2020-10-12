import cv2
import os

scale = 100
save_path = "E:\\datasets\\face\\"
classifier = cv2.CascadeClassifier("D:\OpenCV\opencv\sources\data\haarcascades_cuda\haarcascade_frontalface_alt2.xml")
cap = cv2.VideoCapture(0)
index = 0
while(True):
    ret, frame = cap.read()
    sp = frame.shape
    faceRects = classifier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            # if (x - scale) < 0:
            #     left = 0
            # else:
            #     left = x - scale
            #
            # if (y - scale) < 0:
            #     top = 0
            # else:
            #     top = y - scale
            #
            # if (x + w + scale) > sp[1]:
            #     right = sp[1]
            # else:
            #     right = x + w + scale
            #
            # if (y + h + scale) > sp[0]:
            #     bottom = sp[0]
            # else:
            #     bottom = y + h + scale

            # roi = frame[top:bottom, left:right]
            roi = frame[x:x+w, y:y+h]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('find face', frame)

            img_path = os.path.join(save_path, str(index)) + '.jpg'
            cv2.imwrite(img_path, roi)
            index = index + 1

            if 0xFF == ord('q'):
                break
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()