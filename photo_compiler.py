import cv2
import uuid
import os

# image capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# should run against different gradiants of mask colors
while True:
    # captures form webcam
    ret, frame = cap.read()
    imgname = './Images/NoMask/{}.jpg'.format(str(uuid.uuid1()))

    # writes to file
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)

    # q stops capturing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
