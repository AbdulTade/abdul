import cv2
import numpy as np
import socket

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

cap = cv2.VideoCapture('./videos/What happens if goalkeeper Neuer challenges Messi .mp4')
i = 0

capInfo = []

while( i <= 18):
    capInfo.append(cap.get(i))
    i += 1
print(capInfo)

while(cap.isOpened()):
    ret,frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',img)
    if(cv2.waitKey(25) & 0xFF == ord("q")):
        break
cap.release()
cv2.destroyAllWindows()