import cv2
from urllib.request import urlopen
import numpy as np

def ip_webcam(url):
    while True:
        img_s= urlopen(url)
        img_mat= np.array(bytearray(img_s.read()),dtype=np.uint8)
        img= cv2.imdecode(img_mat,-1)

        cv2.putText(img, 'press q to exit', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(102, 255, 102), 3, cv2.LINE_AA)

        cv2.imshow('phncam_shot', img)
        if cv2.waitKey(1)== ord('q') :
            break
    cv2.destroyAllWindows()

def webcam():
    cam = cv2.VideoCapture(0)
    while True:
        (_, img) = cam.read()

        cv2.putText(img, 'press q to exit', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(102, 255, 102), 3, cv2.LINE_AA)

        cv2.imshow('webcam_shot', img)
        if cv2.waitKey(1)== ord('q') :
            break
    cv2.destroyAllWindows()

if __name__=='__main__':
    choice = int(input("1:webcam\n2:ip-webcam\n3:exit\n"))
    if choice==1:
        # PC webcam 
        webcam()
    elif choice==2:
        # IP webcam
        # url='http://192.168.0.107:8080/shot.jpg'
        url = input("Enter the public ip:")
        url += '/shot.jpg'
        print(url)
        ip_webcam(url)
    else:
        pass