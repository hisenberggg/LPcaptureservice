import cv2
import os

def get_frames(img_path, videopath='/'):
    cap = cv2.VideoCapture(0)  # for live cam
    while True:
        ret, frame = cap.read()
        if ret:
            # give the frames to yolo detector, crop the detections and enhance LP imgs
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
