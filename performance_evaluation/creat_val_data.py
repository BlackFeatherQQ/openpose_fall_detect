import cv2
import os
import time

def video2image(video_dir):
    cap = cv2.VideoCapture(video_dir)
    i = 0
    while(cap.isOpened()):
        ret,frame = cap.read()

        if frame is None:
            break

        if i % 2 == 0:
            cv2.imshow('test', frame)
            t = time.time()
            t = int(round(t * 1000))
            cv2.imwrite(f'C:/Users/lieweiai/Desktop/val_openpose/images/{t}.jpg', frame)

            cv2.waitKey(10)

        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    root = r'D:\code_data\Le2i\data\Home_01\Videos'

    for list in os.listdir(root):
        video_dir = os.path.join(root,list)

        video2image(video_dir)