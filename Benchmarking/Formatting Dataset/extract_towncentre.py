import os
import cv2
import numpy as np


def video2im(src, images_path='frames', factor=2):
    """
    Extracts the specified range of frames from a video and saves them as jpgs
    """
    os.mkdir(images_path)

    frame = 0
    cap = cv2.VideoCapture(src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print('Total Frame Count:', length )
    
    while True:
        check, img = cap.read()
        if check:
            if frame < 100:
                path = images_path
            else:
                break
            
            img = cv2.resize(img, (1920 // factor, 1080 // factor))
            cv2.imwrite(os.path.join(path, str("frame" + str(frame)) + ".jpg"), img)

            frame += 1
            print('Processed: ',frame, end = '\r')
        
        else:
            break
    
    cap.release()

if __name__ == '__main__':
    video2im('TownCentreXVID.mp4')
