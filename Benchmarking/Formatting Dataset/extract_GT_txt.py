import os
import numpy as np 
import pandas as pd

if __name__ == '__main__':

    GT = pd.read_csv('TownCentre-groundtruth.top', header=None)
    indent = lambda x,y: ''.join(['  ' for _ in range(y)]) + x

    factor = 2
    total_frames = 100

    os.mkdir('groundtruths')
    name = 'person'
    width, height = 1920 // factor, 1080 // factor

    for frame_number in range(total_frames):
        
        Frame = GT.loc[GT[1] == frame_number] 
        x1 = list(Frame[8])
        y1 = list(Frame[11])
        x2 = list(Frame[10])
        y2 = list(Frame[9])
        points = [[(round(x1_), round(y1_)), (round(x2_), round(y2_))] for x1_,y1_,x2_,y2_ in zip(x1,y1,x2,y2)]

        with open(os.path.join('groundtruths',str(frame_number) + '.txt'), 'w') as file:
            for point in points:
                
                # Turns out not all annotations are in the form [(x_min, y_min), (x_max, y_max)]
                # We make sure the above form holds true for all
                top_left = point[0]
                bottom_right = point[1]

                if top_left[0] > bottom_right[0]:
                    xmax, xmin = top_left[0] // factor, bottom_right[0] // factor
                else:
                    xmin, xmax = top_left[0] // factor, bottom_right[0] // factor

                if top_left[1] > bottom_right[1]:
                    ymax, ymin = top_left[1] // factor, bottom_right[1] // factor
                else:
                    ymin, ymax = top_left[1] // factor, bottom_right[1] // factor

                file.write("person ")
                file.write(str(xmin) + " ")
                file.write(str(ymin) + " ")
                file.write(str(xmax) + " ")
                file.write(str(ymax) + "\n")


        print('File:', frame_number, end = '\r')
