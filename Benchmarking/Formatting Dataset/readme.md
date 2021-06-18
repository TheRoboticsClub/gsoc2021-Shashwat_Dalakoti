### This folder contains all the required code to preprocess the dataset(vidoes and annotations) to the desired form used by the benchmarking code.

`**extract_towncentre.py**` - Converts the town centre video to the specified range of frames. Stores them in `frames` folder.

`**extract_GT_txt.py**` - Converts the ground truth annotations from the `TownCentre-groundtruth.top` file and stores all detections per frame into a seperate txt file inside the `groundtruths` folder. The format followed for storing each ground truth value is <class_name> <x_min> <y_min> <x_max> <y_max>.

`**extract_GT.py**`  - Converts the ground truth annotations from the `TownCentre-groundtruth.top` file and stores all detections per frame into a seperate xml file inside the `xmls` folder.
