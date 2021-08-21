---
layout: post
title:  "Project Overview  and Commits"
date:   2021-08-18 02:30:50 +0530
categories: jekyll update
---

## What was done

I worked on building a Deep Learning Human Detection Exercise to identify the presence of humans and identification of the rectangular boundary around them. Apart from the live and video inference features, the exercise also includes model benchmarking and model visualization. The user is expected to upload a Deep Learning model which fits the required input and output specifications for inference. The input model is supposed to be in the ONNX format. <br/>
The project work also included compiling and testing an exercise guide for the user, which contains everything from fine tuning pre built object detection models in different frameworks(PyTorch and TensorFlow), to its subsequent conversion to the ONNX format. <br/> <br/>
Below is a very brief timeline of the work done. For more information and details, please refer to the specific week in the blog post. <br/>

| Week      | Work Done |
| ----------- | ----------- |
| **Community Bonding Period**      | Hands on ONNX framework, Hands on AILIA-SDK, Study on object detection, Brushing up with the required tech stack, Fixing minor issues in existing DL exercises     |
| **Week 1**   | Building basic DL Human Detection exercise(Choosing pre-built models, Exporting the model to ONNX format, Preprocessing the input, Running inference, Postprocessing the outputs)        |
| **Week 2**      | Added model analyzer, Started working on model benchmarking, Collecting benchmarking dataset, Preprocessing the dataset       |
| **Week 3**   | Continuing with the benchmarking code, Benchmarking on the Oxford Town Center dataset, Added custom buttons and their functionalities for benchmarking and visualization        |
| **Week 4**      | Documented guides to train and fine tune pre-existing DL models in PyTorch and TensorFlow. This includes everything from making the process of collecting data, preprocessing it, fine tuning with the data on a pre-existing model architecture and converting the model to the ONNX format.  |
| **Week 5**  | Added video inference feature to the exercise, Made template changes to make exercise look minimalistic, Added benchmarking results and graph on the exercise console and canvas        |
| **Week 6**      | Implemented the upload of model/video an independent event from executing an exercise mode, Worked on the base Deep Learning server template for the exercise, Worked on the main exercise template used during the docker execution, Fixed exercise connection issue, Added the new noVNC console to the web template, Implemented more specific activation/deactivation of buttons based on wether the required files have been fully uploaded or not       |
| **Week 7**   | Resolved issue of process thread closing when launching different modes one after another, Resolved benchmarking issue occurring during the docker launch via start.sh, Resolved model visualizer connectivity issue occurring during the docker launch, Resolved output re-direction issue to the NoVnc console which arose after the one RADI launch per session update        |
| **Week 8**   | University Exam break        |
| **Week 9**   | Worked on GPU integration for the exercise(Dependencies included onnxruntime-gpu, CUDA runtime, CUDNN, and supported Nvidia drivers)       |
| **Week 10**   | Continued the work on GPU integration, Cleaning up the code base    |

## Working Demo

<iframe width="700" height="480"
src="https://www.youtube.com/embed/vn4ahq8mElg">
</iframe> 
<br/>

## To Do

* Providing support for GPU and CPU inference from a common docker image

## Merged PR's 

* [Updated launching procedure for Windows and a minor unnecessity in follow_line exercise #855](https://github.com/JdeRobot/RoboticsAcademy/pull/855)
* [Removed saveCode() and loadCode() functions for exercises running via the web template #879](https://github.com/JdeRobot/RoboticsAcademy/pull/879)
* [Made consistent frame rate policy and added FPS counter #981](https://github.com/JdeRobot/RoboticsAcademy/pull/981)
* [**Added Deep Learning Human Detection Exercise Version 1 #992**](https://github.com/JdeRobot/RoboticsAcademy/pull/992)
* [Updated web page for human detection exercise #1044](https://github.com/JdeRobot/RoboticsAcademy/pull/1044)
* [Added Fine-Tuning for PyTorch models](https://github.com/TheRoboticsClub/gsoc2021-Shashwat_Dalakoti/pull/11)
* [Updated Human Detection exercise docs #1123](https://github.com/JdeRobot/RoboticsAcademy/pull/1123)
* [**Updated Human Detection Exercise to Version 2 #1174**](https://github.com/JdeRobot/RoboticsAcademy/pull/1174)
* [Updated start_console() point in human_detection exercise #1193](https://github.com/JdeRobot/RoboticsAcademy/pull/1193)
* [Updated Human Detection exercise webpage #1201](https://github.com/JdeRobot/RoboticsAcademy/pull/1201)
* [Code clean-up #1230](https://github.com/JdeRobot/RoboticsAcademy/pull/1230)

## Extras

* [Added MNIST-PyTorch-To-ONNX #8](https://github.com/TheRoboticsClub/gsoc2021-Shashwat_Dalakoti/pull/8)
* [Added benchmarking data set and the code files to pre-process the data set. #9](https://github.com/TheRoboticsClub/gsoc2021-Shashwat_Dalakoti/pull/9)
