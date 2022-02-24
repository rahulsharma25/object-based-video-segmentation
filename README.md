# Object based Video Segmentation using Yolo model

> This repo contains two files. predict-yolov3.py uses the Yolo v3 pretrained model and predict-tiny-yolo.py uses Yolo-tiny pretrained model for inference generation.

Yolo v3 needs a gpu for running inference at minimum 40 FPS. Using CPU, it only gives a maxiimum of 2 FPS. But the model is very accurate.

Since I dont have CUDA configured currently, I have also provided the detector using YOLO-tiny model which is much smaller and less resorce intensive than original yolo v3. But due to smal size of the network, the detections are not very accurate. This can give upto 15 FPS on CPU.

So, there is a tradeoff between speed and accuracy. This can be resolved by using a GPU enabled implementaion of inference generation.

The model is pretrained on COCO dataset which has 80 diffent classes.
