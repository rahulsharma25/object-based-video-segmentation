
# This program uses CPU for generating inferences. But as the Yolo architecture
# is very complex, it takes a lot of time for generating inference over CPU.
# The current FPS is about 2 frames per second.
# The FPS can be improved greatly by using CUDA enabled implementation of opencv library.
# This way the FPS can be easily increased to over 40 FPS.

# I have used CPU because I currently don't have CUDA enabled on my local machine.

import cv2 as cv 
import numpy as np
import time

if __name__ == "__main__":

    # Initialize required parameters for yolo model
    INPUT_SIZE = (416, 416)
    SCALING_FACTOR = 1/255
    MEAN_SUBTRACTION = (0, 0, 0)
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    cp = cv.VideoCapture('test.mp4')
    ORIGINAL_FPS = int(cp.get(cv.CAP_PROP_FPS))

    # Load the yolo model using pretrained weights and model configurations.
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')


    # Get those layers which are giving the output.
    # In yolo v3 model, there are 3 such output layers. 
    # These 3 output layers help in easily detecting objects of small and large sizes.
    outputLayers = model.getUnconnectedOutLayersNames()
    

    # I am using the yolo version 3 model trained on COCO dataset.
    # It contains 80 classes.
    classes = []
    with open('coco_names.txt', 'r') as name_file:
        classes = name_file.read().splitlines()
    

    # Classifying the objects into various categories to segment the video based
    # on the class which is detected with maximum confidence.
    none = ['none', 'None']
    birds_and_animals = ['birds_and_animals', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    people = ['people', 'person']
    vehicles = ['vehicles', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat']
    road_objects = ['road_objects', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench']
    toys_and_sports = ['toys_and_sports', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket']
    apparels = ['apparels', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase']
    dining = ['dining', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl']
    eatables = ['eatables', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']
    furniture = ['furniture', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable']
    electronics = ['electronics', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'refrigerator', 'hair drier']
    home_objects = ['home_objects', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'toothbrush']

    global_ls = [none, birds_and_animals, people, vehicles, road_objects, toys_and_sports, apparels, dining, eatables, furniture, electronics, home_objects]


    # Initialize a VideoCapture object to capture frames of the video.
    cap = cv.VideoCapture('test.mp4')

    # A list to record various raw timstamps with most confident category of that time period
    timestamps_with_categories = []
    prev_category = "none"
    initial_time = time.time()

    p_time = time.time()
    c_time = 0
    while True:
        imTrue, img = cap.read()
        if not imTrue:
            break

        img_height, img_width = img.shape[:2]


        # Change the image to a format which is required as yolo model input.
        # blob is a (1, 416, 416, 3) shape tensor.
        # 1 is the number of images (since we are passing only one image at a time)
        # 416 x 416 is the input shape
        # 3 is the number of chanels
        blob = cv.dnn.blobFromImage(img, SCALING_FACTOR, INPUT_SIZE, MEAN_SUBTRACTION, swapRB=True, crop=False)
        
        
        # Provide input and give it a forward pass through the pretrained yolo model.
        # all_outputs contains the output from all 3 output layers of the model.
        model.setInput(blob)
        all_outputs = model.forward(outputLayers)


        # Store the required information to detect an object in these Lists.
        # bboxes contain the co-ordinates of key points of bounding box.
        # class_ids contains the id of the classs which are detected for various objects in the image. 
        # class_confidence contains the confidence value with which the model has detected a class.
        bboxes = []
        class_ids = []
        class_confidences = []

        for output in all_outputs:
            for detection in output:
                box_score = detection[4]
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)

                if class_scores[class_id] > CONFIDENCE_THRESHOLD:
                    cx = int(detection[0] * img_width)
                    cy = int(detection[1] * img_height)
                    box_width = int(detection[2] * img_width)
                    box_height = int(detection[3] * img_height)

                    x = int(cx - box_width/2)
                    y = int(cy - box_height/2)

                    bboxes.append([x, y, box_width, box_height])
                    class_ids.append(class_id)
                    class_confidences.append(class_scores[class_id])


        # Since the yolo model can give multiple bounding boxes for the same object, 
        # we apply non-max suppression on the bounding boxes and keep only those boxes
        # whose IOU (Intersection over Union) value is less than a certain threshold.
        # IOU is a measure of overlap of bounding boxes on each other.
        final_indices = cv.dnn.NMSBoxes(bboxes, class_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)


        # variable to keep track of the class which has the maximum detection confidence.
        most_confident_class = -1
        if len(final_indices) != 0:
            most_confident_class = final_indices[0]

        for i in final_indices:
            box = bboxes[i]
            object_class = classes[class_ids[i]]

            if class_confidences[i] > class_confidences[most_confident_class]:
                most_confident_class = i

            confidence = class_confidences[i] * 100
            confidence = round(confidence, 1)

            x, y, w, h = box[:]
            
            # Draw the bounding boxes on frames.
            cv.putText(img, f'{object_class} {confidence}', (x, y-5), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)            
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

        most_confident_obj = "None"
        if most_confident_class != -1:
            most_confident_obj = classes[class_ids[most_confident_class]]
        
        curr_category = "none"
        for ls in global_ls:
            if most_confident_obj in ls:
                curr_category = ls[0]
                break

        # Calculating the FPS
        c_time = time.time()
        fps = int(1/(c_time - p_time))
        p_time = c_time
        cv.putText(img, str(fps), (20, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Update the timstamps if main category in video changes
        if curr_category != prev_category:
            # Since inferencing every frame reduces the Frame rate, the time passed during this program is
            # more than the duration of the video. So we have to map the timestamps recorded here to the original 
            # timestamps in the video. This can be done by using this formula: (c_time-initial_time)*(fps/ORIGINAL_FPS)
            timestamps_with_categories.append([(c_time-initial_time)*(fps/ORIGINAL_FPS), prev_category])
            prev_category = curr_category   

        cv.imshow("video", img)

        if cv.waitKey(1) == ord('q'):
            break
    timestamps_with_categories.append([(c_time-initial_time)*(fps/ORIGINAL_FPS), prev_category])
    
    # Since the confidence is not 100% accurate, there can be irrelevant fluctuations
    # in objects detected in the frames. filtered_timestamps stores filters those and
    # store only relevant timestamps.
    filtered_timestamps = []
    prev_ts, prev_categ = 0, "none"

    for ts, categ in timestamps_with_categories:
        if int(ts - prev_ts) >= 1:
            # If previous filtered timestamp category is same as current category, concatenate the timestamps
            if len(filtered_timestamps) != 0 and filtered_timestamps[-1][1] == categ:
                filtered_timestamps[-1][0] = int(ts)
            else:
                filtered_timestamps.append([int(ts), categ])
        prev_ts = ts
        prev_categ = categ


    print("Video Segments:")
    pt = 0
    for timestamp, category in filtered_timestamps:
        if category != "none":
            print(f'From {pt} to {timestamp} seconds: {category}')
        pt = timestamp

    cap.release()
    cv.destroyAllWindows()