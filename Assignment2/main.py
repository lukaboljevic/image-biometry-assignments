import cv2 as cv
from my_utils import IoU
import torch
import json
import os


def haar_cascades(file_names, thresholds, scale_factor=1.05, 
                  min_neighbors=5, min_size=(30, 30), save=False):
    """
    Detect ears on images using pre-trained Haar-cascade classifiers. 
    Calculate the mean Intersection over Union, using the supplied 
    ground truth bounding box for each image.

    Also calculate the precision and recall for this model, for some
    given IoU thresholds.
    """
    # Load the detectors
    left_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_leftear.xml")
    right_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_rightear.xml")

    tps = [0] * len(thresholds)  # calculate tp for each threshold separately
    fps = [0] * len(thresholds)  # calculate fp for each threshold separately
    ious = []  # here we will store all calculated IoUs

    remaining = num_positive = len(file_names)  # number of positives for calculating recall
    for file_name in file_names:
        img = cv.imread(f"./data/ear_data/test/{file_name}.png") # numpy.ndarray
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_height, img_width = img.shape[0], img.shape[1]

        # Read the ground truth bounding box
        with open(f"./data/ear_data/test/{file_name}.txt", "r") as f:
            line = f.read()
            line = list(map(lambda x: float(x), line.split()))

            gt_width = int(line[3] * img_width)
            gt_height = int(line[4] * img_height)
            gt_x_c = int(line[1] * img_width)  # x of the center of the bounding box
            gt_y_c = int(line[2] * img_height)  # y of the center of the bounding box
            gt_x = gt_x_c - gt_width // 2  # x of top left corner of bounding box
            gt_y = gt_y_c - gt_height // 2  # y of top left corner of bounding box
            gt_box = [gt_x, gt_y, gt_x + gt_width, gt_y + gt_height]

        # Detectors return an array of arrays, namely 
        # [[topleft_x, topleft_y, width, height]]
        left_box = left_ear_detector.detectMultiScale(img,
                                                      scaleFactor=scale_factor,
                                                      minNeighbors=min_neighbors,
                                                      minSize=min_size)
        right_box = right_ear_detector.detectMultiScale(img, 
                                                        scaleFactor=scale_factor,
                                                        minNeighbors=min_neighbors,
                                                        minSize=min_size)

        # Merge detections
        ear_boxes = [e for e in left_box]
        for e in right_box:
            ear_boxes.append(e)

        # Calculate IoU for all detected boxes.
        img_ious = []
        for i in range(len(ear_boxes)):
            iou = IoU([
                ear_boxes[i][0], ear_boxes[i][1],  # x, y of top left corner
                ear_boxes[i][0] + ear_boxes[i][2],  # x + width i.e. x of bottom right
                ear_boxes[i][1] + ear_boxes[i][3]  # y + height i.e. y of bottom right
            ], gt_box)
            ious.append(iou)
            img_ious.append(iou)

        # Go over all thresholds, and for each, evaluate if this image is TP or FP.
        for i, threshold in enumerate(thresholds):
            tp_found = False  # there can only be one TP on a single image for a single threshold
            for iou in img_ious:
                if iou >= threshold:
                    tp_found = True
                    tps[i] += 1
                    break
            fps[i] += len(img_ious) - tp_found

        remaining -= 1
        print(f"Haar-cascades - remaining {remaining} / {len(file_names)}")

    # Average IoU over all detected boxes
    miou = sum(ious) / len(ious)

    # Calculate precisions and recalls for given thresholds
    precisions = [round(tps[i] / (tps[i] + fps[i]), 3) for i in range(len(thresholds))]
    recalls = [round(tps[i] / num_positive, 3) for i in range(len(thresholds))]

    # Save results to a file
    if save:
        # Do some formatting
        f = f"{scale_factor:.2f}"
        n = f"{min_neighbors:02}"
        s = "x".join(str(x) for x in min_size)
        name = f"mIoU-haar-{f}-{n}-{s}.txt"

        with open(f"./results/{name}", "w") as f:
            f.write(f"{miou}\n")

        name = "PR-" + name[5:]
        with open(f"./results/{name}", "w") as f:
            for i in range(len(thresholds)):
                f.write(f"Threshold: {thresholds[i]:.2f}, precision: {precisions[i]}, " + 
                        f"recall: {recalls[i]}\n")

    # Return precision list, recall list, mIoU
    return precisions, recalls, miou


def yolov5(file_names, save=False):
    """
    Detect ears on images using a pretrained YOLOv5 model. 
    Calculate the mean Intersection over Union, using the 
    supplied ground truth bounding box for each image.
    """
    # Load the local, pretrained YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path="./data/yolo5s.pt")
    print()

    # Make the folder where we save the IoU values for each image
    if not os.path.exists("./yolo_detections"):
        os.mkdir("./yolo_detections")

    ious = []  # here we will store all calculated IoUs
    remaining = len(file_names)
    for file_name in file_names:
        # Read image
        img_path = f"./data/ear_data/test/{file_name}.png"
        img_gt_box = f"./data/ear_data/test/{file_name}.txt"
        img = cv.imread(img_path)
        img_height, img_width = img.shape[0], img.shape[1]

        # The image has to be converted to RGB, since PyTorch models are
        # trained to work with RGB images.
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Read the ground truth bounding box
        with open(img_gt_box, "r") as f:
            line = f.read()
            line = list(map(lambda x: float(x), line.split()))

            gt_width = int(line[3] * img_width)
            gt_height = int(line[4] * img_height)
            gt_x_c = int(line[1] * img_width)  # x of the center of the bounding box
            gt_y_c = int(line[2] * img_height)  # y of the center of the bounding box
            gt_x = gt_x_c - gt_width // 2  # x of top left corner of bounding box
            gt_y = gt_y_c - gt_height // 2  # y of top left corner of bounding box
            gt_box = [gt_x, gt_y, gt_x + gt_width, gt_y + gt_height]
    
        # Detect the ear
        results = model(img)  

        # Get a pandas DF with information on the detected bounding 
        # boxes, confidence, and class.
        df = results.pandas().xyxy[0]
        
        # Calculate IoU for all detected boxes. Store IoU and corresponding 
        # confidence value for each detected bounding box in a JSON file,
        # so we can calculate precision, recall, (mAP) ... later.
        img_ious = []
        confidences = []
        for i in range(len(df)):
            xmin = int(df.at[i, "xmin"])  # x of top left corner
            ymin = int(df.at[i, "ymin"])  # y of top left corner
            xmax = int(df.at[i, "xmax"])  # x of bottom right corner
            ymax = int(df.at[i, "ymax"])  # y of bottom right corner

            iou = IoU([xmin, ymin, xmax, ymax], gt_box)
            img_ious.append(iou)
            ious.append(iou)

            confidence = df.at[i, "confidence"]
            confidences.append(confidence)
            
        # Save the info to a JSON file.
        if save:
            info = {
                "ious": img_ious,
                "confidences": confidences
            }
            with open(f"yolo_detections/{file_name}.json", "w") as f:
                json.dump(info, f)

        remaining -= 1
        print(f"YOLOv5 - remaining {remaining} / {len(file_names)}")
    
    # Average IoU over all detected boxes
    return sum(ious) / len(ious)


def yolo_pr(yolo_json, threshold):
    """
    Calculate the precision and recall score for different IoU
    thresholds for the YOLOv5 model, given a preloaded list
    of YOLOv5 detections from JSON files in ./yolo_detections
    """

    num_positive = len(yolo_json)
    tp = 0
    fp = 0

    # So, if IoU >= IoU threshold, then it's a TP, else it's a FP.
    # FN we don't have to explicitly calculate, since we just need it to
    # know the number of all positives (number of bounding boxes)
    for img in yolo_json:
        # img: { "ious": [...], "confidences": [...] }
        # There can also be only one TP, all the rest are FP.
        tp_found = False
        for iou in img["ious"]:
            if iou >= threshold:
                tp_found = True
                tp += 1
                break
        # If we found a TP, then there are len - 1 FPs.
        # If we didn't find a TP, then there are len FPs
        fp += len(img["ious"]) - tp_found

    return tp / (tp + fp), tp / num_positive
