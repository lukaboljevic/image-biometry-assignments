import cv2 as cv
from my_utils import IoU
import torch


def haar_cascades(file_names, scale_factor=1.05, min_neighbors=0, min_size=(30, 30)):
    """
    Detect ears on images using pre-trained Haar-cascade classifiers. 
    Calculate the mean Intersection over Union, using the supplied 
    ground truth bounding box for each image.
    """
    # Load the detectors
    left_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_leftear.xml")
    right_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_rightear.xml")

    total_iou = 0
    num_dets = 0  # we will calculate mean IoU over all detected boxes
    remaining = len(file_names)
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

        # Calculate IoU for all detected boxes
        ious = []
        if len(ear_boxes) >= 1:
            for i in range(len(ear_boxes)):
                res = IoU([
                    ear_boxes[i][0], ear_boxes[i][1],  # x, y of top left corner
                    ear_boxes[i][0] + ear_boxes[i][2],  # x + width i.e. x of bottom right
                    ear_boxes[i][1] + ear_boxes[i][3]  # y + height i.e. y of bottom right
                ], gt_box)
                ious.append(res)
            increment = len(ious)
        else:
            # There was no ears detected, but still account for this
            increment = 1

        total_iou += sum(ious)
        num_dets += increment

        remaining -= 1
        print(f"Haar-cascades - remaining {remaining} / {len(file_names)}")

    # Save results to a file
    temp = ",".join(str(x) for x in min_size)
    with open(f"./results/haar-{str(scale_factor)}-{str(min_neighbors)}-{temp}.txt", "w") as f:
        f.write(f"{total_iou / num_dets}\n")

    # Average IoU over all detected boxes
    return total_iou / num_dets


def yolov5(file_names):
    """
    Detect ears on images using a pretrained YOLOv5 model. 
    Calculate the mean Intersection over Union, using the 
    supplied ground truth bounding box for each image.
    """
    # Load the local, pretrained YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path="./data/yolo5s.pt")
    print()

    total_iou = 0
    num_dets = 0  # we will calculate mean IoU over all detected boxes 
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
        # boxes, confidence, and class. We only need the coordinates
        # of the bounding boxes.
        df = results.pandas().xyxy[0]
        
        # Calculate IoU for all detected boxes
        ious = []
        if len(df) >= 1:
            for i in range(len(df)):
                xmin = int(df.at[i, "xmin"])  # x of top left corner
                ymin = int(df.at[i, "ymin"])  # y of top left corner
                xmax = int(df.at[i, "xmax"])  # x of bottom right corner
                ymax = int(df.at[i, "ymax"])  # y of bottom right corner
                res = IoU([xmin, ymin, xmax, ymax], gt_box)
                ious.append(res)
            increment = len(ious)
        else:
            # There was no ears detected, but still account for this
            increment = 1
        
        total_iou += sum(ious)
        num_dets += increment

        remaining -= 1
        print(f"YOLOv5 - remaining {remaining} / {len(file_names)}")
    
    # Average IoU over all detected boxes
    return total_iou / num_dets
