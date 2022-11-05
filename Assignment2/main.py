import cv2 as cv
from my_utils import IoU
import torch

left_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_leftear.xml")
right_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_rightear.xml")


def haar_cascades(file_names, draw_boxes = False):
    """
    Detect left and right ears on images using pre-trained 
    Haar-cascade classifiers. Calculate the mean Intersection 
    over Union, using the supplied ground truth bounding box
    for each image.
    """
    left_num_dets, left_iou_total = 0, 0
    right_num_dets, right_iou_total = 0, 0
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
        left_box = left_ear_detector.detectMultiScale(img) 
        right_box = right_ear_detector.detectMultiScale(img)
        left_iou, right_iou = -1, -1

        # Calculate IoU's
        if len(left_box):
            left_iou = IoU([
                left_box[0][0], left_box[0][1],  # x, y of top left corner
                left_box[0][0] + left_box[0][2],  # x + width i.e. x of bottom right
                left_box[0][1] + left_box[0][3]  # y + height i.e. y of bottom right
            ], gt_box)

        if len(right_box):
            right_iou = IoU([
                right_box[0][0], right_box[0][1],  # x, y of top left corner
                right_box[0][0] + right_box[0][2],  # x + width i.e. x of bottom right
                right_box[0][1] + right_box[0][3]  # y + height i.e. y of bottom right
            ], gt_box)

        # One of both detectors detected an ear (or something), so
        # count that towards the final mean IoU
        if left_iou >= 0:
            left_num_dets += 1
            left_iou_total += left_iou
        if right_iou >= 0:
            right_num_dets += 1
            right_iou_total += right_iou

        # Draw the image and bounding boxes if wanted
        if draw_boxes:
            # Predicted box for the left ear
            for x, y, w, h in left_box:
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

            # Predicted box for the right ear
            for x, y, w, h in right_box:
                img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

            # Ground truth box
            img = cv.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), 
                (0, 255, 0), thickness=2)

            # Scale so it fits the screen
            new_height = 700
            ratio = img_width / img_height
            img_scaled = cv.resize(img, (int(new_height * ratio), new_height))

            # Show
            cv.imshow("Detected ears", img_scaled)
            cv.waitKey(0)
            cv.destroyAllWindows()

        remaining -= 1
        print(f"Haar-cascades - remaining {remaining} / {len(file_names)}")

    # Return mean IoU separately for the left/right ear detector
    return left_iou_total / left_num_dets, right_iou_total / right_num_dets


def yolov5(file_names, draw_boxes = False):
    """
    Detect ears on images using a pretrained YOLOv5 model. 
    Calculate the mean Intersection over Union, using the 
    supplied ground truth bounding box for each image.
    """
    # Load the local, pretrained YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path="./data/yolo5s.pt")
    print()

    total_iou = 0
    remaining = len(file_names)
    for file_name in file_names:
        # Read image
        img_path = f"./data/ear_data/test/{file_name}.png"
        img_gt_box = f"./data/ear_data/test/{file_name}.txt"
        img_cv = cv.imread(img_path)
        # The image has to be converted to RGB, since PyTorch models are
        # trained to work with RGB images.
        img_cv = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
        img_height, img_width = img_cv.shape[0], img_cv.shape[1]

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
        results = model(img_cv)  

        # Get a pandas DF with information on the bounding box, confidence, 
        # and class. We only need the bounding box coordinates.
        df = results.pandas().xyxy[0]
        try:
            xmin = int(df.at[0, "xmin"])  # x of top left corner
            ymin = int(df.at[0, "ymin"])  # y of top left corner
            xmax = int(df.at[0, "xmax"])  # x of bottom right corner
            ymax = int(df.at[0, "ymax"])  # y of bottom right corner
            iou = IoU([xmin, ymin, xmax, ymax], gt_box)
        except KeyError:
            # The model didn't find any ears!
            iou = 0
        
        # Add current IoU to the total
        total_iou += iou

        # Draw the predicted and ground truth box if wanted
        if draw_boxes:
            # img_cv = cv.imread(img_path)
            # img_height, img_width = img_cv.shape[0], img_cv.shape[1]
            
            # Predicted
            img_cv = cv.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 
                thickness=2)
            # Ground truth box
            img = cv.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), 
                (0, 255, 0), thickness=2)

            # Scale so it fits the screen
            new_height = 700
            ratio = img_width / img_height
            img_scaled = cv.resize(img_cv, (int(new_height * ratio), new_height))

            # Show
            cv.imshow("Detected ears", img_scaled)
            cv.waitKey(0)
            cv.destroyAllWindows()

        remaining -= 1
        print(f"YOLOv5 - remaining {remaining} / {len(file_names)}")
    
    # Return mean IoU
    return total_iou / len(file_names)
