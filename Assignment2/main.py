import cv2 as cv
from utils import IoU
import os

left_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_leftear.xml")
right_ear_detector = cv.CascadeClassifier("./data/haarcascade_mcs_rightear.xml")

def haar_cascades(file_names, scale = False, draw_boxes = False):
    """
    Detect left and right ears on images using pre-trained 
    Haar-cascade classifiers. Calculate the Intersection over
    Union using the supplied ground truth bounding box.
    """
    left_num_dets, left_iou_total = 0, 0
    right_num_dets, right_iou_total = 0, 0

    for file_name in file_names:
        img = cv.imread(f"./data/ear_data/test/{file_name}.png") # numpy.ndarray
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_height, img_width = img.shape[0], img.shape[1]

        # Scale the image (and keep the aspect ratio): ear detection is thus
        # faster, but the resulting IoU is a bit smaller (between approx 
        # 0.01 and 0.04)
        if scale:
            new_height = 600
            ratio = img_width / img_height
            img = cv.resize(img, (int(new_height * ratio), new_height))
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
            for x, y, w, h in left_box:
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

            for x, y, w, h in right_box:
                img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

            img = cv.rectangle(img, (gt_x, gt_y), (gt_x + gt_width, gt_y + gt_height), 
                (0, 255, 0), thickness=2)

            # Scale so it fits the screen
            new_height = 700
            ratio = img_width / img_height
            img_scaled = cv.resize(img, (int(new_height * ratio), new_height))

            cv.imshow("Detected ears", img_scaled)
            cv.waitKey(0)
            cv.destroyAllWindows()

    return left_iou_total / left_num_dets, right_iou_total / right_num_dets

file_names = map(lambda x: x[:-4], os.listdir("./data/ear_data/test/"))
file_names = list(dict.fromkeys(file_names))  # remove duplicates

left_miou, right_miou = haar_cascades(file_names)
print(left_miou, right_miou)
    