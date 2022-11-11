from main import haar_cascades, yolov5, yolo_pr
import os
import json
import numpy as np


# Get which file names are in our dataset
file_names = map(lambda x: x[:-4], os.listdir("./data/ear_data/test/"))

# Remove duplicates
file_names = list(dict.fromkeys(file_names))

# Two IoU thresholds lists
iou_thresholds = [0.2, 0.5, 0.7, 0.8, 0.9]
iou_thresholds_all = np.arange(0, 1, 0.01)


def evaluate_haar():
    """
    This function does not need to be ran anymore, it was just used
    to establish if there was any sort of connection between the resulting
    values of mIoU, precision, and recall
    """
    scale_factors = [1.05, 1.10, 1.15, 1.20]
    min_neighbors = [1, 3, 5, 6, 10]
    min_sizes = [(30, 30), (70, 70), (100, 100)]

    remaining = total = len(scale_factors) * len(min_neighbors) * len(min_sizes)
    for sc in scale_factors:
        for mn in min_neighbors:
            for ms in min_sizes:
                haar_cascades(file_names,
                              iou_thresholds,
                              scale_factor=sc,
                              min_neighbors=mn,
                              min_size=ms,
                              save=True)
                remaining -= 1
                print(f"Remaining models: {remaining:2} / {total}")


def calculate_haar_pr():
    scale_factor  = [1.05,     1.1,    1.1,      1.2       ]
    min_neighbors = [1,        3,      6,        3         ]
    min_size      = [(30, 30), (5, 5), (30, 30), (100, 100)]

    for i in range(len(min_size)):
        precisions, recalls, miou = haar_cascades(file_names, 
                                                  iou_thresholds_all,
                                                  scale_factor=scale_factor[i],
                                                  min_neighbors=min_neighbors[i],
                                                  min_size=min_size[i])

        # Only run the bottom code if average precision and recall need
        # to be calculated, i.e. if haar_cascades was called with the
        # iou_thresholds_all list
        avg_p = round(sum(precisions) / len(precisions), 3)
        avg_r = round(sum(recalls) / len(recalls), 3)

        f = f"{scale_factor[i]:.2f}"
        n = f"{min_neighbors[i]:02}"
        s = "x".join(str(x) for x in min_size[i])
        name = f"./results/APR-haar-{f}-{n}-{s}.txt"
        with open(name, "w") as f:
            f.write(f"Thresholds: 0:0.01:1, avg precision: {avg_p}, avg recall: {avg_r}\n")


def evaluate_yolov5():
    miou = yolov5(file_names)
    name = "./results/mIoU-yolo-default.txt"
    with open(name, "w") as f:
        f.write(f"{miou}\n")


def calculate_yolo_pr():
    # Read IoU and confidence for each detected box, detected by YOLOv5
    folder = "./yolo_detections"
    file_names = os.listdir(folder)
    yolo_json = []
    for file_name in file_names:
        with open(f"{folder}/{file_name}") as f:
            yolo_json.append(json.load(f))

    # Calculate precision and recall values for some IoU thresholds
    name = "PR-yolo-default.txt"
    precisions, recalls = [], []
    for t in iou_thresholds:
        prec, rec = yolo_pr(yolo_json, t)
        prec = round(prec, 3)
        rec = round(rec, 3)
        precisions.append(prec)
        recalls.append(prec)

    with open(f"./results/{name}", "w") as f:
        for i, t in enumerate(iou_thresholds):
            f.write(f"Threshold: {t:.2f}, precision: {precisions[i]}, recall: {recalls[i]}\n")

    # Calculate "average precision" and "average recall" for all IoU thresholds
    name = "A" + name
    precisions, recalls = [], []
    for t in iou_thresholds_all:
        prec, rec = yolo_pr(yolo_json, t)
        precisions.append(prec)
        recalls.append(rec)

    avg_p = round(sum(precisions) / len(precisions), 3)
    avg_r = round(sum(recalls) / len(recalls), 3)
    with open(f"./results/{name}", "w") as f:
        f.write(f"Thresholds: 0:0.01:1, avg precision: {avg_p}, avg recall: {avg_r}\n")


# calculate_haar_pr()
# evaluate_yolov5()
calculate_yolo_pr()