from main import haar_cascades, yolov5, yolo_pr
import os
import json

file_names = map(lambda x: x[:-4], os.listdir("./data/ear_data/test/"))
file_names = list(dict.fromkeys(file_names))  # remove duplicates
iou_thresholds = [0.2, 0.5, 0.7, 0.8, 0.9]


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
    scale_factor = 1.1
    min_neighbors = 3
    min_size = (5, 5)

    precisions, recalls, miou = haar_cascades(file_names, 
                                              iou_thresholds,
                                              scale_factor=scale_factor,
                                              min_neighbors=min_neighbors,
                                              min_size=min_size,
                                              save=True)

    print(scale_factor, min_neighbors, min_size)
    print(iou_thresholds)
    print(precisions)
    print(recalls)
    print(miou)


def evaluate_yolov5():
    miou = yolov5(file_names)
    print(miou)


def calculate_yolo_pr():
    folder = "./yolo_detections"
    file_names = os.listdir(folder)
    yolo_json = []
    for file_name in file_names:
        with open(f"{folder}/{file_name}") as f:
            yolo_json.append(json.load(f))

    precisions, recalls = [], []
    for t in iou_thresholds:
        pr, rec = yolo_pr(yolo_json, t)
        precisions.append(round(pr, 2))
        recalls.append(round(rec, 2))

    print(iou_thresholds)
    print(precisions)
    print(recalls)


calculate_haar_pr()