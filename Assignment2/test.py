from main import haar_cascades, yolov5
import os

file_names = map(lambda x: x[:-4], os.listdir("./data/ear_data/test/"))
file_names = list(dict.fromkeys(file_names))  # remove duplicates


def evaluate_haar():
    left_miou, right_miou = haar_cascades(file_names)
    print(left_miou, right_miou)


def evaluate_yolov5():
    miou = yolov5(file_names)
    print(miou)


evaluate_yolov5()