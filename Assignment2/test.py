from main import haar_cascades, yolov5
import os

file_names = map(lambda x: x[:-4], os.listdir("./data/ear_data/test/"))
file_names = list(dict.fromkeys(file_names))  # remove duplicates


def evaluate_haar():
    scale_factor = 1.15
    min_neighbors = 3
    min_size = (0, 0)

    miou = haar_cascades(file_names, 
                         scale_factor=scale_factor,
                         min_neighbors=min_neighbors,
                         min_size=min_size)
    print(f"{scale_factor}, {min_neighbors}, {min_size} -> {miou}")


def evaluate_yolov5():
    miou = yolov5(file_names)
    print(miou)


evaluate_haar()