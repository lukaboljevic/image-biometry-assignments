from main import haar_cascades, yolov5
import os

file_names = map(lambda x: x[:-4], os.listdir("./data/ear_data/test/"))
file_names = list(dict.fromkeys(file_names))  # remove duplicates


def evaluate_haar():
    scale_factors = [1.05, 1.10, 1.15, 1.20]
    min_neighbors = [1, 3, 5, 6, 10]
    min_sizes = [(30, 30), (70, 70), (100, 100)]

    remaining = total = len(scale_factors) * len(min_neighbors) * len(min_sizes)
    for sc in scale_factors:
        for mn in min_neighbors:
            for ms in min_sizes:
                miou = haar_cascades(file_names, 
                                     scale_factor=sc,
                                     min_neighbors=mn,
                                     min_size=ms)
                remaining -= 1
                print(f"Remaining models: {remaining:2} / {total}")


def evaluate_yolov5():
    miou = yolov5(file_names)
    print(miou)


evaluate_haar()