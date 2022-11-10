# Assignment 2

## YOLOv5 results

To obtain the `yolo_detections` folder, so precision and recall can be calculated for different thresholds, the function `evaluate_yolov5`, from `test.py`, has to be ran, with `save` set to  `True`. Then, to calculate precision and recall values for a specific threshold, run the `calculate_yolo_pr` function, also found in `test.py`.


## Haar cascades results

To calculate the mean IoUs, precisions and recalls for different settings of the Haar cascades model, set the parameters you want in `calculate_haar_pr` function in `test.py`, and run it. The results will be saved inside the `results` folder.