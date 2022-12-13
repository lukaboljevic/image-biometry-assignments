# Assignment 2

Data, pretrained cascades for Haar-cascades classifier, and pretrained model for YOLOv5: [download link](https://tinyurl.com/ibba2). The files, as well as the extracted `ear_data.tar` archive, should be placed under `data/`:

```
./data
│   yolo5s.pt
│   haarcascade_mcs_leftear.xml
|   haarcascade_mcs_rightear.xml
│
└───ear_data
│   └───test
│       │   0501.png
│       │   0501.txt
│       │   0502.png
|       |   ...
```

The requirement for Assignment 2 is YOLOv5. The GitHub repo is available [here](https://github.com/ultralytics/yolov5). Since we are using PyTorch Hub, it is not necessary to clone the repository, as stated in the documentation for loading YOLOv5 using PyTorch Hub [here](https://github.com/ultralytics/yolov5/issues/36). It is **not** necessary to run their command to install requirements, namely:
```
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```
because they are already a part of this repo's `requirements.txt` file. The command can of course be ran though, if needed. In whichever case, to load a custom model (like the one we're gonna use), we can look under "Custom Models" in the above linked PyTorch Hub documentation.

# YOLOv5 results

To obtain the `yolo_detections` folder, so precision and recall can be calculated for different thresholds, the function `evaluate_yolov5`, from `test.py`, has to be ran, with `save` set to  `True`. Then, to calculate precision and recall values for a specific threshold, run the `calculate_yolo_pr` function, also found in `test.py`.


# Haar cascades results

To calculate the mean IoUs, precisions and recalls for different settings of the Haar cascades model, set the parameters you want in `calculate_haar_pr` function in `test.py`, and run it. The results will be saved inside the `results` folder.