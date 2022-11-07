# Image Based Biometry assignments


## Assignment 1

`awe` dataset: [download link](https://tinyurl.com/3ucw29ar)
TODO change link to dbox.si

Original LBP paper (Ojala et. al.): [download link](https://dbox.si/index.php/s/Yqd3Y34q4wxDjNd)


## Assignment 2

Data, pretrained cascades for Haar-cascades classifier, and pretrained model for YOLOv5: [download link](https://tinyurl.com/ibba2). The files, as well as the extracted `ear_data.tar` archive, should be placed under `Assignment2/data/`. So, the `Assignment2/data/` folder should look like:
```
data
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
because they are already a part of this repo's `requirements.txt` file. The command can of course be ran though, if needed of course. In whichever case, to load a custom model (like the one we're gonna use), we can look under "Custom Models" in the above linked PyTorch Hub documentation.

