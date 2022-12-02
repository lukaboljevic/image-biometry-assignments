# Image Based Biometry assignments


## Assignment 1

The required `awe` dataset: [download link](https://tinyurl.com/3ucw29ar). It should be placed under `Assignment1/`.

For implementing LBP, it is useful to read the original LBP paper (Ojala et. al.): [download link](https://dbox.si/index.php/s/Yqd3Y34q4wxDjNd)


## Assignment 2

Data, pretrained cascades for Haar-cascades classifier, and pretrained model for YOLOv5: [download link](https://tinyurl.com/ibba2). The files, as well as the extracted `ear_data.tar` archive, should be placed under `Assignment2/data/`:
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
because they are already a part of this repo's `requirements.txt` file. The command can of course be ran though, if needed. In whichever case, to load a custom model (like the one we're gonna use), we can look under "Custom Models" in the above linked PyTorch Hub documentation.


## Assignment 3

Data is available on [this link](https://tinyurl.com/ibb-a3data), under `images-zips` and `masks-zips`. Each student was assigned a subset of all the images. My assigned images (and masks) were **151-180**, **211-240**, **271-300** (90 subjects, 2539 images in total).

Downloaded images and masks should be placed under `Assignment3/images/` and `Assignment3/masks/` respectively. In other words, the structure of `Assignment3` folder should look something like

```
Assignment3
│   annotations.csv
|   annotator.py
|   errors.py
|   gimp.py
│
└───images
│   └───151
│       │   01.png
│       │   02.png
|       |   03.png
|       |   ...
|   
│   └───152
|       |   01.png
|       |   ..
|
└───masks
│   └───151
│       │   01.png
│       │   02.png
|       |   03.png
|       |   ...
|
│   └───152
|       |   01.png
|       |   ..
```

The purpose of `annotator.py` is to load images one by one in a `matplotlib` "UI", so the annotation process is easier. The annotations for each image are found in `annotations.csv`. It will load images starting after the last annotated one. So for example, if the last annotated image is from subject 151, image 02, it will continue from image 03 of subject 151.

File `errors.py` is a simple script that loads all incorrect images and masks. This is mostly for "double checking" if the assigned error is assigned for a reason, to put it what way.

`gimp.py` is a simple script that loads all incorrect images and masks in GIMP, so that they can be fixed. Note that the path to the GIMP executable may not be the same as the one that's there right now. This script, like `annotator.py`, loads incorrect images/masks starting from the last corrected ones.

Corrected images and masks are found under `Assignment3/corrected/`. I should have added the masks to a zip, but it is what it is. The corrected images are all found inside `images.zip`.
