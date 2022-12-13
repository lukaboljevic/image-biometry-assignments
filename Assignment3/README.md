# Assignment 3

Data is available on [this link](https://tinyurl.com/ibb-a3data), under `images-zips` and `masks-zips`. Each student was assigned a subset of all the images. My assigned images (and masks) were **151-180**, **211-240**, **271-300** (90 subjects, 2539 images in total).

Downloaded images and masks should be placed under `images/` and `masks/` respectively. In other words, the structure of this folder should look something like

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

Corrected images and masks are found under `corrected/`. I should have added the masks to a zip, but it is what it is. The corrected images are all found inside `images.zip`.