# Assignment 4

Dataset for this assignment is the CASIA (I think V1) iris image database, which can be downloaded from [here](https://www.dropbox.com/s/uc1fokmab5av91h/CASIA.zip?dl=0). It should be placed in the root of this folder.

The code was cloned from [this repo](https://github.com/Th3nn3ss/python-iris-recognition), and renamed to `iris-recognition`. Before trying to run any code found there, be sure to activate a separate virtual environment (I named it `iris-venv`), as the package versions for that code were outdated by some 2.5 years (at the time of writing this) - and I didn't want to mess around with updating all of them and fixing all errors that arise from that. Only thing I had to update were `Pillow` to 9.3.0 and `numpy` to 1.22 due to some security issues. Commands for enabling a new virtual env (assuming current directory is **this folder**):

```
cd iris-recognition
python -m venv .iris-venv
source ./.iris-venv/Scripts/activate  # or source ./.iris-venv/bin/activate if on Linux
pip install -r requirements.txt
```

Nothing regarding the pipeline was changed from the original, other than renaming and removing irrelevant files, changing the necessary paths, changing some variable names and std outputs, and changing the return value of the matching process, found in `iris-recognition/fnc/matching.py`. The return value is now a dictionary, which for each matching threshold contains a list of names to all templates the query template matched with.

Two scripts were also added, namely `draw_segmented.py`, and `unenrolled_verify.py`. The former just shows the segmented iris and pupil, as well as the noise mask for an input image. The latter tests pipeline performance on all unenrolled images, and reports rank-1 and rank-5 accuracies for various thresholds. The results of those tests are found under `iris-recognition/accuracies`. The prefix `testOn1` means that the testing was done on unenrolled images that match the path `CASIA/*/1/*.png`, while the enrolled images were those that match path `CASIA/*/2/*.png`. Similar applies for prefix `testOn2`. Prior to running `unenrolled_verify.py`, `enroll_casia.py` needs to be ran.

