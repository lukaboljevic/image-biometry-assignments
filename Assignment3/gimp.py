import os
import subprocess
from errors import get_errors

def get_corrected():
    """
    Return the currently corrected masks (we always correct masks,
    but we don't always correct images, unless there were two ears
    present)
    """
    root = "./corrected/masks"
    res = []

    for d in os.listdir(root):
        for img in os.listdir(f"{root}/{d}"):
            res.append((d, img[:2]))  # remove the .png extension

    return res
    

def main():
    """
    Open an image and its corresponding mask in GIMP. Wait for
    any key to be pressed to load the next image and mask in line.
    """
    error_list = get_errors()
    corrected = get_corrected()
    gimp = "C:/Applications/Gimp 2/bin/gimp-2.10.exe"
    last_dir = corrected[-1][0]

    print("##########################")
    print(f"Corrected: {len(corrected)} / {len(error_list)}")
    print("##########################\n")

    for dir_name, img_name, error in error_list:
        if error == "wrong subject":
            # we don't need to do anything in this instance
            continue

        if (dir_name, img_name[1:]) in corrected:
            # don't open already corrected stuff
            continue

        subprocess.call([gimp, f"images/{dir_name}/{img_name[1:]}.png"], shell=True)
        subprocess.call([gimp, f"masks/{dir_name}/{img_name[1:]}.png"], shell=True)

        if dir_name != last_dir:
            # just to "group" the output by subject ID
            print()
            last_dir = dir_name
        input(f"{dir_name}, {img_name}, {error} | Press any key to load new images...")


if __name__ == "__main__":
    main()