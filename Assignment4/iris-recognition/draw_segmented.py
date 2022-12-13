import os
import cv2
from fnc.segment import segment
from fnc.extractFeature import eyelashes_thres


# Define image to be shown - only changing image_name is enough
use_multiprocess = True
casia      = "..\\CASIA"
image_name = "091_1_2.jpg"
subject    = image_name[0:3]
folder     = image_name[4]


def main():
    # Perform segmentation
    im_path = os.path.join(casia, subject, folder, image_name)
    im = cv2.imread(im_path, 0)
    ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)


    # Draw iris and pupil on the original image
    im = cv2.circle(im, (ciriris[1], ciriris[0]), ciriris[2], 255)
    im = cv2.circle(im, (cirpupil[1], cirpupil[0]), cirpupil[2], 255)


    # Show
    cv2.imshow("Original: " + image_name, im)
    cv2.imshow("Noise (black): " + image_name, imwithnoise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()