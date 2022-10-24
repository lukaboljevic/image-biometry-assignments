from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
import numpy as np
from utils import generate_neighbor_steps, cyclic_shift_to_smallest, read_dataset, inside
from random import sample


def pixels(images):
    """ Returns the pixels of all images """
    return list(img.getdata() for img in images)


def lbp(images, R=1, pattern_length=0, hist=False, uniform=False, local_region_step=1):
    """
    LBP ... 
    """
    vectors = []
    neighbor_steps = generate_neighbor_steps(R)
    width, height = images[0].size

    if pattern_length > 8*R:
        raise ValueError("Binary pattern length too large - cannot be more than 8*R")

    # If P = pattern_length = 0, then we take all 8*R pixels around the center i.e.
    # P = 8*R and we do not touch the neighbor_indices list.
    # Otherwise, if it's > 0, then it determines the length of the pattern (duh). 
    # If it's < 8*R, then it's basically the sample size for the neighborhood.
    # We can of course sample between 1 and 8*R neighbors.
    if pattern_length > 0:
        # I'm sampling indices, rather than actual elements, so the 
        # operation of collecting neighbors i.e. building a binary number,
        # starting to the right of the center pixel, in a counter-clockwise 
        # notion, is preserved
        l = len(neighbor_steps)
        indices = sorted(sample(range(l), pattern_length))
        neighbor_steps = [neighbor_steps[index] for index in indices]

    # Validate the value for the step size
    if local_region_step < 1 or local_region_step > 2*R + 1:
        raise ValueError("Step size for local regions invalid - must be between 1 and 2*R + 1.")

    for img in images:
        all_pixels = img.load()
        converted_pixels = []

        for x in range(R, width-R):
            for y in range(R, height-R, local_region_step):
        # for x in range(0, width):
        #     for y in range(0, height, local_region_step):
                # Calculate the local binary pattern for this pixel
                binary_pattern = ""
                for move in neighbor_steps:
                    new_x = x + move[0]
                    new_y = y + move[1]
                    # if not inside(new_x, new_y, width, height):
                    #     continue
                    
                    binary_pattern += "1" if all_pixels[new_x,
                                                        new_y] >= all_pixels[x, y] else "0"

                # num = cyclic_shift_to_smallest(binary_pattern)
                num = int(binary_pattern, 2)
                converted_pixels.append(num)

        if hist:
            # bins have to be explicitly stated, otherwise doesn't make sense
            # np.histogram(...)[0] is the actual histogram, [1] are bins
            vectors.append(np.histogram(converted_pixels, bins=range(256))[0])
        else:
            vectors.append(converted_pixels)

    return vectors


def calculate_rank1_accuracy(vectors, classes, distance_metric):
    """
    Calculate rank-1 recognition rate/accuracy for the given
    feature vectors, and using the given distance metric.
    """
    total = len(vectors)
    correct = 0
    distances_pairwise = cdist(vectors, vectors, distance_metric)

    for i in range(total):
        closest_value = 10000000
        closest_index = -1
        for j in range(len(distances_pairwise[0])):
            if j == i:
                continue

            if distances_pairwise[i][j] < closest_value:
                closest_value = distances_pairwise[i][j]
                closest_index = j

        correct += (classes[i] == classes[closest_index])

    return correct / total


def compare_with_scikit(image_size, distance_metric, hist=True):
    """ Compare my implementation of LBP with Scikit's """
    R = 1
    P = 8 * R
    images, classes = read_dataset(image_size)

    # Scikit LBP
    vectors = []

    for img in images:
        converted_image = local_binary_pattern(img, P, R, method="default")
        if hist:
            img_hist = np.histogram(converted_image.flatten(), bins=range(256))[0]
            vectors.append(img_hist)
        else:
            vectors.append(converted_image.flatten())

    scikit_rank1 = calculate_rank1_accuracy(vectors, classes, distance_metric)

    # My implementation
    vectors = lbp(images, hist=hist)
    my_rank1 = calculate_rank1_accuracy(vectors, classes, distance_metric)

    print("Scikit implementation: ", scikit_rank1)
    print("My implementation: ", my_rank1)


image_size = (128, 128)

images, classes = read_dataset(image_size)
vectors = lbp(images)
print(calculate_rank1_accuracy(vectors, classes, "euclidean"))

# compare_with_scikit("euclidean")