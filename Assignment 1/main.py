from scipy.spatial.distance import cdist
import numpy as np
from utils import generate_neighbor_steps, cyclic_shift_to_smallest, \
    inside, count_transitions, apply_histogram
from random import sample, seed


def pixels(images):
    """ Returns the pixels of all images """
    return list(img.getdata() for img in images)


def lbp(images, R=1, P=8, local_region_step=1,
        use_histograms=False, cyclic_shifting=False, uniform=False):
    """
    Implementation of LBP (Local Binary Patterns).

    ==========
    Parameters
    ==========

    R: int (=1 by default)
        Radius i.e. how far away from the current observed pixel
        do we take neighbors
    P: int (=8 by default)
        The number of pixels we take around the center;
        Default value 8, which translates to 8*R in the
        default case, otherwise it's between 1 and 8*R
    local_region_step: int (=1 by default)
        Defines how many pixels to the right we move after
        processing a pixel; default value is 1, maximum value
        is 2*R + 1. If it's 2*R + 1, then that means there is
        no local region overlap
    use_histograms: bool (=False by default)
        Whether to use histograms as the feature vector or not
    cyclic_shifting: bool (=False by default)
        Whether to use cyclic shifting (to achieve rotational
        invariance) or not
    uniform: bool (=False by default)
        Whether we are using uniform LBP or not. Notice that the
        cyclic_shifting parameter is irrelevant if uniform is
        set to True.
    """
    vectors = []
    neighbor_steps = generate_neighbor_steps(R)
    width, height = images[0].size
    bins_size = (P + 1) if uniform else 2 ** P
    seed(0)  # for (potentially) sampling neighbors

    # I will assume parameters are passed within their correct intervals,
    # to not complicate unnecessarily.

    # If P = 8*R, then we take all 8*R pixels around the center i.e.
    # we do not touch the neighbor_indices list.
    # Otherwise, if it's > 0 (and < 8*R), then it determines the length of 
    # the pattern (duh). In this case, it's basically the sample size for the
    # neighborhood.
    if P == 8 * R:
        pass
    else:
        # I'm sampling indices, rather than actual elements, so the
        # operation of collecting neighbors i.e. building a binary number,
        # starting to the right of the center pixel, in a counter-clockwise
        # notion, is preserved

        # I'm sampling randomly for simplicity
        length = len(neighbor_steps)
        indices = sorted(sample(range(length), P))
        neighbor_steps = [neighbor_steps[index] for index in indices]

    for img in images:
        all_pixels = img.load()
        converted_image = np.zeros((height, width), dtype=np.int16)

        for x in range(0, height):
            for y in range(0, width, local_region_step):
                # Calculate the local binary pattern for this pixel
                binary_pattern = ""
                for move in neighbor_steps:
                    new_x = x + move[0]
                    new_y = y + move[1]
                    if not inside(new_x, new_y, width, height):
                        continue

                    binary_pattern += "1" if all_pixels[new_x,
                                                        new_y] >= all_pixels[x, y] else "0"

                if uniform:
                    # Count 0-1 and 1-0 transitions
                    transitions = count_transitions(binary_pattern)
                    if transitions <= 2:
                        num = binary_pattern.count("1")
                    else:
                        num = P + 1
                else:
                    if cyclic_shifting:
                        num = cyclic_shift_to_smallest(binary_pattern)
                    else:
                        num = int(binary_pattern, 2)

                converted_image[x][y] = num

        if use_histograms:
            # bins have to be explicitly stated, otherwise doesn't make sense
            vectors.append(apply_histogram(converted_image, (16, 16), bins_size))
        else:
            vectors.append(converted_image.flatten())

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
