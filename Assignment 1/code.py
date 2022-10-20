from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
import numpy as np
from utils import generate_neighbor_indices, cyclic_shift_to_smallest, read_dataset, inside


def pixels(images):
    """ Returns the pixels of all images """
    return list(img.getdata() for img in images)


def lbp(images, R=1, hist=True, uniform=False, sample_neighbors=-1, region_overlap=-1):
    """
    LBP ... 
    """
    vectors = []
    neighbor_indices = generate_neighbor_indices(R)
    width, height = images[0].size

    if sample_neighbors == -1:
        P = 8 * R  # simplify for now
    else:
        pass

    if region_overlap == -1:
        # 2*R + 1 for no region overlaps
        step = 2*R + 1
    else:
        step = 2*R + 1  # TODO: CHANGE!

    for img in images:
        all_pixels = img.load()
        converted_pixels = []

        for x in range(R, width-R):
            for y in range(R, height-R, step):
        # for x in range(0, width):
        #     for y in range(0, height, step):
                # Calculate the local binary pattern for this pixel
                binary_pattern = ""
                for move in neighbor_indices:
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

    for i in range(len(distances_pairwise)):
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


def compare_with_scikit(distance_metric, hist=True):
    """ Compare my implementation of LBP with Scikit's """
    R = 1
    P = 8 * R
    images, classes = read_dataset((128, 128))

    # Scikit LBP
    vectors = []

    for img in images:
        converted_image = local_binary_pattern(img, P, R, method="default")
        if hist:
            hist = np.histogram(converted_image.ravel(), bins=range(256))[0]
            vectors.append(hist)
        else:
            vectors.append(converted_image.flatten())

    scikit_rank1 = calculate_rank1_accuracy(vectors, classes, distance_metric)

    # My implementation
    vectors = lbp(images, hist=hist)
    my_rank1 = calculate_rank1_accuracy(vectors, classes, distance_metric)

    print("Scikit implementation: ", scikit_rank1)
    print("My implementation: ", my_rank1)


# compare_with_scikit("euclidean")
images, classes = read_dataset((128, 128))
vectors = lbp(images, hist=False)
print(calculate_rank1_accuracy(vectors, classes, "euclidean"))
