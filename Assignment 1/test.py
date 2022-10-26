from utils import read_dataset, apply_histogram
from main import lbp, calculate_rank1_accuracy, pixels
from skimage.feature import local_binary_pattern


image_size = (128, 128)
images, classes = read_dataset(image_size)


# Please ignore the stupidity of the below for loops,
# I am fully aware how dumb it looks.

#######################################
# Non uniform LBP with various params #
#######################################

def test_non_uniform():
    region_step = 1
    Rs = [1, 2]
    use_hists = [False, True]
    cyclic_shiftings = [False, True]
    distance_metrics = ["euclidean", "cosine", "cityblock"]
    file_name = "LBP_128x128_rs1.txt"

    for distance_metric in distance_metrics:
        with open(file_name, "a") as f:
            f.write("\n")
            f.write("================================\n")
            f.write(f"{distance_metric} distance\n")
            f.write("================================\n")
            f.write("\n")

        for R in Rs:
            Ps = [4*R, 8*R]
            for P in Ps:
                for use_hist in use_hists:
                    for cyclic_shifting in cyclic_shiftings:
                        vectors = lbp(images, R=R, P=P,
                                    local_region_step=region_step,
                                    use_histograms=use_hist,
                                    cyclic_shifting=cyclic_shifting)
                        print("LBP done")
                        result = calculate_rank1_accuracy(
                            vectors, classes, distance_metric)
                        print("Rank-1 accuracy calculated\n")

                        with open(file_name, "a") as f:
                            f.write(f"R={R}, P={P}, hist={use_hist}, " +
                                    f"cyclic={cyclic_shifting}\n")
                            f.write(f"\t{result}\n")
                            f.write("\n")


###################################
# Uniform LBP with various params #
###################################

def test_uniform():
    region_step = 1
    Rs = [1, 2]
    use_hists = [False, True]
    distance_metrics = ["euclidean", "cosine", "cityblock"]
    file_name = "LBP_128x128_rs1.txt"

    for distance_metric in distance_metrics:
        with open(file_name, "a") as f:
            f.write("\n")
            f.write("================================\n")
            f.write(f"{distance_metric} distance\n")
            f.write("================================\n")
            f.write("\n")
        for R in Rs:
            Ps = [4*R, 8*R]
            for P in Ps:
                for use_hist in use_hists:
                    vectors = lbp(images, R=R, P=P,
                                local_region_step=region_step,
                                use_histograms=use_hist,
                                uniform=True)
                    print("LBP done")
                    result = calculate_rank1_accuracy(
                        vectors, classes, distance_metric)
                    print("Rank-1 accuracy calculated\n")

                    with open(file_name, "a") as f:
                        f.write(f"R={R}, P={P}, hist={use_hist}\n")
                        f.write(f"\t{result}\n")
                        f.write("\n")


####################################
# Testing with "static" parameters #
####################################

def basic_LBP_testing():
    R = 1
    P = 8 * R
    region_step = 1
    use_hist = True
    cyclic_shift = False
    uniform = False
    distance_metric = "cityblock"

    vectors = lbp(images, R=R, P=P,
                  local_region_step=region_step,
                  use_histograms=use_hist,
                  uniform=uniform,
                  cyclic_shifting=cyclic_shift)
    print("LBP done")

    result = calculate_rank1_accuracy(vectors, classes, distance_metric)
    print(result)


########################################
# Testing the pixels feature extractor #
########################################

def test_pixels():
    distance_metrics = ["euclidean", "cosine", "cityblock", "hamming"]
    vectors = pixels(images)

    for distance_metric in distance_metrics:
        result = calculate_rank1_accuracy(vectors, classes, distance_metric)
        with open("Pixels-128x128.txt", "a") as f:
            f.write(f"distance metric={distance_metric}: {result}\n")
            f.write("\n")
        print(f"{distance_metric} done")


###############################################
# Comparing my LBP implementation with Scikit #
###############################################

def compare_with_scikit(distance_metric, R=1, P=8, use_histograms=True, 
                        uniform=False, cyclic_shifting=False):
    """ Compare my implementation of LBP with Scikit's """

    method = "uniform" if uniform else "default"
    bins_size = (P + 1) if uniform else 2 ** P

    # Scikit LBP
    vectors = []

    for img in images:
        converted_image = local_binary_pattern(img, P, R, method=method)
        if use_histograms:
            vectors.append(apply_histogram(converted_image, (16, 16), bins_size))
        else:
            vectors.append(converted_image.flatten())

    scikit_rank1 = calculate_rank1_accuracy(vectors, classes, distance_metric)

    # My implementation
    vectors = lbp(images, R=R, P=P, use_histograms=use_histograms,
                  uniform=uniform, cyclic_shifting=cyclic_shifting)
    my_rank1 = calculate_rank1_accuracy(vectors, classes, distance_metric)

    print("Scikit implementation: ", scikit_rank1)
    print("My implementation: ", my_rank1)


test_pixels()
test_non_uniform()
test_uniform()
# basic_LBP_testing()
# compare_with_scikit("cosine", use_histograms=True)
