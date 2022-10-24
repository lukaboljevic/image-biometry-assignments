from utils import read_dataset
from code import lbp, calculate_rank1_accuracy


image_size = (128, 128)
images, classes = read_dataset(image_size)


#######################################
# Non uniform LBP with various params #
#######################################

def test_non_uniform():
    Rs = [1, 2, 3, 4]
    use_hists = [False, True]
    cyclic_shiftings = [False, True]
    distance_metric = "cosine"

    for R in Rs:
        Ps = [4*R, 8*R]
        region_step = 2*R + 1

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

                    with open("128x128-defaultLBP.txt", "a") as f:
                        f.write(f"R={R}, P={P}, step={region_step}, hist={use_hist},\n" +
                                f"cyclic shifting={cyclic_shifting}, " +
                                f"distance metric={distance_metric}")
                        f.write(f"\n\t{result}\n")
                        f.write("\n")


###################################
# Uniform LBP with various params #
###################################

def test_uniform():
    Rs = [1, 2, 3, 4]
    use_hists = [False, True]
    distance_metric = "cosine"

    for R in Rs:
        Ps = [4*R, 8*R]
        region_step = 2*R + 1
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

                with open("128x128-uniformLBP.txt", "a") as f:
                    f.write(f"R={R}, P={P}, step={region_step}, hist={use_hist}, " +
                            f"distance metric={distance_metric}")
                    f.write(f"\n\t{result}\n")
                    f.write("\n")


####################################
# Testing with "static" parameters #
####################################

def basic_testing():
    R = 1
    P = 8 * R
    region_step = 2*R + 1
    use_hist = False
    cyclic_shift = False
    uniform = False
    distance_metric = "euclidean"

    vectors = lbp(images, R=R, P=P,
                  local_region_step=region_step,
                  use_histograms=use_hist,
                  uniform=uniform,
                  cyclic_shifting=cyclic_shift)
    print("LBP done")

    result = calculate_rank1_accuracy(vectors, classes, distance_metric)
    print(result)

test_uniform()
