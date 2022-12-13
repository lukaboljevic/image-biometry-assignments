import os
from glob import glob
from time import perf_counter
from itertools import product
from fnc.extractFeature import extractFeature
from fnc.matching import matching


TEMPLATE_DIR = "./templates/"

def rank_n_accuracy(N_list, threshold_list):
    """
    Verify all unenrolled iris images (all iris images with name *_2_*.jpg)
    against the enrolled templates (iris images with name *_1_*.jpg). This
    can be changed of course.
    
    Calculate rank-1 and rank-5 accuracy for a given list of matching thresholds.
    """
    files = glob(os.path.join("../CASIA/*/*/*_2_*.jpg"), recursive=True)
    num_files = len(files)

    combinations = list(product(N_list, threshold_list))  # pair every N with every threshold
    combinations = list(map(lambda x: f"rank{x[0]}-{x[1]:.2f}", combinations))  # map pairs to (later) file names
    verified_dict = dict.fromkeys(combinations, 0)  # we'll count the number of verified query images for each combination

    start = perf_counter()
    for i, file in enumerate(files):
        parsed = file.split("\\")
        subject = parsed[1]
        image_name = parsed[-1]
        print(f">>> Verifying image {image_name} ({(i+1):03d} / {num_files})...")

        # Extract iris template and noise mask
        template, mask, _ = extractFeature(file)

        # Match for all thresholds
        result_dict = matching(template, mask, TEMPLATE_DIR, threshold_list)

        if result_dict == -1:
            print("\n>>> No enrolled templates in the database!\n")
            exit(1)
        else:
            for threshold in threshold_list:
                res = result_dict[threshold]
                for N in N_list:
                    top_n = res[0:N]  # take first N matches; will work even if res = [] i.e. if there were no matches
                    subjects = list(map(lambda x: x[0:3], top_n))  # extract subject ID from template name
                    verified_dict[f"rank{N}-{threshold:.2f}"] += (subject in subjects)

    end = perf_counter()

    for combination, correct in verified_dict.items():
        verified_dict[combination] = correct / num_files

    print("\n>>> Verification of all images completed.\n" + 
          f">>> Elapsed time: {((end - start) / 60):.3f} [minutes]\n")

    return verified_dict


if __name__ == "__main__":
    accuracies_dir = "./accuracies"
    if not os.path.exists(accuracies_dir):
        os.mkdir(accuracies_dir)

    N_list = [1, 5]  # for rank-N accuracy
    threshold_list = [0.1, 0.2, 0.3, 0.38, 0.45, 0.5]
    verified_dict = rank_n_accuracy(N_list, threshold_list)
    
    for combination in verified_dict:
        with open(f"{accuracies_dir}/testOn2-{combination}.txt", "w") as f:
            f.write(f"Accuracy: {verified_dict[combination]:.4f}\n")