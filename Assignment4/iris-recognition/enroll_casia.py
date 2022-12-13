# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
import argparse
import os
from glob import glob
from tqdm import tqdm
from time import perf_counter
from scipy.io import savemat
from multiprocessing import cpu_count, Pool

from fnc.extractFeature import extractFeature


# ------------------------------------------------------------------------------
#	Argument parsing
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="../CASIA/",
                    help="Path to the directory containing CASIA images.")

parser.add_argument("--temps", type=str, default="./templates/",
                    help="Path to the directory containing CASIA templates.")

parser.add_argument("--ncores", type=int, default=cpu_count(),
                    help="Number of cores used for enrolling template.")

args = parser.parse_args()


# -----------------------------------------------------------------------------
# Pool function
# -----------------------------------------------------------------------------
def pool_func(file):
    template, mask, _ = extractFeature(file, use_multiprocess=False)
    basename = os.path.basename(file)
    out_file = os.path.join(args.temps, "%s.mat" % (basename))
    savemat(out_file, mdict={'template': template, 'mask': mask})

def main():
    # -----------------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------------
    start = perf_counter()

    # Check the existence of template directory
    if not os.path.exists(args.temps):
        print("Make template dir:", args.temps)
        os.mkdir(args.temps)

    # Uncomment one of the following two "files" variables
    # 1. Get list of files for enrolling, just "xxx_1_x.jpg" files are selected
    files = glob(os.path.join(args.data, "*/*/*_1_*.jpg"), recursive=True)

    # 2. Or get all the files, i.e. CASIA/*/*/*.jpg
    # files = glob(os.path.join(args.data, "*/*/*.jpg"), recursive=True)

    files = files
    n_files = len(files)
    print("Number of files for enrolling:", n_files)

    # Parallel pools to enroll templates
    print("Start enrolling...")
    pools = Pool(processes=args.ncores)
    for _ in tqdm(pools.imap_unordered(pool_func, files), total=n_files):
        pass

    end = perf_counter()
    print('\n>>> Enrollment time: {} [s]\n'.format(end-start))

if __name__ == '__main__':
    main()
