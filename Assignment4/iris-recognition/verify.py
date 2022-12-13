# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------
import argparse
import re
import os
from pathlib import Path
from time import perf_counter

from fnc.extractFeature import extractFeature
from fnc.matching import matching


# ------------------------------------------------------------------------------
#	Argument parsing
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str,
                    help="Path to the file that you want to verify.")

parser.add_argument("--temps", type=str, default="./templates/",
                    help="Path to the directory containing templates.")

parser.add_argument("--thres", type=float, default=0.38,
                    help="Threshold for matching.")

args = parser.parse_args()

def main():
    # -----------------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------------
    # Extract feature (iris template)
    start = perf_counter()
    parsed = args.file.split("/")
    subject = parsed[2]
    image_name = parsed[-1]
    print(f">>> Subject: {subject}, image name: {image_name}")
    print(f">>> Matching threshold: {args.thres}")
    print(f">>> Start verifying image {image_name}...\n")
    template, mask, file = extractFeature(args.file)

    
    # Matching
    result = matching(template, mask, args.temps, args.thres)


    if result == -1:
        print('>>> Database empty - no enrolled templates.')

    elif result == 0:
        print('>>> No template matched.')

    else:
        print(f">>> Subject {subject} verified.\n" +
              f">>> {len(result)} templates matched (descending reliability):")
        for i, res in enumerate(result):
            print(f"\t({i+1}) Subject: {res[0:3]}, template: {res}")

            # To return the path to the original file
            # reg_pattern = re.compile(res[:-4] + '$')
            # for root, dirs, files in os.walk(os.path.abspath("../")):
            #     for file in files:
            #         if reg_pattern.match(file):
            #             filepath = os.path.join(root, file)
            #             print("\t", "Path to original Image: ",
            #                 filepath)


    # Time measure
    end = perf_counter()
    print(f"\n>>> Verification time: {(end - start):.3f} [s]\n")

if __name__ == '__main__':
    main()

