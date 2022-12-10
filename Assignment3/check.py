import pandas as pd
import os


column_types = {
	"CID": str,
	"Image": str,
	"Gender": str,
	"Ethnicity": int,
	"LR": str,
	"Err": str
}

def check():
    """
    Double check that the correction process went smoothly 
    (at least as much as a simple script can check)
    """
    mask_root = "./corrected/masks/"
    images_root = "./corrected/images/"

    # Get all subjects with an error
    df = pd.read_csv("./annotations.csv", sep=";", dtype=column_types)
    df = df[df["Err"].isna() == False].reset_index(drop=True)
    df = df.drop(["Gender", "Ethnicity", "LR"], axis=1)


    # Check that number of corrected masks == number of errors
    num_wrong = len(df[df["Err"] == "wrong subject"])
    num_masks = sum( [ len(os.listdir(mask_root + d)) for d in os.listdir(mask_root) ] )
    if num_wrong + num_masks != len(df):
        print("There exists a mask which is not fixed!\n")


    # Check that number of errors for a specific subject == number of 
    # corrected masks for that subject (excluding wrong subject errors).
    # Also check that for each two ears error, there is a corresponding 
    # corrected image, and that the size of the image is bigger than the
    # size of the mask (otherwise I saved them in the wrong folders!).
    for subject in os.listdir(mask_root):
        subject_errors = df[df["CID"] == subject]

        num_masks = len(os.listdir(mask_root + subject))
        num_wrong = len(subject_errors[subject_errors["Err"] == "wrong subject"])
        if num_masks != len(subject_errors) - num_wrong:
            print(f"Subject {subject} has an incorrect amount of errors fixed!\n")

        twoears_errors = subject_errors[subject_errors["Err"].str.find("two ears") > -1]
        twoears_errors = twoears_errors[["CID", "Image"]].values
        for subject_id, image_id in twoears_errors:
            # we know mask exists, so check for image
            temp = image_id[1:] # remove the first 0

            if not os.path.exists(f"{images_root}{subject_id}/{temp}.png"):
                print(f"Two ears image for subject {subject_id}, image {temp} missing!")
            else:
                img_size = os.path.getsize(f"{images_root}{subject_id}/{temp}.png")
                mask_size = os.path.getsize(f"{mask_root}{subject_id}/{temp}.png")
                if img_size < mask_size:
                    print(f"Two ears mask size for subject {subject_id}, image {temp} greater than image size!")


if __name__ == "__main__":
    check()