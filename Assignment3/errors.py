from matplotlib import pyplot as plt
import cv2 as cv
import pandas as pd


column_types = {
	"CID": str,
	"Image": str,
	"Gender": str,
	"Ethnicity": int,
	"LR": str,
	"Err": str
}


def get_errors():
    df = pd.read_csv("./annotations.csv", sep=";", dtype=column_types)
    df = df[df["Err"].isna() == False]
    return df[["CID", "Image", "Err"]].values


def main(error_list):
    print("#############################")
    print(f"# Number of errors: {len(error_list)}")
    print("#############################\n")
    for dir_name, img_name, error in error_list:
        image = cv.cvtColor(cv.imread(f"images/{dir_name}/{img_name[1:]}.png"), cv.COLOR_BGR2RGB)
        mask = cv.imread(f"masks/{dir_name}/{img_name[1:]}.png", cv.IMREAD_GRAYSCALE)
        alpha = 0.4*(mask > 0)
        plt.figure(figsize=(9, 7))
        plt.title(f"Subject: {dir_name}, image: {img_name[1:]}, error: {error}")
        plt.imshow(image)
        plt.imshow(mask, alpha = alpha)
        plt.show()


if __name__ == "__main__":
	main(get_errors())