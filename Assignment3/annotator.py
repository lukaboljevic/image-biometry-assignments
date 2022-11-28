from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons, Button, CheckButtons
import cv2 as cv
import os
import pandas as pd
import csv


column_types = {
	"CID": str,
	"Image": str,
	"Gender": str,
	"Ethnicity": int,
	"LR": str,
	"Err": str
}


#############
### Utils ###
#############

def checkpoint():
	if not os.path.exists("./annotations.csv"):
		return None, None

	df = pd.read_csv("./annotations.csv", sep=";", dtype=column_types)
	if not len(df):
		return None, None

	directory_reached, image_reached = df[["CID", "Image"]].tail(1).values.flatten()
	return directory_reached, image_reached[1:] # remove the first 0


def get_files(last_dir=None, last_image=None):
	dirs = list(range(151, 181)) + list(range(211, 241)) + list(range(271, 301))
	# dirs = [151, 152, 211, 212, 271, 300]
	for dir in dirs:
		dstr = "%03d" % dir
		nfiles = len(os.listdir(f"images/{dstr}"))
		for i in range(1, nfiles+1):
			fstr = "%02d" % i
			if not last_dir:
				yield dstr,fstr
			else:
				if int(dir) > int(last_dir) or (int(dir) == int(last_dir) and
				int(fstr) > int(last_image)):
					yield dstr, fstr


def write_data(gender, ethnicity, side, error):
	global dirstr,filestr
	t = "0" + filestr
	print(f"{dirstr};{t};{gender};{ethnicity};{side};{error}")
	return [dirstr, t, gender, ethnicity, side, error]


#####################
### The hard work ###
#####################

dirstr, filestr = None, None
last_dir, last_image = checkpoint()
print(f"Last directory: {last_dir}, last image: {last_image}\n")
gen = get_files(last_dir = last_dir, last_image = last_image)


def update_ui(fig,ax):
	global dirstr, filestr
	try:
		d, f = next(gen)
	except StopIteration:
		plt.close()
		return

	dirstr = d
	filestr = f
	image = cv.cvtColor(cv.imread(f"images/{dirstr}/{filestr}.png"), cv.COLOR_BGR2RGB)
	mask = cv.imread(f"masks/{dirstr}/{filestr}.png", cv.IMREAD_GRAYSCALE)
	alpha = 0.4*(mask > 0)
	plt.suptitle(f"Subject: {dirstr}, Image: {filestr}")
	# fig.set_size_inches(16, 7)
	ax.cla()
	ax.imshow(image)
	ax.imshow(mask, alpha=alpha)


def main():
	print("CID;Image;Gender;Ethnicity;LR;Err")
	fig, axs = plt.subplots(1,1)
	plt.subplots_adjust(right=0.55)

	gender_ax = fig.add_axes([0.7, 0.75, 0.15, 0.2])
	input_gender = RadioButtons(gender_ax, ('Female', 'Male'))
	input_gender_mapper = {'Female': 'f', 'Male': 'm'}
	input_gender.set_active(1)
 
	ethnicity_ax = fig.add_axes([0.7, 0.35, 0.2, 0.4])
	input_ethnicity = RadioButtons(ethnicity_ax, ('Caucasian (European)', 'Asian (Chinese, Jap, ..)', 'South asian (Indian, ..)', 'Black', 'Middle eastern (Saudi, Iran, ..)', 'Hispanic (Spain, spanish speaking)', 'Other'))
	input_ethnicity_mapper = {'Caucasian (European)': 1, 'Asian (Chinese, Jap, ..)': 2, 'South asian (Indian, ..)': 3, 'Black': 4, 'Middle eastern (Saudi, Iran, ..)': 5, 'Hispanic (Spain, spanish speaking)': 6, 'Other': 7}	
	
	error_ax = fig.add_axes([0.7, 0.1, 0.2, 0.25])
	input_error = RadioButtons(error_ax, ("NO error", "Mask off", "Mask off a bit", "Two ears present", "Wrong subject"))
	input_error_mapper = {
		"NO error": "",
		"Mask off": "mask off", 
		"Mask off a bit": "mask off a bit", 
		"Two ears present": "two ears", 
		"Wrong subject": "wrong subject"
	}
	
	submit_left_ax = fig.add_axes([0.65, 0.05, 0.1, 0.05])
	input_submit_left = Button(submit_left_ax, "Submit Left ear")

	submit_right_ax = fig.add_axes([0.8, 0.05, 0.1, 0.05])
	input_submit_right = Button(submit_right_ax, "Submit Right ear")
	
	def on_submit(side):
		# error_status = input_error.get_status()
		
		# if error_status[0]:
		# 	error = "mask off"
		# 	input_error.set_active(0)
		# elif error_status[1]:
		# 	error = "mask off a bit"
		# 	input_error.set_active(1)
		# elif error_status[2]:
		# 	error = "two ears"
		# 	input_error.set_active(2)
		# elif error_status[3]:
		# 	error = "wrong subject"
		# 	input_error.set_active(3)
		# else:
		# 	error = ''

		new_row = write_data( 
			input_gender_mapper[input_gender.value_selected],
			input_ethnicity_mapper[input_ethnicity.value_selected],
			side,
			input_error_mapper[input_error.value_selected])

		with open("./annotations.csv", "a", newline='') as f:
			writer = csv.writer(f, delimiter=";")
			writer.writerow(new_row)

		update_ui(fig,axs)
		input_error.set_active(0)  # most of the times there's no error
		fig.canvas.draw_idle()

	input_submit_left.on_clicked(lambda x: on_submit('l'))
	input_submit_right.on_clicked(lambda x: on_submit('r'))

	update_ui(fig,axs)
	plt.show()


if __name__ == "__main__":
	if not os.path.exists("./annotations.csv"):
		with open("./annotations.csv", "w", newline='') as f:
			writer = csv.writer(f, delimiter=";")
			writer.writerow(column_types.keys())
	main()