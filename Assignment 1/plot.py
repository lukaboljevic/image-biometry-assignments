import matplotlib.pyplot as plt
import numpy as np

# Plot simple bar plot used in report

x = np.arange(2)
x_labels = ["Default LBP", "Uniform LBP"]
width = 0.25
scikit = [0.348, 0.124]
mine = [0.355, 0.134]

rects1 = plt.bar(x - width/2, scikit, width, label="LBP")
rects2 = plt.bar(x + width/2, mine, width, label="Uniform LBP")
plt.ylabel("Rank-1 accuracy")
plt.title("Comparing our implementation against Scikit")
plt.xticks(x, x_labels)
plt.bar_label(rects1)#, padding=3)
plt.bar_label(rects2)#, padding=3)
plt.legend()
# plt.tight_layout()
plt.show()
