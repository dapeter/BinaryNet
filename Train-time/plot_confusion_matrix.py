import os
import numpy as np
import csv
from glob import glob
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# MNIST
confusion_folder = "./results/confusion/"
sub_dirs = glob(confusion_folder + "*/")

for sub_dir in sub_dirs:
    meta = {}
    with open(sub_dir + "params.txt", "r") as param_file:
        for line in param_file:
            key, val = line.partition("=")[::2]
            meta[key.strip()] = val.strip()

    alpha = int(meta["nalpha"])
    confusion_matrix = np.loadtxt(sub_dir + "confusion.dat")
    if meta["binary"] == "True":
        binary = True
    elif meta["binary"] == "False":
        binary = False
    else:
        print("???")
        exit(-1)

    # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    labels = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    confusion_matrix = confusion_matrix  # Convert to percent
    df_cm = pd.DataFrame(confusion_matrix.astype(int), index=labels, columns=labels)
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, fmt="d", vmin=0, vmax=850, cmap="Blues", cbar_kws={'ticks': [0, 100, 200, 300, 400, 500, 600, 700, 800]})
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        plt.xticks(rotation=45)
    plt.xlabel("Predicted label", fontweight='bold')
    plt.ylabel("True label", fontweight='bold')
    plt.savefig("./plots/confusion/cifar_conv4_b" + str(binary) + "_a" + str(alpha) + ".png", bbox_inches='tight')
    plt.close()
