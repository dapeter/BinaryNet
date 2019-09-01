import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from glob import glob


data_dirs = ["mnist_mlp_1_layer", "mnist_mlp_2_layer", "mnist_cnn_2_layer"]

fig = plt.plot()
plt.grid()
plt.xlabel("alpha")
plt.ylabel("prediction accuracy")

for data_dir in data_dirs:
    sub_dirs = glob('./results/' + data_dir + "/*/")

    binary_performance = []
    nonbinary_performance = []

    for sub_dir in sub_dirs:
        meta = {}
        with open(sub_dir + "params.txt", "r") as param_file:
            for line in param_file:
                key, val = line.partition("=")[::2]
                meta[key.strip()] = val.strip()

        alpha = int(meta["nalpha"])
        accuracy = np.loadtxt(sub_dir + "performance.dat")[-1, 8]
        if meta["binary"] == "True":
            binary_performance.append([alpha, accuracy])
        elif meta["binary"] == "False":
            nonbinary_performance.append([alpha, accuracy])
        else:
            print("???")
            exit(-1)

    binary_performance = np.sort(np.array(binary_performance), axis=0)
    nonbinary_performance = np.sort(np.array(nonbinary_performance), axis=0)
    meta = {}
    with open('./results/' + data_dir + "/meta.txt", "r") as param_file:
        for line in param_file:
            key, val = line.partition("=")[::2]
            meta[key.strip()] = val.strip()

    plt.plot(nonbinary_performance[:, 0], 100-nonbinary_performance[:, 1], linestyle="-", label=meta["network"] + meta["layer"]+"nb")
    plt.plot(binary_performance[:, 0], 100-binary_performance[:, 1], linestyle="--", label=meta["network"] + meta["layer"]+"b")

plt.legend(loc="upper right")
plt.show()
exit(0)