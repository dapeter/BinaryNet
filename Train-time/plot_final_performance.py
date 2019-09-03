import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from glob import glob

# MNIST
data_dirs = ["mnist_mlp_1_layer", "mnist_mlp_2_layer", "mnist_cnn_2_layer"]
colors = ["r", "g", "b", "m", "o", "c"]

c = 0
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

    plt.figure(1)
    plt.plot(nonbinary_performance[:, 0], 100-nonbinary_performance[:, 1], colors[c]+"o-", label=meta["network"] + meta["layer"])

    plt.figure(2)
    plt.plot(binary_performance[:, 0], 100-binary_performance[:, 1], colors[c]+"s--", label="Binarized " + meta["network"] + meta["layer"])

    plt.figure(3)
    plt.plot(nonbinary_performance[:, 0], 100-nonbinary_performance[:, 1], colors[c]+"o-", label=meta["network"] + meta["layer"])
    plt.plot(binary_performance[:, 0], 100-binary_performance[:, 1], colors[c]+"s--", label="Binarized " + meta["network"] + meta["layer"])

    plt.grid()
    plt.title("MNIST - Uniform label noise - " + meta["network"] + meta["layer"])
    plt.xlabel("Number of noisy labels per clean label")
    plt.ylabel("Prediction accuracy")
    plt.ylim(50, 100)
    plt.xlim(0, 100)
    plt.legend(loc="lower left")
    plt.savefig("./plots/mnist_uniform_" + meta["network"] + meta["layer"] + ".png")
    plt.close(3)

    c = c + 1

plt.figure(1)
plt.grid()
plt.title("MNIST - Uniform label noise - Overview")
plt.xlabel("Number of noisy labels per clean label")
plt.ylabel("Prediction accuracy")
plt.ylim(50, 100)
plt.xlim(0, 100)
plt.legend(loc="lower left")
plt.savefig("./plots/mnist_uniform_overview.png")
plt.close(1)

plt.figure(2)
plt.grid()
plt.title("MNIST - Uniform label noise - Overview")
plt.xlabel("Number of noisy labels per clean label")
plt.ylabel("Prediction accuracy")
plt.ylim(50, 100)
plt.xlim(0, 100)
plt.legend(loc="lower left")
plt.savefig("./plots/mnist_uniform_overview_binary.png")
plt.close(2)


# Cifar10
data_dirs = ["cifar_cnn_2_layer"]
colors = ["r", "g", "b", "m", "o", "c"]

c = 0
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

    plt.figure(4)
    plt.plot(nonbinary_performance[:, 0], 100-nonbinary_performance[:, 1], colors[c]+"o-", label=meta["network"] + meta["layer"])

    plt.figure(5)
    plt.plot(binary_performance[:, 0], 100-binary_performance[:, 1], colors[c]+"s--", label="Binarized " + meta["network"] + meta["layer"])

    plt.figure(6)
    plt.plot(nonbinary_performance[:, 0], 100-nonbinary_performance[:, 1], colors[c]+"o-", label=meta["network"] + meta["layer"])
    plt.plot(binary_performance[:, 0], 100-binary_performance[:, 1], colors[c]+"s--", label="Binarized " + meta["network"] + meta["layer"])

    plt.grid()
    plt.title("CIFAR10 - Uniform label noise - " + meta["network"] + meta["layer"])
    plt.xlabel("Number of noisy labels per clean label")
    plt.ylabel("Prediction accuracy")
    plt.ylim(40, 80)
    plt.xlim(0, 10)
    plt.legend(loc="lower left")
    plt.savefig("./plots/cifar_uniform_" + meta["network"] + meta["layer"] + ".png")
    plt.close(6)

    c = c + 1

plt.figure(4)
plt.grid()
plt.title("CIFAR10 - Uniform label noise - Overview")
plt.xlabel("Number of noisy labels per clean label")
plt.ylabel("Prediction accuracy")
plt.ylim(40, 80)
plt.xlim(0, 10)
plt.legend(loc="lower left")
plt.savefig("./plots/cifar_uniform_overview.png")
plt.close(4)

plt.figure(5)
plt.grid()
plt.title("CIFAR10 - Uniform label noise - Overview")
plt.xlabel("Number of noisy labels per clean label")
plt.ylabel("Prediction accuracy")
plt.ylim(40, 80)
plt.xlim(0, 10)
plt.legend(loc="lower left")
plt.savefig("./plots/cifar_uniform_overview_binary.png")
plt.close(5)