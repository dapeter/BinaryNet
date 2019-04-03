import os
import numpy as np
import matplotlib.pyplot as plt
import csv

data_dirs = ["single_layer_64", "single_layer_256", "single_layer_1024"]
comp_files = ["./finished_runs/" + d + "/results/comparison.csv" for d in data_dirs]

for fname in comp_files:
    if not os.path.isfile(fname):
        raise FileNotFoundError

# Single plot for every architecture
for i, fname in enumerate(comp_files):
    # Init csv file reader
    csvfile = open(fname, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    binary = []
    non_binary = []
    for row in csv_reader:
        if row[0] == "True":
            binary.append([float(row[2]), float(row[4])])
        elif row[0] == "False":
            non_binary.append([float(row[2]), float(row[4])])
        else:
            raise ValueError

    binary = np.array(binary)
    non_binary = np.array(non_binary)

    fig = plt.plot()
    plt.grid()
    plt.xlim([0, 50])
    plt.ylim([70, 100])
    plt.title("MNIST - " + data_dirs[i])
    plt.xlabel("alpha")
    plt.ylabel("prediction accuracy")

    plt.plot(non_binary[:, 0], 100 - non_binary[:, 1], color="red", linestyle="-", label="NN")
    plt.plot(binary[:, 0], 100-binary[:, 1], color="green", linestyle="--", label="BNN")
    plt.legend(loc="upper right")

    plt.savefig("./plots/MNIST_{}.png".format(data_dirs[i]))
    plt.close()

# Plot comparision of NNs
fig = plt.plot()
plt.grid()
plt.xlim([0, 50])
plt.ylim([70, 100])
plt.xlabel("alpha")
plt.ylabel("prediction accuracy")
plt.title("MNIST - NN")
color = ["red", "green", "blue"]
for i, fname in enumerate(comp_files):
    # Init csv file reader
    csvfile = open(fname, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    non_binary = []
    for row in csv_reader:
        if row[0] == "True":
            pass
        elif row[0] == "False":
            non_binary.append([float(row[2]), float(row[4])])
        else:
            raise ValueError

    non_binary = np.array(non_binary)

    plt.plot(non_binary[:, 0], 100-non_binary[:, 1], color=color[i], linestyle="-", label=data_dirs[i])

plt.legend(loc="upper right")
plt.savefig("./plots/MNIST_nn_comp.png")
plt.close()

# Plot comparision of BNNs
fig = plt.plot()
plt.grid()
plt.xlim([0, 50])
plt.ylim([70, 100])
plt.xlabel("alpha")
plt.ylabel("prediction accuracy")
plt.title("MNIST - BNN")
color = ["red", "green", "blue"]
for i, fname in enumerate(comp_files):
    # Init csv file reader
    csvfile = open(fname, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    binary = []
    for row in csv_reader:
        if row[0] == "True":
            binary.append([float(row[2]), float(row[4])])
        elif row[0] == "False":
            pass
        else:
            raise ValueError

    binary = np.array(binary)

    plt.plot(binary[:, 0], 100-binary[:, 1], color=color[i], linestyle="-", label=data_dirs[i])

plt.legend(loc="upper right")
plt.savefig("./plots/MNIST_bnn_comp.png")
plt.close()

# Plot comparision for cifar CNN/BCNN
data_dirs = ["cnn_cifar"]
comp_files = ["./finished_runs/" + d + "/results/comparison.csv" for d in data_dirs]

# Single plot for every architecture
for i, fname in enumerate(comp_files):
    # Init csv file reader
    csvfile = open(fname, 'r')
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    binary = []
    non_binary = []
    for row in csv_reader:
        if row[0] == "True":
            binary.append([float(row[2]), float(row[4])])
        elif row[0] == "False":
            non_binary.append([float(row[2]), float(row[4])])
        else:
            raise ValueError

    binary = np.array(binary)
    non_binary = np.array(non_binary)

    fig = plt.plot()
    plt.grid()
    plt.xlim([0, 5])
    plt.ylim([40, 80])
    plt.title("cifar10 - " + data_dirs[i])
    plt.xlabel("alpha")
    plt.ylabel("prediction accuracy")

    plt.plot(non_binary[:, 0], 100 - non_binary[:, 1], color="red", linestyle="-", label="CNN")
    plt.plot(binary[:, 0], 100-binary[:, 1], color="green", linestyle="--", label="BCNN")
    plt.legend(loc="upper right")

    plt.savefig("./plots/cifar10_{}.png".format(data_dirs[i]))
    plt.close()