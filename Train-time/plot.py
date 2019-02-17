import os
import re
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    result_dir = "./results/"

    for root, dirs, data_files in os.walk(result_dir):
        data_files = [fname for fname in data_files if fname.endswith(".dat")]

    data = []

    for fname in data_files:
        data = np.loadtxt(result_dir + fname)
        fname = fname.replace(".dat", "")

        fname_split = fname.split("_")
        parameters = [fname_split[i] for i in range(1, len(fname_split), 2)]
        parameters[0] = (parameters[0] == "True")
        parameters[2] = int(parameters[2])

        fig, ax_list = plt.subplots(1, 2)
        plt.tight_layout(pad=2, w_pad=2, h_pad=1)

        ax_list[0].plot(data[:, 0], color='blue', label='train loss', lw=2)
        ax_list[0].plot(data[:, 1], color='green', label='val loss', lw=2)
        ax_list[0].legend(loc="upper right")

        ax_list[1].set_ylim([0, 100])
        ax_list[1].plot(data[:, 2], color='orange', label='val error', lw=2)
        ax_list[1].axhline(y=data[:, 3][0], color='red', linestyle='--', label='best test error', lw=2)
        ax_list[1].legend(loc="upper right")

        plt.savefig("./plots/" + fname + ".png")
        plt.close(fig)
