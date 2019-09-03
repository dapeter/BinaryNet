import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "./results/???/"  # Change here
    save_folder = "./plots/learning/???/"  # Change here

    meta = {}
    with open(data_path + "/meta.txt", "r") as param_file:
        for line in param_file:
            key, val = line.partition("=")[::2]
            meta[key.strip()] = val.strip()

    data = []
    for folder in glob(data_path + "/*/"):
        if os.path.isfile(folder + "performance.dat"):
            data = np.loadtxt(folder + "performance.dat")
        else:
            print("Missing performance data in " + folder)
            exit(-1)

        # Epoch, LR, Training Loss, Validation Loss, Validation Error,
        # Best Epoch, Best Validation Error, Test Loss, Test Error
        train_loss = data[:, 2]
        val_loss = data[:, 3]
        test_loss = data[:, 7]

        val_error = data[:, 4]
        test_error = data[:, 8]

        best_epoch = data[-1, 5]

        fig, ax_list = plt.subplots(1, 2)
        plt.tight_layout(pad=2, w_pad=2, h_pad=1)

        ax_list[0].plot(train_loss, color='blue', label='train loss', lw=2)
        ax_list[0].plot(val_loss, color='green', label='val loss', lw=2)
        ax_list[0].plot(test_loss, color='red', label='test loss', lw=2)
        ax_list[0].legend(loc="upper right")

        ax_list[1].set_ylim([0, 100])
        ax_list[1].plot(val_error, color='green', label='val error', lw=2)
        ax_list[1].plot(test_error, color='red', label='test error', lw=2)
        ax_list[1].axvline(x=best_epoch, color='gray', linestyle='--', label='best epoch', lw=2)
        ax_list[1].legend(loc="upper right")

        params = {}
        with open(folder + "params.txt", "r") as param_file:
            for line in param_file:
                key, val = line.partition("=")[::2]
                params[key.strip()] = val.strip()

        fname = meta["dataset"] + "_" + meta["network"] + meta["layer"] + "_a" + params["nalpha"]
        if params["binary"] == "True":
            plt.savefig(save_folder + fname + "_b.png")
        elif params["binary"] == "False":
            plt.savefig(save_folder + fname + ".png")
        else:
            print("Param file corrupted in " + folder)
            exit(-1)

        plt.close(fig)
