import matplotlib

matplotlib.use('TkAgg')
print("Using:",matplotlib.get_backend())



import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gc
import cv2
from multiprocessing import Pool

import json
#plt.style.use('dark_background')


def read_matrix(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())), file_matrix.read().strip().split("\n"))))
    return matrix

def read_mc_tensor(path):
    with open(path, "r", encoding="utf8") as file_tensor:
        data = json.load(file_tensor)
    return np.array(data)




if __name__ == '__main__':
    args = sys.argv
    example_path = args[1]
    fig, axes = plt.subplots(4, 16, figsize=(20, 30))

    images = []

    true_matrix = read_matrix(os.path.join(example_path, "Uvych2_abs.xls"))
    tensor = read_mc_tensor(os.path.join(example_path, "Uvych2_abs_noised.tensor"))


    delthas = [np.mean(np.abs(true_matrix - tensor[i])) for i in range(len(tensor))]
    means = [np.mean(np.abs((sum(tensor[:i]) / i) - true_matrix) / true_matrix) for i in range(1, len(tensor) + 1)]
    print(means)


    # images.append(axes[0, 0].scatter(range(0, len(delthas)), delthas))
    # axes[0, 0].set_title("Delthas")
    #
    # images.append(axes[0, 1].plot(means))
    # axes[0, 1].set_title("Means")

    print(tensor[0])

    p = 0
    for i in range(16):
        for j in range(4):
            images.append(axes[j, i].imshow(tensor[p], cmap="jet"))
            p+= 1

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

#plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.15)

    # Save the full figure...
    #fig.savefig(os.path.join(save_dir, f'{name}.png'))
    plt.show(block=True)

    # plt.clf()
    # matplotlib.pyplot.close()

    # del fig, axes, images
    # gc.collect()

# plt.tight_layout()    # Your code here, the script continues to run
