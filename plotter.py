import matplotlib

matplotlib.use('TkAgg')
print("Using:",matplotlib.get_backend())



import matplotlib.pyplot as plt
import numpy as np
import sys


#plt.style.use('dark_background')


def read_matrix(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())), file_matrix.read().strip().split("\n"))))
    return matrix


if __name__ == '__main__':
    args = sys.argv
    file_plot_path = args[1]
    matrix = read_matrix(file_plot_path)
    plt.figure(figsize=(15, 15), num=args[1])
    plt.tight_layout()
    im1 = plt.imshow(matrix, cmap="jet")
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.046, pad=0.04, format='%.7f')
   # plt.yticks(fontsize=16)
    cb1.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show(block=True)

   # plt.tight_layout()    # Your code here, the script continues to run
