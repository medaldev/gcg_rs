import matplotlib.pyplot as plt
import numpy as np
import sys

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
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.046, pad=0.04)

    plt.show(block=True)
        # Your code here, the script continues to run
