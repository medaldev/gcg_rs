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
    plt.figure(figsize=(15, 15))
    plt.tight_layout()
    plt.imshow(matrix, cmap="jet")
    plt.show()