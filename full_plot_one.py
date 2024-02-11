import matplotlib

matplotlib.use('TkAgg')
print("Using:",matplotlib.get_backend())



import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import gc

from multiprocessing import Pool


#plt.style.use('dark_background')


def read_matrix(path):
    with open(path, "r", encoding="utf8") as file_matrix:
        matrix = np.array(
            list(map(lambda row: list(map(lambda el: float(el), row.split())), file_matrix.read().strip().split("\n"))))
    return matrix





if __name__ == '__main__':
    args = sys.argv
    example_path = args[1]
    fig, axes = plt.subplots(4, 3, figsize=(20, 30))

    images = []

    images.append(axes[0, 0].imshow(read_matrix(os.path.join(example_path, "K_re.xls")), cmap="jet"))
    axes[0, 0].set_title("K_re")
    images.append(axes[0, 1].imshow(read_matrix(os.path.join(example_path, "K_im.xls")), cmap="jet"))
    axes[0, 1].set_title("K_im")
    images.append(axes[0, 2].imshow(read_matrix(os.path.join(example_path, "K_abs.xls")), cmap="jet"))
    axes[0, 2].set_title("K_abs")

    images.append(axes[1, 0].imshow(read_matrix(os.path.join(example_path, "J_re.xls")), cmap="jet"))
    axes[1, 0].set_title("J_re")
    images.append(axes[1, 1].imshow(read_matrix(os.path.join(example_path, "J_im.xls")), cmap="jet"))
    axes[1, 1].set_title("J_im")
    images.append(axes[1, 2].imshow(read_matrix(os.path.join(example_path, "J_abs.xls")), cmap="jet"))
    axes[1, 2].set_title("J_abs")

    images.append(axes[2, 0].imshow(read_matrix(os.path.join(example_path, "Uvych_re.xls")), cmap="jet"))
    axes[2, 0].set_title("Uvych_re")
    images.append(axes[2, 1].imshow(read_matrix(os.path.join(example_path, "Uvych_im.xls")), cmap="jet"))
    axes[2, 1].set_title("Uvych_im")
    images.append(axes[2, 2].imshow(read_matrix(os.path.join(example_path, "Uvych_abs.xls")), cmap="jet"))
    axes[2, 2].set_title("Uvych_abs")

    images.append(axes[3, 0].imshow(read_matrix(os.path.join(example_path, "K_inv_re.xls")), cmap="jet"))
    axes[3, 0].set_title("K_inv_re")
    images.append(axes[3, 1].imshow(read_matrix(os.path.join(example_path, "K_inv_im.xls")), cmap="jet"))
    axes[3, 1].set_title("K_inv_im")
    images.append(axes[3, 2].imshow(read_matrix(os.path.join(example_path, "K_inv_abs.xls")), cmap="jet"))
    axes[3, 2].set_title("K_inv_abs")


    for im in images:
        fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format='%.7f')

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
