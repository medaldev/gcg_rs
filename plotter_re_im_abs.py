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
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    images = []

    name = example_path.split("/")[-1]

    images.append(axes[0].imshow(read_matrix(example_path + f"_re.xls"), cmap="jet"))
    axes[0].set_title(f"{name}_re", fontsize=30, pad=20)
    images.append(axes[1].imshow(read_matrix(example_path + f"_im.xls"), cmap="jet"))
    axes[1].set_title(f"{name}_im", fontsize=30, pad=20)
    images.append(axes[2].imshow(read_matrix(example_path + f"_abs.xls"), cmap="jet"))
    axes[2].set_title(f"{name}_abs", fontsize=30, pad=20)

    for im in images:
        fig.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, format='%.7f')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.15)



    # Save the full figure...
    #fig.savefig(os.path.join(save_dir, f'{name}.png'))
    plt.show(block=True)

    # plt.clf()
    # matplotlib.pyplot.close()

    # del fig, axes, images
    # gc.collect()

# plt.tight_layout()    # Your code here, the script continues to run
