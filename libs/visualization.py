import numpy as np
from matplotlib import pyplot as plt


def plot_images(images, filename, figsize=(7.1, 2.41)):
    fig, ax = plt.subplots(ncols=len(images), tight_layout=True, figsize=figsize)
    for i, image in enumerate(images):
        ax[i].imshow(image)
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.savefig(filename)


def plot_image_grid(image_grid, filename, figsize=(14.40, 14.40)):
    fig, ax = plt.subplots(nrows=len(image_grid), ncols=7, figsize=figsize, tight_layout=True)
    for row in range(len(image_grid)):
        for col in range(7):
            ax[row, col].imshow(image_grid[row][col].image)
            ax[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[],
                             xlabel=image_grid[row][col].x_label)

    fig.savefig(filename, bbox_inches='tight')


def plot_color_class(classes, palette, filename):
    nrows = len(classes)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axes = plt.subplots(nrows=len(classes), tight_layout=True, figsize=(6.40, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    for i in range(len(classes)):
        rgb = palette[i]
        image = np.array(rgb).reshape((1, 1, -1))
        axes[i].imshow(image, aspect='auto')
        axes[i].text(-0.05, 0.5, classes[i], ha='right', va='center', transform=axes[i].transAxes)
        axes[i].set_axis_off()

    fig.savefig(filename, bbox_inches='tight')