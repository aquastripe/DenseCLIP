import matplotlib.pyplot as plt


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
