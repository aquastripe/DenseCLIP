import matplotlib.pyplot as plt


def plot_images(images, filename, figsize=(7.1, 2.41)):
    fig, ax = plt.subplots(ncols=len(images), tight_layout=True, figsize=figsize)
    for i, image in enumerate(images):
        ax[i].imshow(image)
        ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.savefig(filename)
