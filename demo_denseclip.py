import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from mmseg.core.evaluation import get_palette, get_classes
from torchvision.utils import draw_segmentation_masks

from libs.models import DenseClip
from libs.visualization import plot_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    # classnames = ['Batman', 'Joker', 'background']
    classnames = get_classes('pascal_voc')

    templates = ['A photo of a {}.']
    model = DenseClip('RN50x16', classnames, templates, device=args.device)
    model.eval()
    threshold = 0.7

    filename = '2012_004325.jpg'
    with Image.open(filename, 'r') as image:
        image = image.convert('RGB')
        image_tensor = TF.to_tensor(image).multiply(255).to(torch.uint8)

        prep_image = model.preprocess(image)
        prep_image = prep_image.unsqueeze(dim=0)
        prep_image = prep_image.to(args.device)
        output = model(prep_image)
        output = F.interpolate(output, size=image_tensor.shape[-2:], mode='bilinear')  # [1, C, H, W]
        output = F.softmax(output, dim=1)
        output = output.squeeze(dim=0)  # [C, H, W]
        # output = output[:-1]  # [C-1, H, W]
        # masks = output > threshold
        masks = torch.zeros_like(output, dtype=torch.bool)
        for i in range(output.shape[0]):
            masks[i] = torch.where(output.argmax(dim=0) == i, True, False)

        masks = masks.cpu()
        colors = list(map(tuple, get_palette('pascal_voc')))
        segmentation_masks = draw_segmentation_masks(image_tensor, masks, colors=colors)
        segmentation_masks = segmentation_masks.permute(1, 2, 0)

        images = [np.array(image), segmentation_masks]
        plot_images(images, f'example_{filename}')


if __name__ == '__main__':
    main()
