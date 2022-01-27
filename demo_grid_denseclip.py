import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from mmseg.core import get_palette
from torchvision.utils import draw_segmentation_masks

from libs.datasets import OxfordIIITPet
from libs.models import DenseClip, Clip
from libs.visualization import plot_image_grid

np.random.seed(0)


@dataclass
class ImageInGrid:
    def __init__(self, image, x_label):
        self.image = image
        self.x_label = x_label


def draw_segmentation_masks_from_prob(image_tensor, prob_map, palette=get_palette('pascal_voc')):
    masks = torch.zeros_like(prob_map, dtype=torch.bool)
    for j in range(prob_map.shape[0]):
        masks[j] = torch.where(prob_map.argmax(dim=0) == j, True, False)

    colors = list(map(tuple, palette))
    segmentation_masks = draw_segmentation_masks(image_tensor, masks, colors=colors)
    segmentation_masks = segmentation_masks.permute(1, 2, 0)
    return segmentation_masks


def draw_topk_confidence_map(dataset, prep_image, prob_map, raw_clip, top_k):
    score = raw_clip(prep_image)
    score = score.squeeze(dim=0)
    top_values, top_indices = torch.topk(score, k=top_k)
    confidence_maps = []
    for k in range(top_k):
        top_index = top_indices[k]
        label = dataset.classes[top_index]
        confidence_maps.append(
            ImageInGrid(prob_map[top_index], label)
        )
    return confidence_maps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--data_root', type=str)
    return parser.parse_args()


@torch.no_grad()
def main():
    n_samples = 7
    top_k = 5
    args = parse_args()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    dataset = OxfordIIITPet(args.data_root, transform=transform)
    templates = ['A photo of a {}, a type of pet.']
    dense_clip = DenseClip('RN50', dataset.classes, templates, args.device)
    dense_clip.eval()
    raw_clip = Clip('RN50', dataset.classes, templates, args.device)
    raw_clip.eval()

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    image_grid = []

    # quantize color map
    color_map = plt.get_cmap('turbo')
    colors = np.array(color_map.colors) * 256
    colors = colors.astype(int)
    color_indices = np.linspace(0, 255, len(dataset.classes)).astype(int)
    palette = [tuple(colors[color_index])
               for color_index in color_indices]
    for idx in indices[:n_samples]:
        with Image.open(dataset.filenames[idx], 'r').convert('RGB') as image:
            image_tensor = TF.to_tensor(image).multiply(255).to(torch.uint8)

            prep_image = dense_clip.preprocess(image)
            prep_image = prep_image.unsqueeze(dim=0)
            prep_image = prep_image.to(args.device)

            score_map = dense_clip(prep_image)
            score_map = F.interpolate(score_map, size=image_tensor.shape[-2:], mode='bilinear')  # [1, C, H, W]
            prob_map = F.softmax(score_map, dim=1)
            prob_map = prob_map.squeeze(dim=0)  # [C, H, W]
            prob_map = prob_map.cpu()

            segmentation_masks = draw_segmentation_masks_from_prob(image_tensor, prob_map, palette)
            confidence_maps = draw_topk_confidence_map(dataset, prep_image, prob_map, raw_clip, top_k)

            image_grid.append([
                ImageInGrid(np.array(image), dataset.labelnames[idx]),
                ImageInGrid(segmentation_masks, 'segmentation'),
                *confidence_maps,
            ])

    plot_image_grid(image_grid, 'Oxford_IIIT_Pet_grid_example.png')


if __name__ == '__main__':
    main()
