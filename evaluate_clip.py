import argparse

import torch
import torchvision.transforms as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.datasets import OxfordIIITPet
from libs.models import Clip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--data_root', type=str)
    return parser.parse_args()


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main():
    args = parse_args()
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    dataset = OxfordIIITPet(args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=8)
    templates = ['A photo of a {}, a type of pet.']
    model = Clip('RN50', dataset.classes, templates, args.device)

    with torch.no_grad():
        sum_top1, sum_top5, n = 0., 0., 0.
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(args.device), labels.to(args.device)
            probs = model(images)

            top1, top5 = accuracy(probs, labels, topk=(1, 5))
            sum_top1 += top1
            sum_top5 += top5
            n += images.size(0)

        top1 = (sum_top1 / n)
        top5 = (sum_top5 / n)

        print(f'Top-1 accuracy: {top1: .4f}')
        print(f'Top-5 accuracy: {top5: .4f}')


if __name__ == '__main__':
    main()
