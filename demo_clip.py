import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from mmseg.core import get_classes

from libs.models import Clip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    classnames = get_classes('pascal_voc')
    # templates = ['A photo of a {}.']
    model = Clip('RN50', classnames, device=args.device)
    model.eval()

    filename = 'dog.jpg'
    with Image.open(filename, 'r') as image:
        image = image.convert('RGB')

        prep_image = model.preprocess(image)
        prep_image = prep_image.unsqueeze(dim=0)
        prep_image = prep_image.to(args.device)
        output = model(prep_image)
        output = F.softmax(output, dim=1)
        output = output.squeeze(dim=0)
        idx = output.argmax(dim=0)
        print(f'Class: {classnames[idx]}, confidence: {output}')


if __name__ == '__main__':
    main()
