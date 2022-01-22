import pytest
import torch

from libs.models import DenseClip

templates = ['A photo of a {}']
classnames = ['apple', 'banana', 'cat', 'dog', 'egg']


@pytest.fixture()
def device(pytestconfig):
    return pytestconfig.getoption('device')


def test_denseclip_encode_image(device):
    model = DenseClip('RN50', classnames=classnames, templates=templates, device=device)
    image = torch.rand(1, 3, 224, 224).to(device)
    output = model.encode_image(image)

    assert output.shape == torch.Size([1, 1024, 7, 7])


def test_zeroshot_weights_shape(device):
    model = DenseClip('RN50', classnames=classnames, templates=templates, device=device)
    assert model.zeroshot_weights.shape == torch.Size([1024, len(classnames)])


def test_denseclip_forward(device):
    model = DenseClip('RN50', classnames=classnames, templates=templates, device=device)
    batch_size = 5
    images = torch.rand((batch_size, 3, 224, 224), device=device)
    output = model(images)
    assert output.shape == torch.Size([batch_size, len(classnames), 224, 224])
