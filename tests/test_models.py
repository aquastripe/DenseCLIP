import pytest
import torch

from libs.models import DenseClip


@pytest.fixture()
def device(pytestconfig):
    return pytestconfig.getoption('device')


def test_denseclip_encode_image(device):
    model = DenseClip('RN50', device=device)
    image = torch.rand(1, 3, 224, 224).to(device)
    output = model.encode_image(image)

    assert output.shape == torch.Size([1, 1024, 7, 7])
