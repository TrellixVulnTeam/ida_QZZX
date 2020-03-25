import logging

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from thesis_impl.places365.hub import Places365Hub


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('Torch is using device {}.'.format(_DEVICE))


def _load_image_as_tensor(file_path, size=(224, 224)):
    image = Image.open(file_path).resize(size)
    return to_tensor(image)


def test_that_resnet18_works():
    hub = Places365Hub()
    resnet18 = hub.resnet18()

    with hub.open_demo_image() as demo_file:
        image_tensor = _load_image_as_tensor(demo_file).unsqueeze(0)
        label_probs = resnet18.predict_probabilities(image_tensor)
        _, label_ids = label_probs.sort(-1, True)

        assert hub.all_labels[label_ids[0]] == 'food court'
