from PIL import Image
import numpy as np
from petastorm.codecs import CompressedImageCodec
from petastorm.pytorch import decimal_friendly_collate
from petastorm.unischema import UnischemaField

from simadv.places365.hub import Places365Hub
from torchvision import transforms as trn


_RESNET18_TRANSFORM = trn.Compose([
    trn.Resize((224, 224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def test_image_preprocessing_works_like_original():
    """
    Transforms the same test image with the original and our new
    data pipeline.
    Checks that the results do not deviate too much.
    Deviations between both pipelines are visualized in a bw-Image.

    The original pipeline is File -> Torch transform to Tensor
    Our new pipeline is File -> Petastorm parquet store -> Tensor
    """
    hub = Places365Hub()
    resnet18 = hub.resnet18()

    with hub.open_demo_image() as demo_file:
        image_1 = Image.open(demo_file)
        image_tensor_1 = _RESNET18_TRANSFORM(image_1).unsqueeze(0)

        image_2 = Image.open(demo_file).resize((224, 224))
        if image_2.mode != 'RGB':
            image_2 = image_2.convert('RGB')
        image_2_arr = np.asarray(image_2)

        codec = CompressedImageCodec('png')
        field = UnischemaField('image', np.uint8, (224, 224, 3),
                               CompressedImageCodec('png'), False)

        image_2_encoded = codec.encode(field, image_2_arr)
        image_2_decoded = codec.decode(field, image_2_encoded)

        assert np.array_equal(image_2_decoded, image_2_arr)

        image_tensor_2 = decimal_friendly_collate(image_2_decoded)\
            .float().unsqueeze(0).permute(0, 3, 1, 2).div(255)
        resnet18._normalize_(image_tensor_2)

        max_diff = (image_tensor_1 - image_tensor_2).max()
        diff_image_tensor = (image_tensor_1 - image_tensor_2)\
            .abs().div(max_diff).squeeze()
        diff_image = trn.ToPILImage('RGB')(diff_image_tensor).convert('L')
        diff_image.show()

        mean_diff = (image_tensor_1 - image_tensor_2).mean()

        assert mean_diff < 0.005
