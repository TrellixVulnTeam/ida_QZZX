import logging

import pytest
import torch
from petastorm import make_reader
from petastorm.codecs import CompressedImageCodec
from petastorm.pytorch import DataLoader, decimal_friendly_collate
import numpy as np
from PIL import Image
from petastorm.unischema import UnischemaField
from torchvision.transforms.functional import to_pil_image

from thesis_impl.places365 import wideresnet
from thesis_impl.places365.hub import Places365Hub


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('Torch is using device {}.'.format(_DEVICE))


def test_that_resnet18_works():
    hub = Places365Hub()
    resnet18 = hub.resnet18()

    with hub.open_demo_image() as demo_file:
        image = Image.open(demo_file)
        image_tensor = wideresnet.IMAGE_TRANSFORM(image).unsqueeze(0)
        label_probs = resnet18.predict_probabilities(image_tensor)
        _, label_ids = label_probs.sort(-1, True)

        assert hub.all_labels[label_ids[0]] == 'food court'


@pytest.fixture
def resnet18_evaluation(validation_url, batch_size):
    hub = Places365Hub()
    resnet18 = hub.resnet18().to(_DEVICE)
    resnet18.eval()
    reader = make_reader(validation_url,
                         num_epochs=1,
                         shuffle_row_groups=True)
    val_loader = DataLoader(reader, batch_size=batch_size)
    return hub, resnet18, val_loader


def _evaluate_on_validation_set(caplog, hub, val_loader, method):
    outer_level = logging.getLogger().level

    with caplog.at_level(logging.INFO):
        correct_of_label = torch.zeros(len(hub.all_labels))
        total_of_label = torch.zeros(len(hub.all_labels))

        num_processed = 0
        total_acc = 0

        try:
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(_DEVICE, torch.float)

                    # pytorch expects channels before width/height
                    images = images.permute(0, 3, 1, 2).div(255)

                    # test_image = to_pil_image(images[0])
                    # test_image.show()

                    assert images.min().item() >= 0
                    assert images.max().item() <= 1
                    labels = batch['label_id'].to(_DEVICE, torch.int16)

                    num_processed += images.size()[0]

                    with caplog.at_level(outer_level):
                        for label_id, prediction_is_correct \
                                in method(images, labels):
                            if prediction_is_correct:
                                correct_of_label[label_id] += 1
                            total_of_label[label_id] += 1

                    logging.info('Processed {} images so far.'
                                 .format(num_processed))
                    total_acc = (correct_of_label.sum() / total_of_label.sum())\
                        .squeeze()
                    logging.info('Current accuracy is {}.'.format(total_acc.item()))
        except KeyboardInterrupt:
            logging.info('---- ! Stopping due to KeyboardInterrupt ! ----')
        else:
            logging.info('----- Finished -----')

        acc_of_label = correct_of_label / total_of_label
        acc_of_label_names = '\n'.join(['{}: {}'.format(l, acc.item())
                                       for l, acc in zip(hub.all_labels,
                                                         acc_of_label)])

        logging.info('Accuracy of individual labels:\n{}'
                     .format(acc_of_label_names))
        logging.info('Total accuracy: {}'.format(total_acc.item()))


def _normalize(images):
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    return images.sub_(mean).div_(std)


def test_resnet18_top_k_accuracy(caplog, resnet18_evaluation, k=5):
    hub, resnet18, val_loader = resnet18_evaluation

    def _eval(images, labels):
        _normalize(images)
        probs = resnet18.predict_probabilities(images)
        _, predicted_labels = torch.topk(probs, k, -1)
        true_labels = labels.unsqueeze(-1).expand_as(predicted_labels)
        success = (predicted_labels == true_labels).sum(-1)

        for pred_no, label_id in enumerate(labels):
            yield label_id, bool(success[pred_no].item())

    _evaluate_on_validation_set(caplog, hub, val_loader, _eval)


def test_resnet18_indoor_outdoor_accuracy(caplog, resnet18_evaluation, k=10):
    hub, resnet18, val_loader = resnet18_evaluation

    def _eval(images_batch, labels_batch):
        _normalize(images_batch)
        probs_batch = resnet18.predict_probabilities(images_batch)
        _, k_predicted_labels_batch = torch.topk(probs_batch, k, -1)

        for true_label_id, k_predicted_labels \
                in zip(labels_batch, k_predicted_labels_batch):
            predict_outdoor = hub.vote_indoor_outdoor(k_predicted_labels)
            is_outdoor = hub.is_outdoor(true_label_id)

            logging.info('Scene "{}" is {}. Predicted as {}.'
                         .format(hub.label_name(true_label_id), is_outdoor,
                                 predict_outdoor))

            yield true_label_id, predict_outdoor == is_outdoor

    _evaluate_on_validation_set(caplog, hub, val_loader, _eval)


def test_preprocessing_white_box():
    hub = Places365Hub()

    with hub.open_demo_image() as demo_file:
        image_1 = Image.open(demo_file)
        image_tensor_1 = wideresnet.IMAGE_TRANSFORM(image_1).unsqueeze(0)

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
        _normalize(image_tensor_2)

        max_diff = (image_tensor_1 - image_tensor_2).max()
        diff_image_tensor = (image_tensor_1 - image_tensor_2)\
            .abs().div(max_diff).squeeze()
        diff_image = to_pil_image(diff_image_tensor, 'RGB').convert('L')
        diff_image.show()

        mean_diff = (image_tensor_1 - image_tensor_2).mean()

        assert mean_diff < 0.005
