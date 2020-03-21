import logging

import pytest
import torch
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from torch.autograd import Variable as V
from PIL import Image

from thesis_impl.places365.hub import Places365Hub


def test_that_resnet18_works():
    hub = Places365Hub()
    resnet18 = hub.resnet18()

    with hub.open_demo_image() as demo_file:
        image = Image.open(demo_file)
        image_tensor = V(resnet18.transform_image(image).unsqueeze(0))
        label_probs = resnet18.predict_probabilities(image_tensor)
        _, label_ids = label_probs.sort(-1, True)

        assert hub.all_labels[label_ids[0]] == 'food court'


@pytest.fixture
def resnet18_evaluation(validation_url, batch_size):
    hub = Places365Hub()
    resnet18 = hub.resnet18()
    reader = make_reader(validation_url,
                         num_epochs=1,
                         shuffle_row_groups=False)
    val_loader = DataLoader(reader, batch_size=batch_size)
    return hub, resnet18, val_loader


def _evaluate_on_validation_set(caplog, hub, val_loader, method):
    outer_level = logging.getLogger().level
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with caplog.at_level(logging.INFO):
        correct_of_label = torch.zeros(len(hub.all_labels))
        total_of_label = torch.zeros(len(hub.all_labels))

        num_processed = 0
        total_acc = 0

        try:
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label_id'].to(device)

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


def test_resnet18_top_k_accuracy(caplog, resnet18_evaluation, k=5):
    hub, resnet18, val_loader = resnet18_evaluation

    def _eval(images, labels):
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
