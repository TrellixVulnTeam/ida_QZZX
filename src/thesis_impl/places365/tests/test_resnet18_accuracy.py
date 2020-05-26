import logging

import pytest
import torch
from petastorm import make_reader
from petastorm.pytorch import DataLoader

from thesis_impl.places365.hub import Places365Hub


@pytest.fixture
def resnet18_evaluation(supervised_url):
    hub = Places365Hub()
    resnet18 = hub.resnet18()
    reader = make_reader(supervised_url, schema_fields=['image', 'label_id'])
    # we set batch size to sth > 1
    # to not have the first tensor dimension removed
    return hub, resnet18, DataLoader(reader, batch_size=2)


def evaluate_accuracy(caplog, hub, loader, method):
    outer_level = logging.getLogger().level

    with caplog.at_level(logging.INFO):
        correct_of_label = torch.zeros(len(hub.label_names))
        total_of_label = torch.zeros(len(hub.label_names))

        num_processed = 0
        total_acc = 0

        try:
            with torch.no_grad():
                for batch in loader:
                    images = batch['image'].permute(0, 3, 1, 2) \
                        .to(torch.float32).div(255.)
                    labels = batch['label_id']
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
                                       for l, acc in zip(hub.label_names,
                                                         acc_of_label)])

        logging.info('Accuracy of individual labels:\n{}'
                     .format(acc_of_label_names))
        logging.info('Total accuracy: {}'.format(total_acc.item()))


def test_top_k_accuracy(caplog, resnet18_evaluation, k=5):
    hub, resnet18, loader = resnet18_evaluation

    def _eval(images, labels):
        probs = resnet18(images)
        _, predicted_labels = torch.topk(probs, k, -1)
        true_labels = labels.unsqueeze(-1).expand_as(predicted_labels)
        success = (predicted_labels == true_labels).sum(-1)

        for pred_no, label_id in enumerate(labels):
            yield label_id, bool(success[pred_no].item())

    evaluate_accuracy(caplog, hub, loader, _eval)


def test_indoor_outdoor_accuracy(caplog, resnet18_evaluation, k=10):
    hub, resnet18, loader = resnet18_evaluation

    def _eval(images_batch, labels_batch):
        probs_batch = resnet18(images_batch)
        _, k_predicted_labels_batch = torch.topk(probs_batch, k, -1)

        for true_label_id, k_predicted_labels \
                in zip(labels_batch, k_predicted_labels_batch):
            predict_outdoor = hub.vote_indoor_outdoor(k_predicted_labels)
            is_outdoor = hub.is_outdoor(true_label_id)

            logging.info('Scene "{}" is {}. Predicted as {}.'
                         .format(hub.label_name(true_label_id), is_outdoor,
                                 predict_outdoor))

            yield true_label_id, predict_outdoor == is_outdoor

    evaluate_accuracy(caplog, hub, loader, _eval)
