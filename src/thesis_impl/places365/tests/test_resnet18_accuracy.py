import logging

import pytest
import torch

from thesis_impl.places365.tests.evaluate import evaluate_on_validation_set


@pytest.mark.usefixtures('resnet18_evaluation')
def test_top_k_accuracy(caplog, resnet18_evaluation, k=5):
    hub, resnet18, val_loader = resnet18_evaluation

    def _eval(images, labels):
        probs = resnet18.predict_probabilities(images)
        _, predicted_labels = torch.topk(probs, k, -1)
        true_labels = labels.unsqueeze(-1).expand_as(predicted_labels)
        success = (predicted_labels == true_labels).sum(-1)

        for pred_no, label_id in enumerate(labels):
            yield label_id, bool(success[pred_no].item())

    evaluate_on_validation_set(caplog, hub, val_loader, _eval)


@pytest.mark.usefixtures('resnet18_evaluation')
def test_indoor_outdoor_accuracy(caplog, resnet18_evaluation, k=10):
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

    evaluate_on_validation_set(caplog, hub, val_loader, _eval)
