import logging

import torchvision
import torch
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


def test_resnet18_accuracy(caplog, validation_path):
    caplog.set_level(logging.INFO)

    hub = Places365Hub()
    resnet18 = hub.resnet18()

    val = torchvision.datasets.ImageFolder(root=validation_path,
                                           transform=resnet18.transform_image)
    val_loader = torch.utils.data.DataLoader(val, batch_size=512,
                                             num_workers=6,
                                             pin_memory=False,
                                             shuffle=True)

    correct_of_label = torch.zeros(len(hub.all_labels))
    total_of_label = torch.zeros(len(hub.all_labels))

    num_processed = 0
    total_acc = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            probs = resnet18.predict_probabilities(images)
            _, predicted = torch.max(probs, -1)
            c = (predicted == labels).squeeze()

            for img_count, label_id in enumerate(labels):
                correct_of_label[label_id] += c[img_count].item()
                total_of_label[label_id] += 1

                num_processed += 1

            logging.info('Processed {} images so far.'.format(num_processed))
            total_acc = (correct_of_label.sum() / total_of_label.sum())\
                .squeeze()
            logging.info('Current accuracy is {}.'.format(total_acc.item()))

            if num_processed >= 1024:
                break

    acc_of_label = correct_of_label / total_of_label
    acc_of_label_names = '\n'.join(['{}: {}'.format(l, acc.item())
                                   for l, acc in zip(hub.all_labels,
                                                     acc_of_label)])

    logging.info('----- FINISHED PROCESSING VALIDATION DATA -----')
    logging.info('Accuracy of individual labels:\n{}'
                 .format(acc_of_label_names))
    logging.info('Total accuracy: {}'.format(total_acc.item()))

    assert total_acc > 0.7


def test_resnet18_top_k_accuracy(caplog, validation_path, k=5):
    caplog.set_level(logging.INFO)

    hub = Places365Hub()
    resnet18 = hub.resnet18()

    val = torchvision.datasets.ImageFolder(root=validation_path,
                                           transform=resnet18.transform_image)
    val_loader = torch.utils.data.DataLoader(val, batch_size=16,
                                             num_workers=6,
                                             pin_memory=False,
                                             shuffle=True)

    correct_of_label = torch.zeros(len(hub.all_labels))
    total_of_label = torch.zeros(len(hub.all_labels))

    num_processed = 0
    total_acc = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            probs = resnet18.predict_probabilities(images)
            _, predicted = torch.topk(probs, k, -1)
            k_labels = labels.unsqueeze(-1).expand_as(predicted)
            c = (predicted == k_labels).sum(-1)

            for img_count, label_id in enumerate(labels):
                correct_of_label[label_id] += c[img_count].item()
                total_of_label[label_id] += 1

                num_processed += 1

            logging.info('Processed {} images so far.'.format(num_processed))
            total_acc = (correct_of_label.sum() / total_of_label.sum())\
                .squeeze()
            logging.info('Current accuracy is {}.'.format(total_acc.item()))

            if num_processed >= 1024:
                break

    acc_of_label = correct_of_label / total_of_label
    acc_of_label_names = '\n'.join(['{}: {}'.format(l, acc.item())
                                   for l, acc in zip(hub.all_labels,
                                                     acc_of_label)])

    logging.info('----- FINISHED PROCESSING VALIDATION DATA -----')
    logging.info('Accuracy of individual labels:\n{}'
                 .format(acc_of_label_names))
    logging.info('Total accuracy: {}'.format(total_acc.item()))

    assert total_acc > 0.7
