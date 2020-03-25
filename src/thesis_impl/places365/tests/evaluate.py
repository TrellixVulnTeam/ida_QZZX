import logging

import pytest
import torch
from petastorm import make_reader
from petastorm.pytorch import DataLoader

from thesis_impl.places365.hub import Places365Hub

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('Torch is using device {}.'.format(_DEVICE))


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


def evaluate_on_validation_set(caplog, hub, val_loader, method):
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
