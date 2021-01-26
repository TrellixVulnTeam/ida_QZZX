from collections import Counter
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from petastorm.codecs import ScalarCodec
from petastorm.unischema import UnischemaField
from pyspark.sql.types import StringType
from simple_parsing import ArgumentParser

from simadv.common import LoggingConfig, TorchConfig, PetastormTransformer, \
    TorchImageClassifier, TorchImageClassifierSerialization

from anchor import anchor_image


@dataclass
class TorchAnchorExplainer(PetastormTransformer):
    classifier_serial: TorchImageClassifierSerialization
    boxes_field_name: str
    torch_cfg: TorchConfig

    min_coverage = 0.1
    # if more than this fraction of an objects bounding box is covered by an anchor,
    # the object is considered relevant for the corresponding prediction of the classifier

    def __post_init__(self):
        # load the classifier from its name
        @dataclass
        class PartialTorchImageClassifier(TorchImageClassifier):
            torch_cfg: TorchConfig = field(default_factory=lambda: self.torch_cfg, init=False)

        self.classifier = PartialTorchImageClassifier.load(self.classifier_serial.path)

    def run(self):
        anchor = anchor_image.AnchorImage()
        with Image.open('/home/renftlem/Downloads/zebra.jpg') as img:
            img = img.convert('RGB')
            explanation = anchor.explain_instance(np.asarray(img),
                                                  self.classifier.predict_proba,
                                                  max_anchor_size=2,
                                                  coverage_samples=5,
                                                  stop_on_first=True,
                                                  tau=0.2,
                                                  delta=0.1,
                                                  batch_size=10)
            # explanation has format (segmentation_mask, [(segment_id, '', mean, [negatives], 0)])
            # where
            #    segmentation mask: assigns each pixel a segment id
            #    segment_id: identifies a segment
            #    mean: ?
            #    negatives: the strongest (?) counter examples, i.e., images where the anchor fails
            segmentation_mask, relevant_segments = explanation

            relevant_mask = np.zeros(segmentation_mask.shape, dtype=int)
            for segment_id, _, _, _, _ in relevant_segments:
                relevant_mask = np.bitwise_or(relevant_mask,
                                              segmentation_mask == segment_id)

            x_min, x_max, y_min, y_max = 0, 960, 0, 1023  # box of object 1
            f1 = np.count_nonzero(relevant_mask[x_min:x_max, y_min:y_max]) / ((x_max - x_min) * (y_max - y_min))

            x_min, x_max, y_min, y_max = 950, 960, 1000, 1023  # box of object 2
            f2 = np.count_nonzero(relevant_mask[x_min:x_max, y_min:y_max]) / ((x_max - x_min) * (y_max - y_min))

            # goal: perturb all objects that do not overlap with mask
            # rule: if mask covers more than 5% of the object, leave it.

            print(f1, f2)
            print()

        with self.read_cfg.make_reader(None) as reader:
            for row in reader:
                row_id = getattr(row, self.id_field.name)
                segmentation_mask, relevant_segments = anchor.explain_instance(row.image, self.classifier.predict_proba)

                relevant_mask = np.zeros(segmentation_mask.shape, dtype=int)
                for segment_id, _, _, _, _ in relevant_segments:
                    relevant_mask = np.bitwise_or(relevant_mask,
                                                  segmentation_mask == segment_id)

                # use multiple explainers here! → factor out

                min_counts = Counter()
                for box in row.boxes:
                    obj_id, x_min, x_max, y_min, y_max = box[:4]
                    coverage = np.count_nonzero(relevant_mask[x_min:x_max, y_min:y_max]) / \
                        ((x_max - x_min) * (y_max - y_min))

                    if coverage > self.min_coverage:
                        min_counts.update(obj_id=1)

                # sample random object counts
                # then perform union of both counters with |

                # mark perturbed rows with their *explainer-id*, to easily compare later on


def main(id_field: UnischemaField):
    """
    Explain an image classifier by representing its decision boundary with visible objects.
    The images are read from a petastorm parquet store. In this parquet store,
    there must be three fields:

      - A field holding a unique id for each image.
        This field is specified by the caller of this method as `id_field`.
      - A field holding image data encoded with the petastorm png encoder.
        This field must be named *image*.
      - A field holding bounding boxes of visible objects on the image.
        This field must be named *boxes*.
    """
    parser = ArgumentParser(description='Explain an image classifier by representing its decision boundary '
                                        'with visible objects.')

    @dataclass  # set the id_field attribute
    class PartialTorchAnchorExplainer(TorchAnchorExplainer):
        id_field: UnischemaField = field(default_factory=lambda: id_field, init=False)

    parser.add_arguments(PartialTorchAnchorExplainer, dest='explainer')
    parser.add_arguments(LoggingConfig, dest='logging')
    args = parser.parse_args()

    args.explainer.run()


if __name__ == '__main__':
    main(UnischemaField('image_id', np.unicode_, (), ScalarCodec(StringType()), False))
