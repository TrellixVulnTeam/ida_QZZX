from dataclasses import field, dataclass
from typing import Iterator

import numpy as np
import pyspark.sql.types as st
from petastorm.codecs import ScalarCodec
from petastorm.unischema import Unischema, UnischemaField

from simexp.common import Classifier, RowDict
from simexp.spark import DictBasedDataGenerator, Field, ConceptMasksUnion


@dataclass
class TestDataGenerator(ConceptMasksUnion, DictBasedDataGenerator):

    # name of this generator
    name: str = field(default='test_data', init=False)

    # the output schema of this data generator.
    output_schema: Unischema = field(default=None, init=False)

    # a classifier
    classifier: Classifier

    # url to a petastorm parquet store of schema `Schema.IMAGE`
    images_url: str

    def __post_init__(self):
        super().__post_init__()

        pred_fields = [Field.IMAGE_ID, Field.PREDICTED_CLASS]
        concept_fields = [UnischemaField(concept_name, np.uint8, (), ScalarCodec(st.IntegerType()), False)
                          for concept_name in self.all_concept_names]
        self.output_schema = Unischema('ConceptCounts', pred_fields + concept_fields)

        images_df = self.spark_cfg.session.read.parquet(self.images_url)
        self.joined_df = self.union_df.join(images_df, on=Field.IMAGE_ID.name, how='inner')

    def generate(self) -> Iterator[RowDict]:
        for per_image_row in self.joined_df.collect():
            image = Field.IMAGE.decode(per_image_row[Field.IMAGE.name])
            image_id = Field.IMAGE_ID.decode(per_image_row[Field.IMAGE_ID.name])

            with self._log_task('Processing image {}'.format(image_id)):
                counts = np.zeros((len(self.all_concept_names, )), dtype=np.uint8)

                # each image has multiple arrays of concept names from different describers
                for concept_names in per_image_row[Field.CONCEPT_NAMES.name]:
                    for concept_name in concept_names:
                        counts[self.all_concept_names.index(concept_name)] += 1

                pred = np.uint16(np.argmax(self.classifier.predict_proba(np.expand_dims(image, 0))[0]))

                yield {Field.PREDICTED_CLASS.name: pred,
                       Field.IMAGE_ID.name: image_id,
                       **dict(zip(self.all_concept_names, counts))}
