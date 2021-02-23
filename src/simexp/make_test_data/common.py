from dataclasses import field, dataclass

import numpy as np
import pyspark.sql.types as st
import torch
from petastorm.codecs import ScalarCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import DataFrame

from simexp.common import Classifier
from simexp.spark import Field, ConceptMasksUnion, SparkSessionConfig, DataGenerator


@dataclass
class TestDataGenerator(ConceptMasksUnion, DataGenerator):

    # spark session for creating the result dataframe
    spark_cfg: SparkSessionConfig

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

    def __getstate__(self):
        # note: we only return the state necessary for the method `_process_row`. other attributes will be lost.
        return self.all_concept_names, self.classifier, self.output_schema

    def __setstate__(self, state):
        self.all_concept_names, self.classifier, self.output_schema = state

    def _process_row(self, image_row):
        image = Field.IMAGE.decode(image_row[Field.IMAGE.name])
        image_id = Field.IMAGE_ID.decode(image_row[Field.IMAGE_ID.name])

        counts = np.zeros((len(self.all_concept_names, )), dtype=np.uint8)

        # each image has multiple arrays of concept names from different describers
        for concept_names in image_row[Field.CONCEPT_NAMES.name]:
            for concept_name in concept_names:
                counts[self.all_concept_names.index(concept_name)] += 1

        pred = np.uint16(np.argmax(self.classifier.predict_proba(np.expand_dims(image, 0))[0]))

        return {Field.PREDICTED_CLASS.name: pred,
                Field.IMAGE_ID.name: image_id,
                **dict(zip(self.all_concept_names, counts))}

    def to_df(self) -> DataFrame:
        rdd = self.joined_df.rdd.coalesce(torch.cuda.device_count()) \
            .map(self._process_row) \
            .map(lambda r: dict_to_spark_row(self.output_schema, r))
        return self.spark_cfg.session.createDataFrame(rdd, self.output_schema.as_spark_schema())
