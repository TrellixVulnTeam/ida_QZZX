import argparse
import random
import logging
from functools import partial
from typing import Iterable, Optional, Tuple

import torch.nn

from ida.experiments.experiment import Experiment, run_experiments
from ida.interpret.colors import PerceivableColorsInterpreter
from ida.interpret.common import JoinedInterpreter, Interpreter
from ida.interpret.ground_truth_objects import GroundTruthObjectsInterpreter
from ida.torch_extensions.classifier import TorchImageClassifier
from ida.type1.tree import TreeType1Explainer
from ida.type2.common import Type2Explainer
from ida.type2.gradient import SaliencyType2Explainer, IntegratedGradientsType2Explainer, DeepLiftType2Explainer, \
    GuidedGradCamType2Explainer
from ida.oiv4.metadata import OIV4MetadataProvider
from ida.places365.metadata import Places365Task

SEED = 2372775441  # seed obtained with np.random.randint(0, 2 ** 32 -1 )
random.seed(SEED)  # set seed so that petastorm read order becomes deterministic

logger = logging.getLogger()
logger.setLevel(logging.INFO)


p365_task = Places365Task()
oiv4_meta = OIV4MetadataProvider()


MAX_CONCEPT_OVERLAP = .4
MAX_PERTURBED_AREA = .6
GREY_RGB = (127., 127., 127.)


def get_interpreter():
    gt_objects_interpreter = GroundTruthObjectsInterpreter(gt_object_provider=oiv4_meta,
                                                           subset='test',
                                                           ignore_images_without_objects=True)
    color_interpreter = PerceivableColorsInterpreter(random_state=SEED)
    return JoinedInterpreter(gt_objects_interpreter,
                             color_interpreter,
                             random_state=SEED,
                             max_perturbed_area=MAX_PERTURBED_AREA,
                             max_concept_overlap=MAX_CONCEPT_OVERLAP)


def get_type2_explainers(classifier: TorchImageClassifier,
                         interpreter: Interpreter,
                         layer: Optional[torch.nn.Module] = None) -> Iterable[Type2Explainer]:
    for quantile_level in [.9, .5, .2, .1, 0.05, 0.01]:
        for type2_cls in [partial(GuidedGradCamType2Explainer,
                                  layer=layer),
                          SaliencyType2Explainer,
                          IntegratedGradientsType2Explainer,
                          DeepLiftType2Explainer]:
            yield type2_cls(classifier=classifier,
                            interpreter=interpreter,
                            quantile_level=quantile_level)


def get_classifiers() -> Iterable[Tuple[TorchImageClassifier, Optional[torch.nn.Module]]]:
    alexnet = TorchImageClassifier.from_json_file('places365_alexnet.json', memoize=True)
    yield alexnet, alexnet.torch_module.features[10]


def get_num_train_obs() -> Iterable[int]:
    yield 1000


def get_num_calibration_obs() -> int:
    return 1000


def get_top_k_acc() -> Iterable[int]:
    yield from range(1, 3 + 1)


def get_num_test_obs() -> int:
    return 5000


def get_repetitions() -> int:
    return 10


def get_experiment(images_url: str,
                   num_train_obs: int,
                   type2: Type2Explainer,
                   model_agnostic_picker,
                   param_grid,
                   cv_params):
    return Experiment(random_state=SEED,
                      repetitions=get_repetitions(),
                      top_k_acc=list(get_top_k_acc()),
                      images_url=images_url,
                      num_train_obs=num_train_obs,
                      num_calibration_obs=get_num_calibration_obs(),
                      num_test_obs=get_num_test_obs(),
                      class_names=p365_task.class_names,
                      type1=TreeType1Explainer(),
                      type2=type2,
                      model_agnostic_picker=model_agnostic_picker,
                      param_grid=param_grid,
                      cv_params=cv_params)


def get_experiments(images_url: str):
    approximate_grid = {'approximate__tree__max_depth': [30],
                        'approximate__tree__ccp_alpha': (.0025, .005, 0.01)}
    interpret_pick_grid = {'interpret-pick__max_num_type2_calls': [5000],
                           'interpret-pick__quantile': [False],
                           'interpret-pick__threshold': [.2, .6, .8]}

    interpreter = get_interpreter()
    for num_train_obs in get_num_train_obs():
        for classifier, last_conv_layer in get_classifiers():
            for type2 in get_type2_explainers(classifier, interpreter, layer=last_conv_layer):
                yield get_experiment(images_url=images_url,
                                     num_train_obs=num_train_obs,
                                     type2=type2,
                                     model_agnostic_picker='passthrough',
                                     param_grid=[{**approximate_grid,
                                                  **interpret_pick_grid}],
                                     cv_params={'n_jobs': 4})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare six quantile levels for each Type2 explainer.')
    parser.add_argument('--images_url')
    args = parser.parse_args()

    run_experiments(name='2022-01-13-17:49:42 all_type2-alexnet-1000_train_obs-varying_quantile_levels',
                    prepend_timestamp=False,
                    continue_previous_run=True,
                    description='we want to determine the best quantile level for each Type 2 explainer',
                    experiments=get_experiments(images_url=args.images_url))
