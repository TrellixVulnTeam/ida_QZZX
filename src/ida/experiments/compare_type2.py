import argparse
import random
import logging
from typing import Iterable, Optional, Tuple

import torch.nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from ida.experiments.experiment import Experiment, run_experiments
from ida.interpret.colors import PerceivableColorsInterpreter
from ida.interpret.common import JoinedInterpreter, Interpreter
from ida.interpret.ground_truth_objects import GroundTruthObjectsInterpreter
from ida.torch_extensions.classifier import TorchImageClassifier
from ida.type1.tree import TreeType1Explainer
from ida.type2.common import Type2Explainer, NoType2Explainer
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
    # quantile levels were determined with the script `optimize_quantiles.py` from the same directory
    yield GuidedGradCamType2Explainer(classifier=classifier,
                                      interpreter=interpreter,
                                      layer=layer,
                                      quantile_level=0.05)

    yield SaliencyType2Explainer(classifier=classifier,
                                 interpreter=interpreter,
                                 quantile_level=0.5)

    yield IntegratedGradientsType2Explainer(classifier=classifier,
                                            interpreter=interpreter,
                                            quantile_level=0.1)

    yield DeepLiftType2Explainer(classifier=classifier,
                                 interpreter=interpreter,
                                 quantile_level=0.2)


def get_classifiers() -> Iterable[Tuple[TorchImageClassifier, Optional[torch.nn.Module]]]:
    alexnet = TorchImageClassifier.from_json_file('places365_alexnet.json', memoize=True)
    resnet18 = TorchImageClassifier.from_json_file('places365_resnet18.json', memoize=True)
    yield alexnet, alexnet.torch_module.features[10]
    yield resnet18, resnet18.torch_module.layer4[1].conv2


def get_num_train_obs() -> Iterable[int]:
    yield from [500, 1000, 5000]  # , 10000, 20000]


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

            no_type_2 = NoType2Explainer(classifier=classifier,
                                         interpreter=interpreter)
            yield get_experiment(images_url=images_url,
                                 num_train_obs=num_train_obs,
                                 type2=no_type_2,
                                 model_agnostic_picker='passthrough',
                                 param_grid=[approximate_grid],
                                 cv_params={'n_jobs': 62})

            rf_picker = SelectFromModel(estimator=ExtraTreesClassifier(random_state=SEED))
            rf_grid = {'pick_agnostic__threshold': ['mean'],  # was always better than median
                       'pick_agnostic__estimator__n_estimators': [250],
                       'pick_agnostic__estimator__criterion': ['gini'],
                       'pick_agnostic__estimator__max_depth': [35],  # very important to keep memory consumption down
                       'pick_agnostic__estimator__min_impurity_decrease': [0.0001]}
            yield get_experiment(images_url=images_url,
                                 num_train_obs=num_train_obs,
                                 type2=no_type_2,
                                 model_agnostic_picker=rf_picker,
                                 param_grid=[rf_grid],
                                 cv_params={'n_jobs': 62})


if __name__ == '__main__':
    description = ('Compare different Type2 explainers by how well they help '
                   'to select concepts for training surrogate models.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--images_url')
    args = parser.parse_args()

    run_experiments(name='2021-12-22-00:36:54 type2_vs_baselines-varying_train_obs',
                    prepend_timestamp=False,
                    continue_previous_run=True,
                    description=description,
                    experiments=get_experiments(images_url=args.images_url))
