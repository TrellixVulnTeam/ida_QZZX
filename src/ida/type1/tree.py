import base64
from typing import Dict, Any

import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.validation import check_is_fitted

from ida.experiments.experiment import get_experiment_row
from ida.type1.common import Type1Explainer


class TreeType1Explainer(Type1Explainer):

    @staticmethod
    def _create_pipeline(tree: DecisionTreeClassifier):
        return Pipeline([
            ('tree', tree)
        ])

    @staticmethod
    def create_pipeline(random_state: int) -> Pipeline:
        return TreeType1Explainer._create_pipeline(DecisionTreeClassifier(random_state=random_state))

    @staticmethod
    def get_complexity_metrics(pipeline: Pipeline) -> Dict[str, Any]:
        tree = pipeline['tree']
        return {'tree_n_leaves': tree.get_n_leaves(),
                'tree_depth': tree.get_depth()}

    @staticmethod
    def get_fitted_params(pipeline: Pipeline) -> Dict[str, Any]:
        tree = pipeline['tree']
        check_is_fitted(tree)
        attrs = [v for v in vars(tree) if v.endswith("_") and not v.startswith("__")]
        return {attr: getattr(tree, attr) for attr in attrs}

    @staticmethod
    def serialize(pipeline: Pipeline) -> Dict[str, Any]:
        tree = pipeline['tree']
        return {'encoded_tree': base64.b64encode(pickle.dumps(tree))}

    @staticmethod
    def _load_tree(experiment_name: str, exp_no: int, rep_no: int):
        row = get_experiment_row(experiment_name=experiment_name, exp_no=exp_no, rep_no=rep_no)
        assert 'encoded_tree' in row, 'The given experiment did not use a TreeType1Explainer.'
        tree: DecisionTreeClassifier = pickle.loads(base64.b64decode(row['encoded_tree']))
        return row, tree

    @staticmethod
    def load(experiment_name: str, exp_no: int, rep_no: int) -> Pipeline:
        _, tree = TreeType1Explainer._load_tree(experiment_name=experiment_name, exp_no=exp_no, rep_no=rep_no)
        return TreeType1Explainer._create_pipeline(tree=tree)

    @staticmethod
    def plot(experiment_name: str, exp_no: int, rep_no: int):
        row, tree = TreeType1Explainer._load_tree(experiment_name=experiment_name, exp_no=exp_no, rep_no=rep_no)
        plot_tree(decision_tree=tree,
                  class_names=np.asarray(row['class_names'])[tree.classes_],
                  feature_names=np.asarray(row['concept_names'])[row['picked_concepts_']])

    def __str__(self):
        return 'TreeType1Explainer'
