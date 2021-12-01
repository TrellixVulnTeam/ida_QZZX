import base64
from typing import Dict, Any

import numpy as np
import pickle

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from ida.experiments.experiment import get_experiment_df
from ida.type1.common import CrossValidatedType1Explainer


class TreeType1Explainer(CrossValidatedType1Explainer):

    def get_pipeline(self, random_state: int) -> Pipeline:
        pipeline = Pipeline([
            ('sel', SelectFromModel(ExtraTreesClassifier(random_state=random_state))),
            ('tree', DecisionTreeClassifier(random_state=random_state))
        ])
        return pipeline

    @staticmethod
    def get_complexity_metrics(pipeline: Pipeline, **kwargs) -> Dict[str, Any]:
        tree = pipeline['tree']
        return {'tree_n_leaves': tree.get_n_leaves(),
                'tree_depth': tree.get_depth()}

    @staticmethod
    def get_fitted_params(pipeline: Pipeline) -> Dict[str, Any]:
        return {k: v for (k, v) in pipeline.get_params().items()
                if k != 'sel' and k.startswith('sel')
                or k != 'tree' and k.startswith('tree')}

    @staticmethod
    def get_plot_representation(pipeline: Pipeline) -> Dict[str, Any]:
        sel = pipeline['sel']
        tree = pipeline['tree']

        if isinstance(sel, SelectFromModel):
            selected_concepts = np.asarray(sel.get_support(), dtype=bool).tobytes()
        else:  # this happens if one sets sel = 'passthrough' in the pipeline
            selected_concepts = np.ones(pipeline.n_features_in_, dtype=bool).tobytes()

        return {'selected_concepts': selected_concepts,
                'encoded_tree': base64.b64encode(pickle.dumps(tree)),
                'seen_classes': np.asarray(tree.classes_, dtype=int).tobytes()}

    @staticmethod
    def plot(experiment_name: str, exp_no: int, rep_no: int, **kwargs):
        assert 'concepts' in kwargs and 'all_classes' in kwargs
        concepts = kwargs.pop('concepts')
        all_classes = kwargs.pop('all_classes')

        df = get_experiment_df(experiment_name)
        row = df.loc[df['exp_no'] == exp_no].loc[df['rep_no'] == rep_no].iloc[0]

        seen_classes = np.fromstring(row['seen_classes'], dtype=int)
        clf = pickle.loads(base64.b64decode(row['encoded_tree']))
        support = np.fromstring(row['selected_concepts'], dtype=bool)
        plot_tree(decision_tree=clf,
                  feature_names=np.asarray(concepts)[support],
                  class_names=np.asarray(all_classes)[seen_classes],
                  **kwargs)

    def __str__(self):
        return 'TreeType1Explainer'
