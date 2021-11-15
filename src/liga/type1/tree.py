import base64
from collections import Counter
from typing import List, Union, Dict, Any, Tuple, Optional, Iterable
import itertools as it

import ast
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import top_k_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from liga.experiments.experiment import get_experiment_df
from liga.type1.common import Type1Explainer


class TreeType1Explainer(Type1Explainer[GridSearchCV]):

    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 all_classes: Optional[List[int]],
                 random_state: int = 42,
                 min_k_folds: int = 5,
                 max_k_folds: int = 5,
                 top_k_acc: int = 5,
                 n_jobs: int = 10,
                 pre_dispatch: Union[int, str] = 'n_jobs',
                 scoring: Optional[str] = None,
                 select_top_k_influential_concepts: int = 30,
                 **fit_params) -> GridSearchCV:
        if scoring is None:
            scoring = make_scorer(top_k_accuracy_score, k=top_k_acc)

        selector = SelectFromModel(ExtraTreesClassifier(random_state=random_state))
        tree = DecisionTreeClassifier(random_state=random_state)
        pipeline = Pipeline([
            ('sel', selector),
            ('clf', tree)
        ])

        concept_counts = np.asarray(concept_counts)
        predicted_classes = np.asarray(predicted_classes)

        assert min_k_folds <= max_k_folds
        indices = self.filter_for_split(predicted_classes=predicted_classes,
                                        min_k_folds=min_k_folds,
                                        all_classes=all_classes)
        k_folds = self.determine_k_folds(predicted_classes[indices], max_k_folds)

        search = GridSearchCV(estimator=pipeline,
                              param_grid=fit_params,
                              cv=StratifiedKFold(n_splits=k_folds),
                              n_jobs=n_jobs,
                              pre_dispatch=pre_dispatch,
                              scoring=scoring)
        return search.fit(concept_counts[indices],
                          predicted_classes[indices])

    @staticmethod
    def class_counts(predicted_classes, all_classes) -> [Tuple[str, int]]:
        """
        Returns tuples of class ids and corresponding counts of observations in the training data.
        """
        counts = Counter(it.chain(predicted_classes, all_classes))
        m = 0 if all_classes is None else 1
        return sorted(((s, c - m) for s, c in counts.items()),
                      key=lambda x: (1.0 / (x[1] + 1), x[0]))

    def filter_for_split(self, predicted_classes, min_k_folds, all_classes):
        """
        Filters the training data so that all predicted classes appear at least
        `ceil(security_factor * num_splits)` times.
        Use this to keep only classes where at least one prediction for each CV split is available.
        """
        enough_samples = list(s for s, c in self.class_counts(predicted_classes, all_classes) if c >= min_k_folds)
        return np.isin(predicted_classes, enough_samples)

    @staticmethod
    def determine_k_folds(predicted_classes: [int], max_k_folds: int):
        return min(max_k_folds, Counter(predicted_classes).most_common()[-1][1])

    @staticmethod
    def get_complexity_metrics(model: GridSearchCV, **kwargs) -> Dict[str, Any]:
        clf = model.best_estimator_['clf']
        return {'tree_n_leaves': clf.get_n_leaves(),
                'tree_depth': clf.get_depth()}

    @staticmethod
    def get_fitted_params(model: GridSearchCV) -> Dict[str, Any]:
        return {**model.best_params_,
                'cv_folds': model.n_splits_}

    @staticmethod
    def get_plot_representation(model: GridSearchCV) -> Dict[str, Any]:
        sel = model.best_estimator_['sel']
        clf = model.best_estimator_['clf']
        return {'selected_concepts': np.asarray(sel.get_support(), dtype=bool).tobytes(),
                'encoded_tree': base64.b64encode(pickle.dumps(clf))}

    @staticmethod
    def plot(experiment_name: str, exp_no: int, rep_no: int, **kwargs):
        assert 'concepts' in kwargs and 'all_classes' in kwargs
        concepts = kwargs.pop('concepts')
        all_classes = kwargs.pop('all_classes')

        df = get_experiment_df(experiment_name)
        row = df.loc[df['exp_no'] == exp_no and df['rep_no'] == rep_no].iloc[0]

        encoded_tree = ast.literal_eval(row['encoded_tree'])
        selected_concepts = ast.literal_eval(row['selected_concepts'])
        clf = pickle.loads(base64.b64decode(encoded_tree))
        support = np.fromstring(selected_concepts, dtype=bool)
        plot_tree(decision_tree=clf,
                  feature_names=np.asarray(concepts)[support],
                  class_names=all_classes,
                  **kwargs)

    def __str__(self):
        return 'tree'
