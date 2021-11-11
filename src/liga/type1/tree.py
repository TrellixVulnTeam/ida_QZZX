from collections import Counter
from typing import List, Union, Dict, Any, Tuple, Optional
import itertools as it

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from liga.type1.common import Type1Explainer


class TreeType1Explainer(Type1Explainer[GridSearchCV]):

    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 all_classes: Optional[List[int]],
                 random_state: int = 42,
                 min_k_folds: int = 2,
                 top_k_acc: int = 5,
                 n_jobs: int = 10,
                 pre_dispatch: Union[int, str] = 'n_jobs',
                 scoring: str = 'neg_log_loss',
                 **fit_params) -> GridSearchCV:
        tree = DecisionTreeClassifier(random_state=random_state)

        concept_counts = np.asarray(concept_counts)
        predicted_classes = np.asarray(predicted_classes)

        indices = self.filter_for_split(predicted_classes=predicted_classes,
                                        min_k_folds=min_k_folds,
                                        all_classes=all_classes)
        k_folds = self.determine_k_folds(predicted_classes[indices])

        search = GridSearchCV(estimator=tree,
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
    def determine_k_folds(predicted_classes: [int]):
        return Counter(predicted_classes).most_common()[-1][1]

    @staticmethod
    def get_complexity_metrics(model: GridSearchCV, **kwargs) -> Dict[str, Any]:
        return {'tree_n_leaves': model.best_estimator_.get_n_leaves(),
                'tree_depth': model.best_estimator_.get_depth()}

    @staticmethod
    def get_fitted_params(model: GridSearchCV) -> Dict[str, Any]:
        return model.best_params_

    def __str__(self):
        return 'tree'
