from typing import List, Union, Dict, Any

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from liga.type1.common import Type1Explainer


class TreeType1Explainer(Type1Explainer[DecisionTreeClassifier]):

    def __call__(self,
                 concept_counts: List[List[int]],
                 predicted_classes: List[int],
                 random_state: int = 42,
                 k_folds: int = 5,
                 top_k_acc: int = 5,
                 n_jobs: int = 62,
                 pre_dispatch: Union[int, str] = 'n_jobs',
                 scoring: str = 'neg_log_loss',
                 **fit_params) -> DecisionTreeClassifier:
        tree = DecisionTreeClassifier(random_state=random_state)
        search = GridSearchCV(estimator=tree,
                              param_grid=fit_params,
                              cv=k_folds,
                              n_jobs=n_jobs,
                              pre_dispatch=pre_dispatch,
                              scoring=scoring)
        return search.fit(concept_counts, predicted_classes)

    def get_complexity_metrics(self, model: DecisionTreeClassifier) -> Dict[str, Any]:
        return {'tree_n_leaves': model.get_n_leaves(),
                'tree_depth': model.get_depth()}
