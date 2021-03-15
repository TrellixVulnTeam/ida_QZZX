from dataclasses import dataclass
from typing import Literal

import pandas as pd
from plotnine import *

from simexp.fit import SurrogatesFitter

clear_theme = (theme_bw() +
               theme(panel_border=element_blank(),
                     panel_grid=element_blank(),
                     panel_grid_major=element_blank(),
                     panel_grid_minor=element_blank(),
                     text=element_text(wrap=True,
                                       family='Latin Modern Roman',
                                       fontstretch='normal',
                                       fontweight='light',
                                       size=16,
                                       colour='black'),
                     plot_title=element_text(size=20, fontweight='normal'),
                     axis_title=element_text(size=14, fontweight='normal'),
                     line=element_line(colour='black', size=.5),
                     axis_ticks=element_blank(),
                     strip_text_x=element_text(size=10),
                     strip_background=element_blank(),
                     legend_key=element_blank()))


@dataclass
class SurrogatesResultPlotter:

    # the results to plot
    results: SurrogatesFitter.Results

    @property
    def df(self) -> pd.DataFrame:
        return self.results.to_flat_pandas()

    @staticmethod
    def _extract_ie_names_and_params(df: pd.DataFrame):
        return df['influence_estimator'] \
            .str.extract(r'^(?P<influence_estimator_name>.*)InfluenceEstimator'
                         r'\((?P<influence_estimator_params>.*)\)$',
                         expand=True) \
            .fillna('None')

    @staticmethod
    def _get_k(df: pd.DataFrame):
        assert df['top_k'].nunique() == 1, 'cannot merge top-k accuracies with different k'
        return df['top_k'][0]

    def plot_accuracy_per_influence_estimator(self, metric: str = 'top_k_accuracy'):
        max_indices = self.df.groupby(by='influence_estimator')[metric].idxmax()
        df = self.df.loc[max_indices]
        df['hyperparameters'] = df.apply(lambda x: 'No perturbation' if x.perturber == 'none'
                                         else '{},\n{}'.format(x.perturber, x.detector), axis=1)
        if metric == 'top_k_accuracy':
            title = 'Highest Top-{}-Accuracy Per Influence Estimator'.format(self._get_k(df))
        else:
            title = 'Lowest Cross-Entropy Per Influence Estimator'

        df = pd.concat([df, self._extract_ie_names_and_params(df)], axis=1)

        return (ggplot(df, aes('influence_estimator_name')) +
                clear_theme +
                geom_col(aes(y=metric, fill='hyperparameters')) +
                ggtitle(title) +
                labs(x='Pixel Influence Estimator', fill='Perturbation parameters') +
                theme(axis_title_x=element_blank(),
                      axis_title_y=element_blank(),
                      axis_text_x=element_text(angle=-45, hjust=0, vjust=1),
                      legend_title=element_text(margin={'b': 10}),
                      legend_entry_spacing=5) +
                scale_fill_brewer(type='qual', palette='Paired'))

    def plot_accuracy_per_perturb_fraction(self,
                                           metric: Literal['top_k_accuracy',
                                                           'cross_entropy'] = 'top_k_accuracy'):
        df = pd.concat([self.df, self._extract_ie_names_and_params(self.df)], axis=1)
        df['hyperparameters'] = df.apply(lambda x: '{}\n{}\n{}'.format(x.influence_estimator_name,
                                                                       x.perturber, x.detector),
                                         axis=1)

        if metric == 'top_k_accuracy':
            metric_in_title = 'Top-{}-Accuracy'.format(self._get_k(df))
        else:
            metric_in_title = 'Cross-Entropy'

        return (ggplot(df, aes('perturb_fraction')) +
                clear_theme +
                geom_path(aes(y=metric, fill='hyperparameters')) +
                ggtitle('{} per Fraction of Perturbed Images'.format(metric_in_title)))
