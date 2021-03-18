from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.font_manager as font_manager
import pandas as pd
from plotnine import *

from simexp.fit import SurrogatesFitter

font_dirs = list((Path(__file__).parent.parent / 'fonts').iterdir())
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)


clear_theme = (theme_bw() +
               theme(panel_border=element_blank(),
                     panel_grid=element_blank(),
                     panel_grid_major=element_blank(),
                     panel_grid_minor=element_blank(),
                     text=element_text(wrap=True,
                                       family='Times New Roman',
                                       size=16,
                                       colour='black'),
                     plot_title=element_text(size=20, fontweight='normal'),
                     axis_title=element_text(size=14, fontweight='normal'),
                     line=element_line(colour='black', size=.5),
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
        df['hyperparameters'] = df.apply(lambda x: 'No perturbation' if x.perturber == 'None'
                                         else '{},\n{}'.format(x.perturber, x.detector), axis=1)
        if metric == 'top_k_accuracy':
            title = 'Highest Top-{}-Accuracy Per Influence Estimator'.format(self._get_k(df))
        else:
            title = 'Lowest Cross-Entropy Per Influence Estimator'

        df = pd.concat([df, self._extract_ie_names_and_params(df)], axis=1)

        return (ggplot(df, aes('influence_estimator_name')) +
                clear_theme +
                geom_col(aes(y=metric, fill='hyperparameters')) +
                expand_limits(y=0) +
                ggtitle(title) +
                labs(x='Pixel Influence Estimator', fill='Perturbation parameters') +
                theme(axis_title_x=element_blank(),
                      axis_title_y=element_blank(),
                      axis_text_x=element_text(angle=-45, hjust=0, vjust=1),
                      legend_title=element_text(margin={'b': 10}),
                      legend_entry_spacing=5) +
                scale_fill_brewer(type='qual', palette='Paired'))

    def plot_accuracy_per_perturb_fraction(self, metric: str = 'top_k_accuracy',
                                           breaks: Optional[List[float]] = None):
        df = pd.concat([self.df, self._extract_ie_names_and_params(self.df)], axis=1)
        df.hyperparameters = df.apply(lambda x: '{}\n{}\n{}'.format(x.influence_estimator_name,
                                                                       x.perturber, x.detector),
                                      axis=1)

        if metric == 'top_k_accuracy':
            metric_in_title = 'Top-{}-Accuracy'.format(self._get_k(df))
        else:
            metric_in_title = 'Cross-Entropy'

        x_args = {} if breaks is None else {'breaks': breaks}

        if self.results.train_obs_balanced:
            s = df.apply(lambda x: '{} (≤{} per class)'.format(x.train_obs_count,
                                                               x.train_obs_per_class_threshold),
                         axis=1)
            df.train_obs_count = pd.Categorical(s, ordered=True, categories=s.unique())  # ensure correct ordering

        return (ggplot(df, aes(x='perturb_fraction', y=metric)) +
                clear_theme +
                geom_path(aes(color='train_obs_count')) +
                geom_point(aes(color='train_obs_count')) +
                scale_x_continuous(**x_args) +
                expand_limits(y=0) +
                ggtitle('{} by Fraction of Perturbed Images'.format(metric_in_title)) +
                labs(x='Fraction', y=metric_in_title, color='Number of training images') +
                scale_fill_brewer(type='qual', palette='Paired', direction=-1))
