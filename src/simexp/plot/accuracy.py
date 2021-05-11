from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
import pandas as pd
from plotnine import *

from simexp.fit import SurrogatesFitter

font_dirs = [p for p in (Path(__file__).parent.parent / 'fonts').iterdir() if p.is_dir()]
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

mpl.rcParams['mathtext.fontset'] = 'stix'

clear_theme = (theme_bw() +
               theme(panel_border=element_blank(),
                     panel_grid=element_blank(),
                     panel_grid_major=element_blank(),
                     panel_grid_minor=element_blank(),
                     text=element_text(wrap=False,
                                       family='Nimbus Roman No9 L',
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

    def _get_normalized_df(self, metric: str, normalization: str):
        assert metric in ['top_k_accuracy', 'cross_entropy']
        assert normalization in ['none', 'difference', 'ratio']

        df = self.df

        if metric == 'top_k_accuracy':
            metric_in_title = 'Top-{}-Accuracy'.format(self._get_k(df))
            sign = 1
        else:
            metric_in_title = 'Cross Entropy'
            sign = -1

        if normalization == 'difference':
            df[metric] = (df[metric] - df['dummy_{}'.format(metric)]) * sign
        elif normalization == 'ratio':
            df[metric] = (df[metric] / df['dummy_{}'.format(metric)]) ** sign

        df['num_decimals'] = df.apply(lambda x: -np.log10(np.abs(x[metric])).clip(0, None).astype(int) + 2
                                      if x[metric] != 0 else 0, axis=1)
        df['label'] = df.apply(lambda x: '${{:.{}f}}$'.format(x.num_decimals).format(x[metric]), axis=1)

        df['hyperparameters'] = df.apply(lambda x: 'No augmentation' if x.perturber == 'None'
                                         else '$\\mathtt{{{}}}$,\n$\\mathtt{{{}}}$'
                                         .format(x.perturber.replace('_', r'\_'), x.detector.replace('_', r'\_')),
                                         axis=1)
        df = pd.concat([df, self._extract_ie_names_and_params(df)], axis=1)

        return df, metric_in_title

    def plot_best_accuracy_per_influence_estimator(self, metric: str = 'cross_entropy', normalization: str = 'none'):
        """
        Plots a bar chart with one bar per influence estimator.
        Each bar shows the best accuracy reached by the respective estimator,
        across all tested hyperparameter combinations.

        :param metric: which accuracy metric to use, one of 'top_k_accuracy' and 'cross_entropy'
        :param normalization: how to normalize the metric with respect to the dummy baseline.
            Choose 'none' for no normalization, 'difference' for computing the difference,
            and 'ratio' for computing the ratio.
        :return: the generated ggplot
        """
        df, metric_in_title = self._get_normalized_df(metric, normalization)

        if normalization != 'none' or metric == 'top_k_accuracy':
            best_indices = df.groupby(by='influence_estimator')[metric].idxmax()  # max diff or ratio per group
            df['winner'] = df[metric] == df[metric].max()  # global max
        else:
            best_indices = df.groupby(by='influence_estimator')[metric].idxmin()
            df['winner'] = df[metric] == df[metric].min()

        df['influence_estimator_name'] = df.apply(lambda x: '* {}'.format(x.influence_estimator_name)
                                                  if x.winner else x.influence_estimator_name, axis=1)

        df = df.loc[best_indices]

        y_label = r'$\widehat{{A}} [{}]$'.format(metric_in_title) if normalization != 'none' else metric_in_title

        return (ggplot(df, aes(x='influence_estimator_name', y=metric)) +
                clear_theme +
                geom_col(aes(fill='hyperparameters', color='winner')) +
                geom_text(aes(label='label'), size=14, va='bottom') +
                scale_color_manual(values=('#00000000', 'black'), guide=None) +
                expand_limits(y=0) +
                labs(x='Attribution Method', y=y_label, fill='Best augmentation parameters') +
                theme(axis_text_x=element_text(angle=-45, hjust=0, vjust=1),
                      legend_title=element_text(margin={'b': 10}),
                      legend_entry_spacing=5) +
                scale_fill_brewer(type='qual', palette='Paired'))

    def plot_accuracy_per_perturb_fraction(self, metric: str = 'cross_entropy', normalization: str = 'none',
                                           breaks: Optional[List[float]] = None):
        """
        Plots a line chart with one line per training sample.
        The chart shows the accuracy of the influence estimator (y-axis)
        depending on the fraction of perturbed images of the training sample (x-axis).

        :param metric: which accuracy metric to use, one of 'top_k_accuracy' and 'cross_entropy'
        :param normalization: how to normalize the metric with respect to the dummy baseline.
            Choose 'none' for no normalization, 'difference' for computing the difference,
            and 'ratio' for computing the ratio.
        :param breaks: optional list that specifies the x-axis breaks
        :return: the generated ggplot
        """
        df, metric_in_title = self._get_normalized_df(metric, normalization)

        x_args = {} if breaks is None else {'breaks': breaks}

        if self.results.train_obs_balanced:
            s = df.apply(lambda x: '{} ($\leq {}$ per class)'.format(x.train_obs_count,
                                                                     x.train_obs_per_class_threshold),
                         axis=1)
            df.train_obs_count = pd.Categorical(s, ordered=True, categories=s.unique())  # ensure correct ordering

        return (ggplot(df, aes(x='perturb_fraction', y=metric)) +
                clear_theme +
                geom_path(aes(color='train_obs_count')) +
                geom_point(aes(color='train_obs_count')) +
                scale_x_continuous(**x_args) +
                expand_limits(y=0) +
                labs(x='Fraction of Augmented Images', y=metric_in_title, color='Number of training images') +
                scale_fill_brewer(type='qual', palette='Paired', direction=-1))
