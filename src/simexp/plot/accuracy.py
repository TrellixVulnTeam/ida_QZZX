import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

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
                                       size=10,
                                       colour='black'),
                     plot_title=element_text(size=14, fontweight='normal'),
                     axis_title=element_text(size=10, fontweight='normal'),
                     axis_title_x=element_text(margin={'t': 5}),
                     axis_title_y=element_text(margin={'r': 5}),
                     line=element_line(colour='black', size=.5),
                     strip_text_x=element_text(size=10),
                     strip_background=element_blank(),
                     legend_key=element_blank()))


@dataclass
class SurrogatesResultPlotter:

    DUMMY_NO_IE_NAME = 'No attribution\n+ Dummy model'

    NO_IE_NAME = 'No attribution'

    PARAM_NO_RESAMPLING = 'No resampling'

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
            .fillna(SurrogatesResultPlotter.NO_IE_NAME)

    @staticmethod
    def _formalize_hyperparameters(df_row):
        if df_row.perturber == 'None':
            return SurrogatesResultPlotter.PARAM_NO_RESAMPLING

        if df_row.perturber.startswith('LocalPerturber'):
            resampler = r'Local resampler $\\mathbf{R}_x^\mathrm{loc}$'
            k = int(re.fullmatch(r'^LocalPerturber\(max_perturbations=(?P<k>.*)\)$', df_row.perturber).group('k'))
        elif df_row.perturber.startswith('GlobalPerturber'):
            resampler = r'Global resampler $\\mathbf{R}_x^\mathrm{glob}$'
            k = int(re.fullmatch(r'^GlobalPerturber\(num_perturbations=(?P<k>.*)\)$', df_row.perturber).group('k'))
        else:
            raise RuntimeError('Unknown perturber: {}'.format(df_row.perturber))

        epsilon = float(re.fullmatch(r'^LiftInfluenceDetector\(lift_threshold=(?P<e>.*)\)$', df_row.detector)
                        .group('e'))

        return '{};\n$k={}$; $\\epsilon={}$'.format(resampler, k, epsilon)

    @staticmethod
    def _get_k(df: pd.DataFrame):
        assert df['top_k'].nunique() == 1, 'cannot merge top-k accuracies with different k'
        return df['top_k'][0]

    def _get_normalized_df(self, metric: str, normalization: str, num_digits: int):
        assert metric in ['top_k_accuracy', 'inverse_top_k_accuracy', 'cross_entropy']
        assert normalization in ['none', 'difference', 'ratio']

        df = self.df

        if metric == 'inverse_top_k_accuracy':
            metric_in_title = '1 - Top-{} Accuracy'.format(self._get_k(df))
            df['metric'] = 1 - df['top_k_accuracy']
            df['dummy_metric'] = 1 - df['dummy_top_k_accuracy']
            sign = -1
        else:
            df['metric'] = df[metric]
            df['dummy_metric'] = df['dummy_{}'.format(metric)]

            if metric == 'top_k_accuracy':
                metric_in_title = 'Top-{} Accuracy'.format(self._get_k(df))
                sign = 1
            elif metric == 'cross_entropy':
                metric_in_title = 'Cross Entropy'
                sign = -1

        if normalization == 'difference':
            df['metric'] = (df['metric'] - df['dummy_metric']) * sign
        elif normalization == 'ratio':
            df['metric'] = (df['metric'] / df['dummy_metric']) ** sign

        # compute mean performance of each hyperparameter set
        df['mean'] = df.groupby(['influence_estimator', 'perturber', 'detector'])['metric'].transform(lambda x: x.mean())
        df['num_decimals'] = df.apply(lambda x: -np.log10(np.abs(x['mean'])).clip(0, None).astype(int) + num_digits - 1
                                      if x['mean'] != 0 else 0, axis=1)
        df['label'] = df.apply(lambda x: '{{:.{}f}}'.format(x.num_decimals).format(x['mean']), axis=1)

        df['hyperparameters'] = df.apply(self._formalize_hyperparameters, axis=1)
        df = pd.concat([df, self._extract_ie_names_and_params(df)], axis=1)

        return df, metric_in_title

    def plot_best_accuracy_per_influence_estimator(self, metric: str = 'cross_entropy', normalization: str = 'none',
                                                   num_digits: int = 3, show_best_params: bool = False,
                                                   expand_limits_kwargs: Dict[str, Any] = {}):
        """
        Plots a bar chart with one bar per influence estimator.
        Each bar shows the best accuracy reached by the respective estimator,
        across all tested hyperparameter combinations.

        :param metric: which accuracy metric to use, one of 'top_k_accuracy', 'inverse_top_k_accuracy'
            and 'cross_entropy'
        :param normalization: how to normalize the metric with respect to the dummy baseline.
            Choose 'none' for no normalization, 'difference' for computing the difference,
            and 'ratio' for computing the ratio.
        :param num_digits: how many digits of the metric to display
        :param show_best_params: whether to use fill colors for showing which hyperparameters
            lead to the respective best accuracy of each influence estimator
        :param expand_limits_kwargs: how to expand the limits of the plot, e.g., to include y=0
        :return: the generated ggplot
        """
        df, metric_in_title = self._get_normalized_df(metric, normalization, num_digits)
        higher_is_better = normalization != 'none' or metric == 'top_k_accuracy'

        # find hyperparameters with best mean performance for each influence estimator
        if higher_is_better:
            df['winner'] = df['mean'] == df['mean'].max()  # global max
            best_indices = df.groupby(by='influence_estimator')['mean'].transform(max) == df['mean']
        else:
            df['winner'] = df['mean'] == df['mean'].min()
            best_indices = df.groupby(by='influence_estimator')['mean'].transform(min) == df['mean']

        df = df.loc[best_indices]

        if normalization == 'none':
            sample_size = np.count_nonzero(df['influence_estimator'] == 'None')
            sample = df['dummy_metric'].sample(sample_size)
            dummy_df = pd.DataFrame({'metric': sample,
                                     'influence_estimator_name': self.DUMMY_NO_IE_NAME,
                                     'hyperparameters': self.PARAM_NO_RESAMPLING,
                                     'mean': sample.mean()})
            df = pd.concat([df, dummy_df])

        # mark best influence estimator with an asterisk and make the corresponding label bold
        # df['influence_estimator_name'] = df.apply(lambda x: '* {}'.format(x.influence_estimator_name)
        #                                           if x.winner else x.influence_estimator_name, axis=1)
        df['label'] = df.apply(lambda x: r'$\mathbf{{{}}}$'.format(x.label) if x.winner else x.label, axis=1)
        # sort influence estimators from worst to best
        ie_names_sorted = pd.unique(df.sort_values(by='mean', ascending=higher_is_better)['influence_estimator_name'])
        df['influence_estimator_name'] = pd.Categorical(df['influence_estimator_name'], ie_names_sorted)

        y_label = 'Advantage in {}'.format(metric_in_title) if normalization != 'none' else metric_in_title

        plot = ggplot(df, aes(x='influence_estimator_name', y='metric')) + clear_theme

        if normalization == 'difference':
            plot += geom_hline(yintercept=0, linetype='dashed')
        elif normalization == 'ratio':
            plot += geom_hline(yintercept=1, linetype='dashed')
        elif normalization == 'none':
            no_attribution_mean = df[df['influence_estimator'] == 'None']['metric'].mean()
            dummy_mean = df[df['influence_estimator_name'] == self.DUMMY_NO_IE_NAME]['metric'].mean()
            baseline_mean = max(no_attribution_mean, dummy_mean) if higher_is_better else min(no_attribution_mean, dummy_mean)
            plot += geom_hline(yintercept=baseline_mean, linetype='dashed')

        if show_best_params:
            plot += stat_summary(aes(fill='hyperparameters'), fun_data='mean_cl_boot', geom='crossbar')
            plot += scale_fill_brewer(type='qual', palette='Paired')
        else:
            plot += stat_summary(fun_data='mean_cl_boot', geom='crossbar')

        return (plot
                + geom_jitter(height=0, width=0.1, shape='o', fill='white', stroke=.3)
                + expand_limits(**expand_limits_kwargs)
                + labs(x='Attribution Method', y=y_label, fill='Best resampling parameters')
                + theme(axis_text_x=element_text(angle=-30, hjust=0, vjust=1),
                        axis_title_x=element_blank(),
                        legend_title=element_text(margin={'b': 10}),
                        legend_entry_spacing=5))

    def plot_accuracy_per_perturb_fraction(self, metric: str = 'cross_entropy', normalization: str = 'none',
                                           num_digits: int = 3, breaks: Optional[List[float]] = None):
        """
        Plots a line chart with one line per training sample.
        The chart shows the accuracy of the influence estimator (y-axis)
        depending on the fraction of perturbed images of the training sample (x-axis).

        :param metric: which accuracy metric to use, one of 'top_k_accuracy' and 'cross_entropy'
        :param normalization: how to normalize the metric with respect to the dummy baseline.
            Choose 'none' for no normalization, 'difference' for computing the difference,
            and 'ratio' for computing the ratio.
        :param num_digits: how many digits of the metric to display
        :param breaks: optional list that specifies the x-axis breaks
        :return: the generated ggplot
        """
        df, metric_in_title = self._get_normalized_df(metric, normalization, num_digits)
        y_label = 'Advantage in {}'.format(metric_in_title) if normalization != 'none' else metric_in_title

        x_args = {} if breaks is None else {'breaks': breaks}

        if self.results.train_obs_balanced:
            s = df.apply(lambda x: '{} ($\leq {}$ per class)'.format(x.train_obs_count,
                                                                     x.train_obs_per_class_threshold),
                         axis=1)
            df.train_obs_count = pd.Categorical(s, ordered=True, categories=s.unique())  # ensure correct ordering

        return (ggplot(df, aes(x='perturb_fraction', y='metric')) +
                clear_theme +
                geom_path(aes(color='train_obs_count')) +
                geom_point(aes(color='train_obs_count')) +
                scale_x_continuous(**x_args) +
                # expand_limits(y=0) +
                labs(x='Fraction $f$ of Resampled Observations', y=y_label, color='Number of training observations') +
                scale_fill_brewer(type='qual', palette='Paired', direction=-1))
