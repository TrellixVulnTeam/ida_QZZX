from dataclasses import dataclass

from plotnine import *

from simexp.fit import SurrogatesFitter

clear_theme = (theme_bw() +
               theme(panel_border=element_blank(),
                     panel_grid=element_blank(),
                     panel_grid_major=element_blank(),
                     panel_grid_minor=element_blank(),
                     text=element_text(wrap=True,
                                       fontfamily='Latin Modern Roman',
                                       fontstretch='normal',
                                       fontweight='light',
                                       size=16,
                                       colour='black'),
                     plot_title=element_text(size=20, fontweight='normal'),
                     axis_text_x=element_text(hjust=.5, vjust=1),
                     axis_title=element_text(size=14, fontweight='normal'),
                     axis_title_x=element_text(margin={'t': 20}, angle=0, vjust=1),
                     axis_title_y=element_text(margin={'b': 20}, angle=90, vjust=1),
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
    def df(self):
        return self.results.to_flat_pandas()

    def plot_best_accuracy_per_influence_estimator(self):
        max_indices = self.df.groupby(by='influence_estimator').idxmax()
        df = self.df.loc[max_indices]
        # baseline_idx = df['influence_estimator', 'perturber', 'detector'].isnull()
        # assert len(baseline_idx) == 1
        df.assign(hyperparameters=lambda x: '{}, {}'.format(x.perturber, x.detector))

        return (ggplot(df, aes('influence_estimator')) +
                clear_theme +
                geom_bar(aes(size='cross_entropy')) +
                ggtitle('Best Accuracy Per Influence Estimator'))

    def plot_accuracy_by_perturb_fraction(self):
        df = self.df
        df.assign(hyperparameters=lambda x: '{}, {}, {}'.format(x.influence_estimator, x.perturber, x.detector))

        return (ggplot(df, aes('perturb_fraction')) +
                clear_theme +
                geom_path(aes(y='cross_entropy', fill='hyperparameters')) +
                facet_wrap(['train_sample_fraction']) +
                ggtitle('Accuracy per Fraction of Perturbed Images'))
