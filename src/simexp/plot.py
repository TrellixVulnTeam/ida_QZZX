import io
from contextlib import redirect_stdout
from typing import Optional, List, Literal

import numpy as np
from sklearn.tree._tree import Tree


def prepare_print_node(indent, tikz_prefix):
    pretty_indent = indent + ' ' * len(tikz_prefix) + ' ' * len('r\makecell[cc]{')

    print((indent + r'{tikz_prefix}\makecell[cc]{{')
          .format(tikz_prefix=tikz_prefix), end='')

    return pretty_indent


def print_node(indent, tikz_prefix, feature, threshold, counts, conf,
               pred_class, is_leaf, show_inner_conf,
               all_features_are_counts):
    pretty_indent = prepare_print_node(indent, tikz_prefix)

    feature = feature.replace('_', ' ')

    if not is_leaf:
        if all_features_are_counts:
            threshold = int(threshold)
            comp = '=' if threshold == 0 else r'\leq'
        else:
            threshold = '{:.3f}'.format(threshold)
            comp = r'\leq'

        print(r'Is $\underline{{\text{{{feature}}} {comp} {threshold}}}$?'
              .format(feature=feature, comp=comp, threshold=threshold), end='')
    else:
        print(r'\underline{{\text{{{pred_class}}}}}'
              .format(counts=counts.tolist(), pred_class=pred_class),
              end='')

    if len(counts) <= 2:
        print((r'\\[1mm]' + '\n' + pretty_indent +
               r'${counts[0]} : {counts[1]} \sim {conf:.2f}$')
              .format(counts=counts.tolist(), conf=conf),
              end='')
    elif show_inner_conf or is_leaf:
        print((r'\\[1mm]' + '\n' + pretty_indent +
               '${conf:.2f}$').format(conf=conf), end='')

    print('}}', end='')


def print_pruning_leaf(indent, tikz_prefix, label):
    prepare_print_node(indent, tikz_prefix)
    print(r'\underline{{{label}}}'.format(label=label), end='')
    print('}}', end='')


def print_overflow_leaf(indent, tikz_prefix):
    prepare_print_node(indent, tikz_prefix)
    print(r'\ldots}}', end='')


def tree_to_tikz(tree_: Tree,
                 seen_classes: np.ndarray,
                 counts: np.ndarray,
                 feature_names: [str],
                 class_names: Optional[List[str]] = None,
                 min_conf: float = 0.,
                 max_depth: int = 20,
                 normalize_shade_by: Literal['kappa', 'max_conf'] = 'kappa',
                 show_inner_conf: bool = True,
                 node_style: str = 'block',
                 all_features_are_counts: bool = True) -> str:
    """
    Generates tikz code that plots `tree_`.

    :param tree_: the sklearn tree to plot
    :param seen_classes: which class ids the tree has seen during training
    :param counts: counts of test samples falling into each node,
        as recorded by the method `TreeSurrogate._get_class_counts_in_nodes`.
    :param feature_names: names of the features that the tree uses
        (names of concepts on images, in our case)
    :param class_names: names of all classes in the classification task that the tree was trained on,
        indexed by their class id
    :param min_conf: nodes where the max. confidence for a class is below this threshold
        will be replaced with the label 'uncertain'
    :param max_depth: nodes deeper than this level will be truncated with a '...' node
    :param normalize_shade_by: how to compute the shade of a node from its test accuracy *a*.
        if 'kappa', the shade is the relative position of a in the interval [1 / num. classes, 1 - 1 / num. classes].
        The rationale is that 1 / num. classes is the accuracy that one gets by choosing a random class,
        if classes are balanced.
        If 'max_conf', the shade is the relative position of a in the interval [0, max conf.], where max conf.
        is the maximum confidence of a node in the tree.
    :param show_inner_conf: whether to display the confidence of each node with a string in the node
    :param node_style: tikz style attribute for each node
    :param all_features_are_counts: enables prettier display of concept counts features,
        set to `False` for other features
    :return: tikz code that plots `tree_`
    """
    assert normalize_shade_by in ('max_conf', 'kappa')

    colors = ['LightMaroon', 'LightRoyalBlue']  # binary case
    if len(seen_classes) > len(colors):
        colors = ['LightMaroon'] * len(seen_classes)  # use the same color for all nodes

    children_left = tree_.children_left
    children_right = tree_.children_right
    features = tree_.feature
    thresholds = tree_.threshold

    if class_names is None:
        class_names = seen_classes

    max_conf = np.max(counts / np.sum(counts, axis=-1)[:, None])
    if node_style:
        node_style = node_style + ','

    def recurse(node_id, parent_depth, min_conf):
        depth = parent_depth + 1
        indent = ' ' * 2 * depth

        is_leaf = children_left[node_id] == children_right[node_id]

        feature = feature_names[features[node_id]]
        threshold = thresholds[node_id]
        class_counts = counts[node_id]

        if tree_.n_outputs == 1:
            value = tree_.value[node_id][0, :]
        else:
            value = tree_.value[node_id]

        pred_class_id = int(np.argmax(value))
        pred_class = class_names[pred_class_id]

        conf = class_counts[pred_class_id] / np.sum(class_counts)

        if normalize_shade_by == 'kappa':
            norm = 1. / len(class_counts)
            shade = int(max(0., conf - norm) / (1 - norm) * 100.)
        else:
            shade = int(conf / max_conf * 100.)

        with io.StringIO() as buf:
            with redirect_stdout(buf):
                if shade > 0:
                    color = '{}!{}'.format(colors[pred_class_id], shade)
                else:
                    color = 'none'

                if is_leaf:
                    prune_left = prune_right = True
                else:
                    prune_left = recurse(children_left[node_id], parent_depth + 1, min_conf)
                    prune_right = recurse(children_right[node_id], parent_depth + 1, min_conf)

            if node_id == 0:  # root node
                prefix = r'\node[{}fill={}] (root) {{'.format(node_style, color)
            else:
                print()
                prefix = 'child {{node[{}fill={}] {{'.format(node_style, color)

            prune = (not conf > min_conf) and prune_left and prune_right

            if prune:
                print_pruning_leaf(indent, prefix, 'uncertain')
            elif depth == max_depth and not is_leaf:
                print_overflow_leaf(indent, prefix)
            else:
                print_node(indent, prefix, feature, threshold,
                           class_counts, conf, pred_class, is_leaf,
                           show_inner_conf, all_features_are_counts)
                print(buf.getvalue(), end='')

            if is_leaf or node_id != 0:  # root node does not encapsulate children in group
                print('}', end='')

        return prune

    with io.StringIO() as buf:
        with redirect_stdout(buf):
            recurse(0, -1, min_conf)  # start with the root node id and its parent depth
            print(';')

        return buf.getvalue()
