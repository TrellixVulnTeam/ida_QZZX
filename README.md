**Interpret-Decorrelate-Approximate**, in short: IDA, is a framework for approximating complex image classifiers with simple, interpretable surrogate models that capture some of the causal dependencies within the classifiers.

Note: The `ida` package in *src/ida* supersedes the `simexp` package in the same directory.
`simexp` is outdated and will be removed in the near future.
The motivation for `ida` is to move from a *batch style* experimental design to an *iterator style* design.
In more detail, `ida` streams each image through all steps before processing the next one, whereas `simexp` processes many images at a time for each experimental step.
The iterator style of `ida` makes it possible to interrupt long running experiments and enables more informative runtime comparisons.
