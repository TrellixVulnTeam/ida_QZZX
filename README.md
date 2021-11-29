**Interpret-Decorrelate-Approximate**, in short: IDA, is a framework for approximating complex image classifiers with simple, interpretable surrogate models that capture some of the causal dependencies within the classifiers.

Note: The `ida` package supersedes the `simexp` package.
`simexp` is outdated and will be removed in the near future.
The reason for the introduction of `ida` is the need to move from a *batch style* experimental design to an *iterator style* design.
In more detail, `simexp` used to process many images at a time for each experimental step,
whereas `ida` streams each image through all steps before processing the next one.
The iterator style of `ida` makes it easier to change experimental parameters and enables more informative runtime comparisons.
