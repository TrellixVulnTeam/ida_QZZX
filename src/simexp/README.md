Attention: The `simexp` package is outdated.
============================================

Most functionality of this package has been reimplemented in the `ida` package.
The reason was a move from a *batch-style* experimental design to an *iterator-style* design.
In a nutshell, `simexp` processes many images at a time for each experimental step,
whereas `ida` streams each image through all steps before processing the next one.
`ida` makes it easier to change experimental parameters and enables more informative runtime comparisons.
The `simexp` package will be removed in the near future.
