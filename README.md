# nixproc: nix concepts for data processing with python

What does this library provide?
- prevent multiple evaluations of the same computation
- lazy evaluation
- semi-automatic dependency resolution (which module needs which data for
  processing, for example when it comes to plots etc.)
- data accesibility across modules
- convenience functions for data processing (working with data with
  uncertainties, displaying a second axis with a different unit, using parameter
  labels in curve fits...)

Roadmap:
- utilities
  - [ ] hyperplot: see
        (https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb),
        but improve it: add linewidth, alpha, thickness change (all per line)
  - [ ] LaTeX export: make it easy to export fit data (and keep it up-to-date in
        the protocol)
