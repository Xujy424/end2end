"""Ensemble entry modules.

Import concrete runners from their modules, e.g. ``v3.ensemble.plain``.
The package init intentionally avoids eager imports so a plain run does not
load bagging or gridsearch.
"""
