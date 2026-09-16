"""Training strategy modules.

Import concrete strategies from their modules, for example
``v3.training.strategy.plain`` or ``v3.training.strategy.kfold``.
The package init intentionally avoids eager imports so a plain run only loads
the plain strategy.
"""
