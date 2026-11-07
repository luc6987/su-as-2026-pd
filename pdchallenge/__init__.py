"""Parkinson "true OFF" challenge: baseline toolkit.

Deliberately import-light — this module must not import subpackages, because
``pdchallenge.paths`` and ``pdchallenge.data.*`` import it on the way up. Import what you
need directly, e.g. ``from pdchallenge.data.loading import load_data``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
