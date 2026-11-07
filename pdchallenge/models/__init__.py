"""Models and the registry that makes them addressable by name.

Importing this package registers every built-in model. Adding a model means writing the
class and adding it to the import list below — nothing else needs to change, and configs
can then refer to it by ``name``.
"""

from .base import BaseModel
from .chain_view import ChainViewModel
from .ensemble import EnsembleModel
from .gbm import GBMModel
from .hierarchical import HierarchicalCurveModel
from .linear import RidgeModel
from .registry import MODELS, build, register, spec_name
from .segmented import SegmentedModel
from .set_transformer import SetTransformerModel, VisitGRUModel
from .smoothed import SmoothedModel

__all__ = [
    "BaseModel",
    "ChainViewModel",
    "EnsembleModel",
    "GBMModel",
    "HierarchicalCurveModel",
    "RidgeModel",
    "SegmentedModel",
    "SetTransformerModel",
    "VisitGRUModel",
    "SmoothedModel",
    "MODELS",
    "build",
    "register",
    "spec_name",
]
