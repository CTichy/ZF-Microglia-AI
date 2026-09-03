import logging as _logging

# Cellpose (`vit.py`/`models.py`) logs a `.warning()`-level notice the
# moment either module is imported if the optional facebookresearch/
# dinov3 CPDINO backbone isn't installed. This plugin only ever uses
# Cellpose-SAM's own CPSAM checkpoints (this project's fine-tuned
# cyto/multi models), never CPDINO -- so it's pure noise on every
# launch, same class of issue as the cupy/cucim stdout warning
# suppressed in _labeling.py's backend probe. Configuring the logger
# by name here, before cellpose is ever actually imported (that
# happens lazily inside _cellpose_seg.py/_epoch_sweep.py), is enough:
# logging.getLogger() is a global singleton registry keyed by name, so
# this takes effect the moment cellpose's own modules later call
# getLogger(__name__) with the same name.
_logging.getLogger("cellpose.vit").setLevel(_logging.ERROR)
_logging.getLogger("cellpose.models").setLevel(_logging.ERROR)

__version__ = "0.2.0"

from ._widget import ZFMicrogliaAIWidget

__all__ = ["ZFMicrogliaAIWidget"]
