import os
Asterix_root = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
model_dir = os.path.join(Asterix_root, "model") + os.path.sep

from .utils.hci_metrics import *

__version__ = "2.3"
