import os
import matplotlib

Asterix_root = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(Asterix_root, "model")

__version__ = "2.6"

matplotlib.rc('image', origin='lower')
