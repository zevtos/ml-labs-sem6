"""Data preprocessing: scaling, encoding, and train/test splitting."""

from mlcore.preprocessing.encoding import label_decode, label_encode, one_hot_encode
from mlcore.preprocessing.scaling import min_max_scale, normalize, standardize
from mlcore.preprocessing.splitting import train_test_split

__all__ = [
    "label_decode",
    "label_encode",
    "min_max_scale",
    "normalize",
    "one_hot_encode",
    "standardize",
    "train_test_split",
]
