"""Data preprocessing: scaling, encoding, and train/test splitting."""

from mlcore.preprocessing.encoding import label_decode, label_encode, one_hot_encode
from mlcore.preprocessing.scaling import min_max_scale, normalize, standardize
from mlcore.preprocessing.splitting import (
    group_shuffle_split,
    stratified_group_shuffle_split,
    train_test_split,
)

__all__ = [
    "group_shuffle_split",
    "label_decode",
    "label_encode",
    "min_max_scale",
    "normalize",
    "one_hot_encode",
    "standardize",
    "stratified_group_shuffle_split",
    "train_test_split",
]
