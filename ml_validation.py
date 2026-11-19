"""Deterministic group partitions: test never participates in model selection."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_validation_test(groups, seed=42):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 5:
        raise ValueError("At least five distinct case_id groups are required; row-wise splitting is forbidden.")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    development, test = next(outer.split(groups, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_local, validation_local = next(inner.split(development, groups=groups[development]))
    train, validation = development[train_local], development[validation_local]
    partitions = [set(groups[indices]) for indices in (train, validation, test)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise AssertionError("case_id leakage between partitions")
    return train, validation, test
