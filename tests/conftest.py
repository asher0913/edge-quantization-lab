import pytest

from edge_quant.model import load_digits_splits, train_mlp


@pytest.fixture(scope="session")
def splits():
    return load_digits_splits(0)


@pytest.fixture(scope="session")
def model(splits):
    return train_mlp(splits.x_train, splits.y_train, seed=0)
