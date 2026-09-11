import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def tiny_cfg():
    from configs import get
    return get("tiny")


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real HF weights or a GPU")
    config.addinivalue_line("markers", "needs_data: reads data/final/*.json")
    config.addinivalue_line("markers", "needs_spacy: requires the spaCy model")
