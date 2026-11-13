"""Kiểm tra khung dự án: các module trong src/ và adapters/ import được."""
import importlib

import pytest

MODULES = [
    "src",
    "src.preprocessing",
    "src.tokenizer",
    "src.cluster_metrics",
    "src.clustering",
    "src.patterns",
    "src.ar_rec",
    "src.graph",
    "src.evaluate",
    "src.eda",
    "adapters",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
