import pytest

torch = pytest.importorskip("torch")

from config import Config
from model_manager import ModelManager


def test_get_torch_dtype():
    manager = ModelManager(Config())

    assert manager._get_torch_dtype("fp32") == torch.float32
    assert manager._get_torch_dtype("fp16") == torch.float16
    assert manager._get_torch_dtype("int4") is None
    assert manager._get_torch_dtype("int8") is None
    assert manager._get_torch_dtype("unknown") is None
