import pytest

from config import Config
from utils.errors import ValidationError


def test_config_defaults():
    config = Config()
    assert config.model.name
    assert config.model.quantization in ["int4", "int8", "fp16", "fp32"]
    assert config.server.port > 0


def test_config_from_file_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join([
            "model:",
            "  name: openbmb/test-int8",
            "  quantization: int8",
            "server:",
            "  port: 9000",
            "monitoring:",
            "  enable_gpu_monitoring: false",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    config = Config.from_file(str(config_path))

    assert config.model.name == "openbmb/test-int8"
    assert config.model.quantization == "int8"
    assert config.server.port == 9000
    assert config.gpu.cuda_visible_devices == "1"
    assert config.monitoring.enable_gpu_monitoring is False


def test_config_validate_errors():
    config = Config()
    config.model.quantization = "invalid"
    config.server.port = 99999
    config.server.timeout = 0
    config.inference.temperature = 3.0
    config.inference.max_tokens = 0

    with pytest.raises(ValidationError):
        config.validate()
