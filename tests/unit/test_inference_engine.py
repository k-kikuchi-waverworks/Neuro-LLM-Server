import pytest

from inference_engine import InferenceEngine
from utils.errors import ValidationError


def test_prepare_messages_without_image(dummy_model_manager, config_no_gpu):
    engine = InferenceEngine(dummy_model_manager, config_no_gpu)
    image, text_messages = engine._prepare_messages([
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ])

    assert image is not None
    assert text_messages == [{"role": "user", "content": "hello"}]


def test_generate_respects_max_tokens(dummy_model_manager, config_no_gpu):
    engine = InferenceEngine(dummy_model_manager, config_no_gpu)
    chunks = list(engine.generate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        max_tokens=1,
        stream=True,
    ))

    assert chunks == ["hello"]


def test_generate_invalid_temperature(dummy_model_manager, config_no_gpu):
    engine = InferenceEngine(dummy_model_manager, config_no_gpu)

    with pytest.raises(ValidationError):
        list(engine.generate(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            temperature=3.0,
            stream=True,
        ))
