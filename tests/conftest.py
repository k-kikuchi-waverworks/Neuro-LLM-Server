import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

import main
from config import Config


@pytest.fixture
def config_no_gpu():
    config = Config()
    config.monitoring.enable_gpu_monitoring = False
    return config


@pytest.fixture
def dummy_model_manager():
    class DummyModel:
        def chat(self, **kwargs):
            yield "hello"
            yield "world"

    class DummyModelManager:
        def __init__(self):
            self.model = DummyModel()
            self.tokenizer = object()

        def get_model(self):
            return self.model

        def get_tokenizer(self):
            return self.tokenizer

    return DummyModelManager()


@pytest.fixture
def client(monkeypatch):
    @asynccontextmanager
    async def no_lifespan(app):
        yield

    monkeypatch.setattr(main.app.router, "lifespan_context", no_lifespan)
    main.inference_engine = None
    main.request_queue = None
    main.monitoring = None
    main.config = None

    with TestClient(main.app) as test_client:
        yield test_client
