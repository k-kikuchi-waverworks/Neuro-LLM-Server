import main


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["name"] == "Neuro-LLM-Server"


def test_health_endpoint_without_monitoring(client):
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_metrics_endpoint_without_monitoring(client):
    response = client.get("/metrics")
    assert response.status_code == 503


def test_chat_completions_not_ready(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "stream": True,
        },
    )

    assert response.status_code == 503


def test_chat_completions_non_streaming_success(client, monkeypatch):
    class DummyInference:
        def generate(self, **_kwargs):
            yield "hello"
            yield "world"

    class DummyQueue:
        async def execute(self, coro):
            return await coro

    monkeypatch.setattr(main, "inference_engine", DummyInference())
    monkeypatch.setattr(main, "request_queue", DummyQueue())

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "helloworld"


def test_health_endpoint_with_monitoring(client, monkeypatch):
    class DummyMonitoring:
        def get_health_status(self):
            return {
                "status": "healthy",
                "error_rate": 0.0,
                "gpu_memory_percent": 0.0,
                "gpu_memory_ok": True,
            }

    monkeypatch.setattr(main, "monitoring", DummyMonitoring())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_endpoint_with_monitoring(client, monkeypatch):
    class DummyMonitoring:
        def get_prometheus_metrics(self):
            return "neuro_llm_requests_total 1\n"

    monkeypatch.setattr(main, "monitoring", DummyMonitoring())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "neuro_llm_requests_total" in response.text
