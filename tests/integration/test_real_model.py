import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml


if os.getenv("RUN_REAL_MODEL_TESTS") != "true":
    pytest.skip("RUN_REAL_MODEL_TESTS=true is required", allow_module_level=True)


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_temp_config(tmp_path: Path, base_config: Path):
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    config.setdefault("server", {})
    config["server"]["host"] = "127.0.0.1"
    config["server"]["port"] = _get_free_port()
    config.setdefault("monitoring", {})
    config["monitoring"]["enable_gpu_monitoring"] = False

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, config["server"]["host"], config["server"]["port"]


def _wait_for_ready(base_url: str, timeout: int = 300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.status_code in (200, 503):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def test_real_model_chat_completion(tmp_path):
    base_config = Path(__file__).resolve().parents[2] / "config.yaml"
    config_path, host, port = _write_temp_config(tmp_path, base_config)
    base_url = f"http://{host}:{port}"

    env = os.environ.copy()
    env["NEURO_LLM_CONFIG"] = str(config_path)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(base_config.parent),
        env=env,
    )

    try:
        assert _wait_for_ready(base_url, timeout=300)

        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
                "stream": False,
                "max_tokens": 10,
            },
            timeout=60.0,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"]
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
