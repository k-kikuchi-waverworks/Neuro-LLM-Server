from monitoring import Monitoring


def test_monitoring_records_metrics(config_no_gpu):
    monitoring = Monitoring(config_no_gpu)

    monitoring.record_request(0.5)
    monitoring.record_request(1.0, error=True)

    metrics = monitoring.get_metrics()
    assert metrics["request_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["min_latency"] <= metrics["max_latency"]

    health = monitoring.get_health_status()
    assert health["status"] == "unhealthy"


def test_prometheus_output_contains_counters(config_no_gpu):
    monitoring = Monitoring(config_no_gpu)
    monitoring.record_request(0.1)

    metrics_text = monitoring.get_prometheus_metrics()
    assert "neuro_llm_requests_total" in metrics_text
    assert "neuro_llm_errors_total" in metrics_text
