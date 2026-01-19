import pytest

from request_queue import RequestQueue


@pytest.mark.asyncio
async def test_execute_returns_result():
    queue = RequestQueue(max_concurrent=1, enable_queue=True)

    async def sample():
        return "ok"

    result = await queue.execute(sample())
    assert result == "ok"
    assert queue.get_queue_size() == 0


@pytest.mark.asyncio
async def test_execute_without_queueing():
    queue = RequestQueue(max_concurrent=2, enable_queue=False)

    async def sample():
        return 123

    result = await queue.execute(sample())
    assert result == 123
    assert queue.get_queue_size() == 0
