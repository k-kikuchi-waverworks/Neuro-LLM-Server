"""Request queue for managing concurrent requests"""

import asyncio
import time
from typing import Optional, Callable, Any, Coroutine
from collections import deque
from utils.errors import ValidationError
from utils.logging import get_logger

logger = get_logger(__name__)


class RequestQueue:
    """Manages request queueing and concurrency limits"""
    
    def __init__(self, max_concurrent: int, enable_queue: bool = True):
        self.max_concurrent = max_concurrent
        self.enable_queue = enable_queue
        self.current_requests = 0
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def acquire(self) -> None:
        """Acquire a slot for processing a request"""
        if not self.enable_queue:
            # If queueing is disabled, just use semaphore
            await self.semaphore.acquire()
            return
        
        # Wait for available slot
        await self.semaphore.acquire()
        
        async with self.lock:
            self.current_requests += 1
            logger.debug(f"Request acquired. Current: {self.current_requests}/{self.max_concurrent}")
    
    async def release(self) -> None:
        """Release a slot after request is processed"""
        async with self.lock:
            if self.current_requests > 0:
                self.current_requests -= 1
            self.semaphore.release()
            logger.debug(f"Request released. Current: {self.current_requests}/{self.max_concurrent}")
    
    async def execute(self, coro: Coroutine) -> Any:
        """Execute a coroutine with queue management"""
        await self.acquire()
        try:
            return await coro
        finally:
            await self.release()
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.current_requests
