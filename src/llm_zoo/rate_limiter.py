import asyncio
import functools
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class RateLimitConfig:
    max_calls: int = 50
    calls_per_minute: Optional[int] = None
    calls_per_second: Optional[int] = None
    delay_between_calls: float = 0.0


class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_calls)
        self.call_times: deque = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        await self.semaphore.acquire()
        if self.config.calls_per_minute or self.config.calls_per_second:
            async with self.lock:
                now = time.time()
                if self.config.calls_per_minute:
                    cutoff = now - 60
                    while self.call_times and self.call_times[0] < cutoff:
                        self.call_times.popleft()
                if self.config.calls_per_second:
                    cutoff = now - 1
                    while self.call_times and self.call_times[0] < cutoff:
                        self.call_times.popleft()
                at_rpm = self.config.calls_per_minute and len(self.call_times) >= self.config.calls_per_minute
                at_rps = self.config.calls_per_second and len(self.call_times) >= self.config.calls_per_second
                if at_rpm or at_rps:
                    wait = (1 if at_rps else 60) - (now - self.call_times[0])
                    if wait > 0:
                        await asyncio.sleep(wait)
                self.call_times.append(time.time())

    def release(self):
        self.semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *_):
        self.release()


def rate_limited_async_call(config: RateLimitConfig):
    def decorator(func: Callable) -> Callable:
        rate_limiter = RateLimiter(config)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with rate_limiter:
                if config.delay_between_calls > 0:
                    await asyncio.sleep(config.delay_between_calls)
                return await func(*args, **kwargs)

        return wrapper
    return decorator


OPENROUTER_RATE_LIMIT = RateLimitConfig(
    max_calls=7000,
    calls_per_minute=1000,
    calls_per_second=10,
    delay_between_calls=0.1,
)
