import time
from uuid import uuid4
import redis.asyncio as redis
from rag.config import settings
from rag.observability import log

_redis = redis.from_url(settings.redis_url, decode_responses=True)

# KEYS[1]=bucket key  ARGV: now_ms, window_ms, limit, member
_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)   -- evict timestamps older than the window
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, ARGV[4])              -- record this request
    redis.call('PEXPIRE', key, window)                 -- auto-clean idle clients
    return limit - count - 1                            -- remaining
end
return -1                                               -- over the limit
"""
_script = _redis.register_script(_SLIDING_WINDOW)

async def check_rate_limit(client_id: str, limit: int, window_seconds: int) -> int:
    """Return remaining allowance, or -1 if the client is over the limit."""
    try:
        now_ms = int(time.time() * 1000)
        remaining = await _script(
            keys=[f"ratelimit:{client_id}"],
            args=[now_ms, window_seconds * 1000, limit, f"{now_ms}:{uuid4()}"],
        )
        return int(remaining)
    except Exception as e:                               # fail-OPEN: don't let a Redis blip 500 the API
        log.warning("ratelimit.error", error=type(e).__name__)
        return limit
