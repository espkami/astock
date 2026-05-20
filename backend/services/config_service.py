"""配置读写服务 — SQLite config 表 <-> 内存缓存"""
import json
from typing import Any, Optional
from loguru import logger
from backend.database import get_db

_cache: dict[str, Any] = {}


async def get_config(key: str, default: Any = None) -> Any:
    if key in _cache:
        return _cache[key]
    async with get_db() as db:
        async with db.execute("SELECT value FROM config WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return default
    val = row["value"]
    try:
        parsed = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        parsed = val
    _cache[key] = parsed
    return parsed


async def set_config(key: str, value: Any) -> None:
    val = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    async with get_db() as db:
        await db.execute(
            "INSERT INTO config(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            (key, val),
        )
        await db.commit()
    _cache[key] = value
    logger.debug(f"config set: {key}")


async def get_all_config() -> dict[str, Any]:
    async with get_db() as db:
        async with db.execute("SELECT key, value FROM config") as cur:
            rows = await cur.fetchall()
    result = {}
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            result[row["key"]] = row["value"]
    _cache.update(result)
    return result


async def set_config_batch(items: list[dict]) -> None:
    async with get_db() as db:
        for item in items:
            val = json.dumps(item["value"], ensure_ascii=False) \
                if not isinstance(item["value"], str) else item["value"]
            await db.execute(
                "INSERT INTO config(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
                (item["key"], val),
            )
            _cache[item["key"]] = item["value"]
        await db.commit()
    logger.info(f"批量保存配置 {len(items)} 条")
