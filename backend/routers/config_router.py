"""配置路由"""
from fastapi import APIRouter
from backend.models import APIResponse, ConfigBatch, TestKeyRequest
from backend.services.config_service import get_all_config, set_config_batch
from backend.services.llm_client import test_llm_connection, PROVIDERS
from backend.services.scheduler import update_collect_interval

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
async def get_config():
    data = await get_all_config()
    return APIResponse(data=data)


@router.post("")
async def save_config(batch: ConfigBatch):
    items = [{"key": i.key, "value": i.value} for i in batch.items]
    await set_config_batch(items)

    # 动态应用各路源间隔
    interval_map = {
        "newsapi_interval": "newsapi",
        "rss_interval":     "rss",
        "llm_search_interval": "llm_search",
    }
    for item in batch.items:
        if item.key in interval_map:
            try:
                from backend.services.scheduler import update_source_intervals
                kwargs = {interval_map[item.key]: int(item.value)}
                await update_source_intervals(**kwargs)
            except Exception:
                pass
        elif item.key == "collect_interval":
            try:
                await update_collect_interval(int(item.value))
            except Exception:
                pass

    return APIResponse(message="配置已保存")


@router.post("/test-key")
async def test_key(req: TestKeyRequest):
    result = await test_llm_connection(
        provider=req.provider,
        api_key=req.api_key,
        model=req.model or "gpt-4o-mini",
        base_url=req.base_url,
    )
    return APIResponse(success=result["success"], data=result)


@router.get("/providers")
async def get_providers():
    return APIResponse(data=PROVIDERS)
