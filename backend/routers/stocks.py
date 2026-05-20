"""股票相关路由"""
import asyncio
import json
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from backend.database import get_db
from backend.models import StockListResponse, StockItem, APIResponse
from backend.services import stock_service

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("")
async def list_stocks(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, le=200),
    keyword: str = Query(""),
):
    offset = (page - 1) * page_size
    where = "WHERE s.name LIKE ? OR s.ts_code LIKE ?" if keyword else ""
    params = [f"%{keyword}%", f"%{keyword}%"] if keyword else []

    async with get_db() as db:
        async with db.execute(f"SELECT COUNT(*) as cnt FROM stocks s {where}", params) as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute(
            f"""SELECT s.ts_code, s.name, s.market, s.industry,
                       COALESCE(sp.business_desc, '') as business_desc
                FROM stocks s LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                {where} ORDER BY s.name LIMIT ? OFFSET ?""",
            params + [page_size, offset],
        ) as cur:
            rows = await cur.fetchall()

    items = [dict(r) for r in rows]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.get("/stats")
async def stock_stats():
    stats = await stock_service.get_stock_stats()
    return APIResponse(data=stats)


@router.post("/update", response_model=APIResponse)
async def trigger_update():
    asyncio.create_task(_run_full_update())
    return APIResponse(message="股票更新任务已触发")


async def _run_full_update():
    await stock_service.update_stock_list()
    await stock_service.update_stock_profiles(limit=9999)


@router.get("/update-progress")
async def update_progress():
    """SSE 进度流"""
    async def event_stream():
        import asyncio
        while True:
            progress = stock_service.get_progress()
            data = json.dumps(progress, ensure_ascii=False)
            yield f"data: {data}\n\n"
            if progress.get("done"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
