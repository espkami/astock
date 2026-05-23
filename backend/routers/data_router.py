"""数据管理路由"""
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.database import get_db
from backend.models import APIResponse, CleanRequest

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/clean")
async def clean_data(req: CleanRequest):
    deleted = {}
    async with get_db() as db:
        if "news" in req.targets:
            cur = await db.execute(
                "DELETE FROM news WHERE created_at < datetime('now', ? || ' days')",
                (f"-{req.days}",),
            )
            deleted["news"] = cur.rowcount
        if "matches" in req.targets:
            cur = await db.execute(
                "DELETE FROM match_results WHERE created_at < datetime('now', ? || ' days')",
                (f"-{req.days}",),
            )
            deleted["matches"] = cur.rowcount
        if "stocks" in req.targets:
            await db.execute("DELETE FROM stock_tags")        # 主营标签
            await db.execute("DELETE FROM stock_board_tags")  # 板块标签
            await db.execute("DELETE FROM stock_profile")
            await db.execute("DELETE FROM stocks")
            deleted["stocks"] = True
        await db.commit()
    return APIResponse(message="清理完成", data=deleted)


@router.get("/preview-clean")
async def preview_clean(days: int = 30):
    async with get_db() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM news WHERE created_at < datetime('now', ? || ' days')",
            (f"-{days}",),
        ) as cur:
            news_count = (await cur.fetchone())["cnt"]
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM match_results WHERE created_at < datetime('now', ? || ' days')",
            (f"-{days}",),
        ) as cur:
            match_count = (await cur.fetchone())["cnt"]
    return APIResponse(data={"news": news_count, "matches": match_count, "days": days})


@router.get("/export")
async def export_data():
    async with get_db() as db:
        async with db.execute("SELECT * FROM news ORDER BY created_at DESC") as cur:
            news = [dict(r) for r in await cur.fetchall()]
        async with db.execute("SELECT * FROM match_results ORDER BY created_at DESC") as cur:
            matches = [dict(r) for r in await cur.fetchall()]
        async with db.execute("SELECT * FROM stocks") as cur:
            stocks = [dict(r) for r in await cur.fetchall()]

    return JSONResponse(
        content={"news": news, "matches": matches, "stocks": stocks},
        headers={"Content-Disposition": "attachment; filename=astock_export.json"},
    )
