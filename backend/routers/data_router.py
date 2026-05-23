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


@router.get("/export/stocks")
async def export_stocks():
    """导出全量股票数据（stocks + stock_profile + stock_tags + stock_board_tags）"""
    import json as _json
    async with get_db() as db:
        async with db.execute("SELECT * FROM stocks ORDER BY ts_code") as cur:
            stocks = [dict(r) for r in await cur.fetchall()]
        async with db.execute("SELECT * FROM stock_profile ORDER BY ts_code") as cur:
            profiles = [dict(r) for r in await cur.fetchall()]
        async with db.execute("SELECT * FROM stock_tags ORDER BY ts_code") as cur:
            tags = [dict(r) for r in await cur.fetchall()]
        async with db.execute("SELECT * FROM stock_board_tags ORDER BY ts_code") as cur:
            board_tags = [dict(r) for r in await cur.fetchall()]

    payload = {
        "version": "1.0",
        "exported_at": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "stocks": len(stocks),
            "profiles": len(profiles),
            "tags": len(tags),
            "board_tags": len(board_tags),
        },
        "stocks": stocks,
        "stock_profile": profiles,
        "stock_tags": tags,
        "stock_board_tags": board_tags,
    }
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": "attachment; filename=astock_stocks_export.json"},
    )


@router.post("/import/stocks")
async def import_stocks(body: dict):
    """导入股票数据（增量合并，不删除现有数据）"""
    version = body.get("version", "1.0")
    stocks      = body.get("stocks", [])
    profiles    = body.get("stock_profile", [])
    tags        = body.get("stock_tags", [])
    board_tags  = body.get("stock_board_tags", [])

    counts = {"stocks": 0, "profiles": 0, "tags": 0, "board_tags": 0}

    async with get_db() as db:
        # stocks
        for s in stocks:
            await db.execute("""
                INSERT INTO stocks(ts_code,name,market,industry,list_date,updated_at)
                VALUES(:ts_code,:name,:market,:industry,:list_date,:updated_at)
                ON CONFLICT(ts_code) DO UPDATE SET
                  name=excluded.name,
                  market=excluded.market,
                  industry=CASE WHEN excluded.industry!='' THEN excluded.industry ELSE industry END,
                  list_date=CASE WHEN excluded.list_date!='' THEN excluded.list_date ELSE list_date END,
                  updated_at=excluded.updated_at
            """, s)
            counts["stocks"] += 1

        # stock_profile
        for p in profiles:
            await db.execute("""
                INSERT INTO stock_profile(ts_code,business_desc,domains,keywords,llm_filled,updated_at)
                VALUES(:ts_code,:business_desc,:domains,:keywords,:llm_filled,:updated_at)
                ON CONFLICT(ts_code) DO UPDATE SET
                  business_desc=excluded.business_desc,
                  domains=excluded.domains,
                  keywords=excluded.keywords,
                  llm_filled=excluded.llm_filled,
                  updated_at=excluded.updated_at
            """, p)
            counts["profiles"] += 1

        # stock_tags
        for t in tags:
            await db.execute("""
                INSERT INTO stock_tags(ts_code,products,techs,sectors,chain_pos,themes,all_tags,updated_at)
                VALUES(:ts_code,:products,:techs,:sectors,:chain_pos,:themes,:all_tags,:updated_at)
                ON CONFLICT(ts_code) DO UPDATE SET
                  products=excluded.products, techs=excluded.techs,
                  sectors=excluded.sectors, chain_pos=excluded.chain_pos,
                  themes=excluded.themes, all_tags=excluded.all_tags,
                  updated_at=excluded.updated_at
            """, t)
            counts["tags"] += 1

        # stock_board_tags
        for b in board_tags:
            await db.execute("""
                INSERT INTO stock_board_tags(ts_code,concepts,industries,all_board_tags,updated_at)
                VALUES(:ts_code,:concepts,:industries,:all_board_tags,:updated_at)
                ON CONFLICT(ts_code) DO UPDATE SET
                  concepts=excluded.concepts,
                  industries=excluded.industries,
                  all_board_tags=excluded.all_board_tags,
                  updated_at=excluded.updated_at
            """, b)
            counts["board_tags"] += 1

        await db.commit()

    return APIResponse(message="导入完成", data=counts)
