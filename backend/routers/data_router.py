"""数据管理路由"""
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.database import get_db
from backend.models import APIResponse, CleanRequest

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/clean")
async def clean_data(req: CleanRequest):
    from fastapi import HTTPException
    # 防止全量清除新闻（news 必须指定 days > 0）
    if req.days <= 0 and "news" in req.targets:
        raise HTTPException(status_code=400, detail="清除新闻数据必须指定保留天数（days > 0）")
    deleted = {}
    async with get_db() as db:
        if "news" in req.targets:
            # 按天数清除历史新闻（days > 0）
            cur = await db.execute(
                "DELETE FROM news WHERE created_at < datetime('now', ? || ' days')",
                (f"-{req.days}",),
            )
            deleted["news"] = cur.rowcount
        if "matches" in req.targets:
            if req.days <= 0:
                # days=0：全量清除匹配结果
                cur = await db.execute("DELETE FROM match_results")
            else:
                # days>0：只清除N天前的匹配结果
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


@router.get("/export/news")
async def export_news(days: int = 0):
    """导出新闻及匹配结果
    days=0 导出全量，days>0 导出最近N天
    """
    import datetime as _dt
    async with get_db() as db:
        if days > 0:
            since = (
                _dt.datetime.utcnow() - _dt.timedelta(days=days)
            ).strftime("%Y-%m-%d %H:%M:%S")
            async with db.execute(
                "SELECT * FROM news WHERE created_at >= ? ORDER BY created_at DESC", (since,)
            ) as cur:
                news = [dict(r) for r in await cur.fetchall()]
            news_ids = [n["id"] for n in news]
        else:
            async with db.execute("SELECT * FROM news ORDER BY created_at DESC") as cur:
                news = [dict(r) for r in await cur.fetchall()]
            news_ids = [n["id"] for n in news]

        # 只导出对应新闻的匹配结果
        matches = []
        if news_ids:
            placeholders = ",".join("?" * len(news_ids))
            async with db.execute(
                f"SELECT * FROM match_results WHERE news_id IN ({placeholders}) ORDER BY created_at DESC",
                news_ids,
            ) as cur:
                matches = [dict(r) for r in await cur.fetchall()]

    import datetime as _dt2
    filename = f"astock_news_export_{_dt2.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return JSONResponse(
        content={
            "version": "1.0",
            "exported_at": _dt2.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": {"news": len(news), "matches": len(matches)},
            "news": news,
            "matches": matches,
        },
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/import/news")
async def import_news(body: dict):
    """导入新闻及匹配结果（增量合并，URL去重，已存在的跳过）"""
    news_list   = body.get("news", [])
    match_list  = body.get("matches", [])

    counts = {"news": 0, "news_skipped": 0, "matches": 0, "matches_skipped": 0}
    # 记录旧id→新id的映射（news.id可能冲突）
    id_map: dict = {}

    async with get_db() as db:
        for n in news_list:
            old_id = n.get("id")
            url    = n.get("url") or ""
            title  = (n.get("title") or "").strip()
            # URL去重
            if url:
                async with db.execute("SELECT id FROM news WHERE url=? LIMIT 1", (url,)) as cur:
                    existing = await cur.fetchone()
                if existing:
                    id_map[old_id] = existing["id"]
                    counts["news_skipped"] += 1
                    continue
            # 标题去重
            if title:
                async with db.execute("SELECT id FROM news WHERE title=? LIMIT 1", (title,)) as cur:
                    existing = await cur.fetchone()
                if existing:
                    id_map[old_id] = existing["id"]
                    counts["news_skipped"] += 1
                    continue
            # 插入
            cur2 = await db.execute(
                """INSERT INTO news
                   (url,title,source,published_at,content,summary,sentiment,
                    industries,event_type,keywords,confidence,raw_source,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    url, title, n.get("source",""),
                    n.get("published_at",""), n.get("content",""),
                    n.get("summary"), n.get("sentiment","neutral"),
                    n.get("industries","[]"), n.get("event_type",""),
                    n.get("keywords","[]"), n.get("confidence",0.0),
                    n.get("raw_source","import"),
                    n.get("created_at") or __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            new_id = cur2.lastrowid
            id_map[old_id] = new_id
            counts["news"] += 1

        # 导入匹配结果（用新的 news_id）
        for m in match_list:
            old_news_id = m.get("news_id")
            new_news_id = id_map.get(old_news_id, old_news_id)
            if not new_news_id:
                counts["matches_skipped"] += 1
                continue
            # 检查是否已存在
            async with db.execute(
                "SELECT id FROM match_results WHERE news_id=? LIMIT 1", (new_news_id,)
            ) as cur:
                if await cur.fetchone():
                    counts["matches_skipped"] += 1
                    continue
            await db.execute(
                "INSERT INTO match_results(news_id, matched_stocks, created_at) VALUES(?,?,?)",
                (
                    new_news_id,
                    m.get("matched_stocks","[]"),
                    m.get("created_at") or __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            counts["matches"] += 1

        await db.commit()

    return APIResponse(
        message=f"导入完成：新闻新增{counts['news']}条(跳过{counts['news_skipped']}条) 匹配结果新增{counts['matches']}条(跳过{counts['matches_skipped']}条)",
        data=counts,
    )
