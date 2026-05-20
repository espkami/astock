"""新闻相关路由"""
import json
from fastapi import APIRouter, Query
from backend.database import get_db
from backend.models import NewsListResponse, NewsItem, APIResponse, MatchResult, MatchedStock
from backend.services.news_collector import run_collection

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("", response_model=NewsListResponse)
async def list_news(
    limit: int = Query(50, le=2000),
    sentiment: str = Query("all"),
    offset: int = Query(0),
):
    async with get_db() as db:
        where = "" if sentiment == "all" else f"WHERE sentiment='{sentiment}'"
        async with db.execute(f"SELECT COUNT(*) as cnt FROM news {where}") as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute(
            f"SELECT * FROM news {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()

    items = []
    for row in rows:
        items.append(NewsItem(
            id=row["id"],
            url=row["url"],
            title=row["title"],
            source=row["source"],
            published_at=row["published_at"],
            summary=row["summary"],
            sentiment=row["sentiment"] or "neutral",
            industries=json.loads(row["industries"] or "[]"),
            event_type=row["event_type"],
            keywords=json.loads(row["keywords"] or "[]"),
            confidence=row["confidence"] or 0.0,
            raw_source=row["raw_source"],
            created_at=row["created_at"],
        ))
    return NewsListResponse(total=total, items=items)


@router.get("/{news_id}/matches")
async def get_news_matches(news_id: int):
    async with get_db() as db:
        async with db.execute(
            "SELECT mr.*, n.title as news_title FROM match_results mr JOIN news n ON mr.news_id=n.id WHERE mr.news_id=?",
            (news_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return APIResponse(success=False, message="未找到匹配结果")

    stocks = json.loads(row["matched_stocks"] or "[]")
    return MatchResult(
        id=row["id"],
        news_id=news_id,
        news_title=row["news_title"],
        matched_stocks=[MatchedStock(**s) for s in stocks],
        created_at=row["created_at"],
    )


@router.post("/collect", response_model=APIResponse)
async def trigger_collect():
    import asyncio
    asyncio.create_task(run_collection())
    return APIResponse(message="采集任务已触发")
