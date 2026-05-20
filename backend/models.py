"""Pydantic 数据模型"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── 配置 ────────────────────────────────────────────────────────────────────

class ConfigItem(BaseModel):
    key: str
    value: Any

class ConfigBatch(BaseModel):
    items: list[ConfigItem]


# ─── 新闻 ────────────────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    id: int
    url: Optional[str] = None
    title: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    summary: Optional[str] = None
    sentiment: str = "neutral"
    industries: list[str] = []
    event_type: Optional[str] = None
    keywords: list[str] = []
    confidence: float = 0.0
    raw_source: Optional[str] = None
    created_at: str


class NewsListResponse(BaseModel):
    total: int
    items: list[NewsItem]


# ─── 匹配结果 ─────────────────────────────────────────────────────────────────

class MatchedStock(BaseModel):
    ts_code: str
    name: str
    score: float
    reason: str
    sentiment_impact: str = "neutral"
    semantic_score: float = 0.0
    industry_score: float = 0.0

class MatchResult(BaseModel):
    id: int
    news_id: int
    news_title: Optional[str] = None
    matched_stocks: list[MatchedStock]
    created_at: str


# ─── 股票 ────────────────────────────────────────────────────────────────────

class StockItem(BaseModel):
    ts_code: str
    name: str
    market: Optional[str] = None
    industry: Optional[str] = None
    list_date: Optional[str] = None
    has_profile: bool = False

class StockListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[StockItem]

class StockUpdateProgress(BaseModel):
    stage: str
    current: int
    total: int
    percent: float
    message: str
    done: bool = False
    error: Optional[str] = None


# ─── 统计 ────────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_news: int = 0
    total_matches: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_stocks: int = 0
    last_collect_at: Optional[str] = None
    current_model: str = "未配置"
    scheduler_running: bool = False
    token_usage: dict = {}


# ─── 响应 ────────────────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: Optional[Any] = None

class TestKeyRequest(BaseModel):
    provider: str
    api_key: str
    base_url: Optional[str] = None
    model: Optional[str] = None

class CollectRequest(BaseModel):
    sources: list[str] = ["newsapi", "rss", "llm"]  # 指定触发哪些源

class CleanRequest(BaseModel):
    days: int = 30
    targets: list[str] = ["news", "matches"]  # news / matches / stocks
