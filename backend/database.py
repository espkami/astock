"""数据库初始化与连接管理"""
import os
import aiosqlite
from contextlib import asynccontextmanager
from loguru import logger


def _db_path() -> str:
    """运行时动态读取，避免模块加载时路径固化"""
    return os.environ.get("DB_PATH", "/app/data/astock.db")


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS news (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT UNIQUE,
    title       TEXT NOT NULL,
    source      TEXT,
    published_at DATETIME,
    content     TEXT,
    summary     TEXT,
    sentiment   TEXT DEFAULT 'neutral',
    industries  TEXT,
    event_type  TEXT,
    keywords    TEXT,
    confidence  REAL DEFAULT 0.0,
    raw_source  TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS match_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id         INTEGER NOT NULL,
    matched_stocks  TEXT NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS stocks (
    ts_code     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    market      TEXT,
    industry    TEXT,
    list_date   TEXT,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS stock_profile (
    ts_code       TEXT PRIMARY KEY,
    business_desc TEXT,
    domains       TEXT,
    keywords      TEXT,
    llm_filled    INTEGER DEFAULT 0,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS stock_financials (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code      TEXT NOT NULL,
    period       TEXT NOT NULL,
    revenue      REAL,
    net_profit   REAL,
    gross_margin REAL,
    debt_ratio   REAL,
    cashflow     REAL,
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, period)
);
CREATE INDEX IF NOT EXISTS idx_news_sentiment   ON news(sentiment);
CREATE INDEX IF NOT EXISTS idx_news_created_at  ON news(created_at);
CREATE INDEX IF NOT EXISTS idx_news_source      ON news(raw_source);
CREATE INDEX IF NOT EXISTS idx_match_news_id    ON match_results(news_id);
CREATE INDEX IF NOT EXISTS idx_stock_industry   ON stocks(industry);
"""


@asynccontextmanager
async def get_db():
    """异步上下文管理器，自动设置 row_factory"""
    path = _db_path()
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        yield conn


async def init_db():
    """初始化数据库，建表"""
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(CREATE_TABLES_SQL)
        await db.commit()
    logger.info(f"数据库初始化完成: {path}")
