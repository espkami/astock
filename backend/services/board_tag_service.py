"""板块标签服务 — 从同花顺概念/行业指数建立股票→板块标签映射

数据来源（Tushare）：
- ths_index(type='N') → 概念指数列表（约400个）
- ths_index(type='I') → 行业指数列表（约90个）
- ths_member(ts_code) → 每个指数的成分股列表

限速：ths_index 1次/小时，ths_member 待测（实测无严格限速）
策略：先拉指数列表，再逐个拉成分股，建立 股票→[所属板块] 反向映射
"""
import json
import asyncio
from loguru import logger
from backend.database import get_db

# 进度状态
_board_progress = {
    "stage": "idle", "current": 0, "total": 0,
    "percent": 0.0, "message": "空闲", "done": True, "error": None
}

def get_board_progress() -> dict:
    return dict(_board_progress)

def _set_board_progress(current, total, message, done=False, error=None):
    _board_progress.update({
        "stage": "board_tagging",
        "current": current, "total": total,
        "percent": round(current / total * 100, 1) if total > 0 else 0,
        "message": message, "done": done, "error": error,
    })


async def init_board_tags_table():
    """建 stock_board_tags 表"""
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stock_board_tags (
                ts_code        TEXT PRIMARY KEY,
                concepts       TEXT DEFAULT '[]',   -- 概念板块标签（如「黄金概念」「新能源汽车」）
                industries     TEXT DEFAULT '[]',   -- 行业板块标签（如「半导体」「白酒」）
                all_board_tags TEXT DEFAULT '[]',   -- 合并
                updated_at     TEXT
            )
        """)
        await db.commit()
    logger.info("stock_board_tags 表初始化完成")


async def generate_board_tags() -> dict:
    """
    从雪球 stock_individual_basic_info_xq 获取每只股票的 affiliate_industry（板块分类）。
    这是市场给股票贴的行业板块标签，与主营标签（产品/业务词）完全不同。

    覆盖率 100%，无限速，约 1s/只，5500只约 90 分钟。
    """
    await init_board_tags_table()
    import asyncio as _asyncio
    import akshare as ak

    async with get_db() as db:
        async with db.execute("SELECT ts_code, name FROM stocks ORDER BY ts_code") as cur:
            all_stocks = await cur.fetchall()

    total = len(all_stocks)
    if total == 0:
        _set_board_progress(0, 0, "无股票数据", done=True, error="无数据")
        return {"success": False, "error": "无股票数据"}

    _set_board_progress(0, total, f"准备从雪球获取 {total} 只股票的板块分类...")

    success_count = 0
    fail_count    = 0
    stock_board_map = {}  # {ts_code: ind_name}

    CONCURRENCY = 5
    sem = _asyncio.Semaphore(CONCURRENCY)
    lock = _asyncio.Lock()

    async def fetch_one(ts_code, name):
        nonlocal success_count, fail_count
        parts = ts_code.split(".")
        xq_symbol = (parts[1] + parts[0]) if len(parts) == 2 else ts_code
        async with sem:
            try:
                df = await _asyncio.wait_for(
                    _asyncio.to_thread(ak.stock_individual_basic_info_xq, symbol=xq_symbol),
                    timeout=12
                )
                if df is not None and not df.empty:
                    xq = dict(zip(df['item'], df['value']))
                    aff = xq.get('affiliate_industry', {})
                    ind_name = aff.get('ind_name', '') if isinstance(aff, dict) else ''
                    if ind_name:
                        async with lock:
                            stock_board_map[ts_code] = ind_name
                            success_count += 1
                    else:
                        async with lock:
                            fail_count += 1
                else:
                    async with lock:
                        fail_count += 1
            except Exception as e:
                async with lock:
                    fail_count += 1
                logger.debug(f"雪球板块 {ts_code} 失败: {e}")
            finally:
                async with lock:
                    done = success_count + fail_count
                    if done % 100 == 0 or done == total:
                        _set_board_progress(done, total,
                            f"获取中 {done}/{total}（✅{success_count} ❌{fail_count}）")
                await _asyncio.sleep(0.2)

    await _asyncio.gather(*[fetch_one(r["ts_code"], r["name"]) for r in all_stocks])

    # 写入数据库
    _set_board_progress(total, total, "写入数据库...")
    written = 0
    async with get_db() as db:
        for ts_code, ind_name in stock_board_map.items():
            # ind_name 作为行业板块标签
            industries = [ind_name]
            all_tags   = [ind_name]
            await db.execute("""
                INSERT OR REPLACE INTO stock_board_tags
                (ts_code, concepts, industries, all_board_tags, updated_at)
                VALUES (?,?,?,?,CURRENT_TIMESTAMP)
            """, (ts_code, '[]',
                  json.dumps(industries, ensure_ascii=False),
                  json.dumps(all_tags,   ensure_ascii=False)))
            written += 1
        await db.commit()

    _set_board_progress(total, total,
        f"✅ 板块标签生成完成，{written} 只股票获得板块分类", done=True)
    logger.info(f"板块标签完成（雪球数据）: {written}/{total}")
    return {"success": True, "stocks": written, "boards": written}



async def _generate_board_tags_from_stock_tags() -> dict:
    """
    降级方案：从 stock_tags 反向构建（当 Tushare 不可用时）
    注意：此方案生成的板块标签与主营标签内容相同，仅作临时替代
    """
    logger.warning("使用降级方案：从主营标签反向构建板块标签（内容与主营标签相同）")
    _set_board_progress(0, 1, "从现有主营标签数据构建板块映射（降级方案）...")

    stock_tags_map: dict[str, dict] = {}

    def add_tag(ts_code: str, tag: str, tag_type: str):
        if ts_code not in stock_tags_map:
            stock_tags_map[ts_code] = {"concepts": [], "industries": []}
        lst = stock_tags_map[ts_code][tag_type]
        if tag not in lst:
            lst.append(tag)

    async with get_db() as db:
        async with db.execute(
            "SELECT ts_code, sectors, products, themes FROM stock_tags WHERE all_tags != '[]'"
        ) as cur:
            tag_rows = await cur.fetchall()

    total = len(tag_rows)
    for i, row in enumerate(tag_rows):
        ts_code = row["ts_code"]
        try:
            for s in json.loads(row["sectors"] or "[]"):
                if s and len(s) >= 2: add_tag(ts_code, s, "industries")
            for p in json.loads(row["products"] or "[]")[:3]:
                if p and len(p) >= 2: add_tag(ts_code, p, "concepts")
            for t in json.loads(row["themes"] or "[]"):
                if t and len(t) >= 2: add_tag(ts_code, t, "concepts")
        except Exception:
            pass
        if i % 500 == 0:
            _set_board_progress(i, total, f"处理标签数据 {i}/{total}...")

    written = 0
    async with get_db() as db:
        for ts_code, tags in stock_tags_map.items():
            concepts   = tags["concepts"]
            industries = tags["industries"]
            all_tags   = list(dict.fromkeys(concepts + industries))
            await db.execute("""
                INSERT OR REPLACE INTO stock_board_tags
                (ts_code, concepts, industries, all_board_tags, updated_at)
                VALUES (?,?,?,?,CURRENT_TIMESTAMP)
            """, (ts_code,
                  json.dumps(concepts,   ensure_ascii=False),
                  json.dumps(industries, ensure_ascii=False),
                  json.dumps(all_tags,   ensure_ascii=False)))
            written += 1
        await db.commit()

    _set_board_progress(total, total,
        f"✅ 板块标签生成完成（降级方案，{written} 只）", done=True)
    return {"success": True, "stocks": written, "boards": 0, "fallback": True}


async def get_stock_board_tags(ts_code: str) -> dict:
    """获取单只股票的板块标签"""
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM stock_board_tags WHERE ts_code=?", (ts_code,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {}
    return {
        "concepts":       json.loads(row["concepts"]       or "[]"),
        "industries":     json.loads(row["industries"]     or "[]"),
        "all_board_tags": json.loads(row["all_board_tags"] or "[]"),
    }


async def fetch_concept_members_em(concept_name: str) -> list[str]:
    """保留旧接口签名兼容性（已废弃，东方财富接口被封）"""
    return []


async def get_board_progress() -> dict:
    return get_board_progress()
