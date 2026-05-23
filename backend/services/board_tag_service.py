"""板块标签服务 — 从概念/行业板块建立股票标签映射

数据来源（优先级）：
1. 东方财富 stock_board_concept_cons_em / stock_board_industry_cons_em（用户服务器可用）
2. 同花顺 stock_board_concept_name_ths（概念列表，沙盒稳定）

标签命名规则：标签名 = 板块名（如"粮食概念"、"机器人概念"、"粮食加工"）
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
    """建 stock_board_tags 表：存储股票→板块标签的映射"""
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stock_board_tags (
                ts_code      TEXT PRIMARY KEY,
                concepts     TEXT DEFAULT '[]',   -- 概念板块标签
                industries   TEXT DEFAULT '[]',   -- 行业板块标签
                all_board_tags TEXT DEFAULT '[]', -- 合并（用于快速搜索）
                updated_at   TEXT
            )
        """)
        await db.commit()
    logger.info("stock_board_tags 表初始化完成")


async def fetch_concept_members_em(concept_name: str) -> list[str]:
    """东方财富：获取概念板块成分股代码列表"""
    import asyncio as _asyncio
    try:
        import akshare as ak
        df = await _asyncio.wait_for(
            _asyncio.to_thread(ak.stock_board_concept_cons_em, symbol=concept_name),
            timeout=15
        )
        if df is None or df.empty:
            return []
        # 返回 ts_code 格式（需要判断交易所）
        codes = []
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).zfill(6)
            if code.startswith("6"):
                codes.append(f"{code}.SH")
            elif code.startswith("9"):
                codes.append(f"{code}.BJ")
            else:
                codes.append(f"{code}.SZ")
        return codes
    except Exception as e:
        logger.debug(f"东方财富概念成分股失败 {concept_name}: {e}")
        return []


async def fetch_industry_members_em(industry_name: str) -> list[str]:
    """东方财富：获取行业板块成分股代码列表"""
    import asyncio as _asyncio
    try:
        import akshare as ak
        df = await _asyncio.wait_for(
            _asyncio.to_thread(ak.stock_board_industry_cons_em, symbol=industry_name),
            timeout=15
        )
        if df is None or df.empty:
            return []
        codes = []
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).zfill(6)
            if code.startswith("6"):
                codes.append(f"{code}.SH")
            elif code.startswith("9"):
                codes.append(f"{code}.BJ")
            else:
                codes.append(f"{code}.SZ")
        return codes
    except Exception as e:
        logger.debug(f"东方财富行业成分股失败 {industry_name}: {e}")
        return []


async def generate_board_tags() -> dict:
    """
    从已有的 stock_tags 数据反向构建板块→股票映射。
    不依赖外部接口，直接用 sectors/products/themes 字段聚合。
    同时尝试从东方财富/同花顺拉取概念板块（若可用则追加）。
    """
    await init_board_tags_table()
    import asyncio as _asyncio
    import json as _json

    # ── 阶段1：从 stock_tags 反向构建板块映射（不依赖外部接口）────────────────
    _set_board_progress(0, 1, "从现有标签数据构建板块映射...")

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

    total_from_tags = len(tag_rows)
    _set_board_progress(0, total_from_tags, f"处理 {total_from_tags} 只股票的标签数据...")

    for i, row in enumerate(tag_rows):
        ts_code = row["ts_code"]
        try:
            sectors = _json.loads(row["sectors"] or "[]")
            products = _json.loads(row["products"] or "[]")
            themes   = _json.loads(row["themes"]   or "[]")
            # sectors → industries（行业板块）
            for s in sectors:
                if s and len(s) >= 2:
                    add_tag(ts_code, s, "industries")
            # products → concepts（概念板块，取前3个）
            for p in products[:3]:
                if p and len(p) >= 2:
                    add_tag(ts_code, p, "concepts")
            # themes → concepts
            for t in themes:
                if t and len(t) >= 2:
                    add_tag(ts_code, t, "concepts")
        except Exception:
            pass
        if i % 500 == 0:
            _set_board_progress(i, total_from_tags, f"处理标签数据 {i}/{total_from_tags}...")

    # ── 阶段2：尝试从东方财富/同花顺追加概念板块（可选，失败不影响结果）────────
    _set_board_progress(total_from_tags, total_from_tags + 1,
                        "尝试从外部接口追加概念板块（可选）...")
    try:
        import akshare as ak
        for fn_name, fn, col in [
            ("东方财富概念", lambda: ak.stock_board_concept_name_em(), "板块名称"),
            ("同花顺概念",   lambda: ak.stock_board_concept_name_ths(), "name"),
        ]:
            try:
                df = await _asyncio.wait_for(_asyncio.to_thread(fn), timeout=15)
                if df is not None and not df.empty and col in df.columns:
                    concept_names = df[col].tolist()
                    logger.info(f"{fn_name}: {len(concept_names)} 个概念板块，逐个拉成分股")
                    for i, concept in enumerate(concept_names[:100]):  # 最多100个避免超时
                        try:
                            members = await fetch_concept_members_em(concept)
                            for ts_code in members:
                                add_tag(ts_code, concept, "concepts")
                        except Exception:
                            pass
                        await _asyncio.sleep(0.2)
                    break  # 成功一个就不再尝试
            except Exception as e:
                logger.debug(f"{fn_name}失败（可选步骤，忽略）: {e}")
    except Exception:
        pass

    # ── 写入数据库 ───────────────────────────────────────────────────────────
    total = len(stock_tags_map)
    _set_board_progress(0, total, f"写入 {total} 只股票的板块标签...")
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
            """, (
                ts_code,
                _json.dumps(concepts,   ensure_ascii=False),
                _json.dumps(industries, ensure_ascii=False),
                _json.dumps(all_tags,   ensure_ascii=False),
            ))
            written += 1
        await db.commit()

    _set_board_progress(total, total,
        f"✅ 板块标签生成完成，{written} 只股票，来源：标签数据反向构建", done=True)
    logger.info(f"板块标签完成: {written} 只股票")
    return {"success": True, "stocks": written, "boards": total}


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
        "concepts":      json.loads(row["concepts"]       or "[]"),
        "industries":    json.loads(row["industries"]     or "[]"),
        "all_board_tags": json.loads(row["all_board_tags"] or "[]"),
    }
