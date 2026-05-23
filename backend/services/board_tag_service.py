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
    批量生成板块标签：
    1. 拉取所有概念板块列表（同花顺，稳定）
    2. 逐个获取成分股（东方财富，用户服务器可用）
    3. 建立 股票→[概念标签] 反向映射
    4. 写入 stock_board_tags 表
    """
    await init_board_tags_table()

    import asyncio as _asyncio

    # ── 获取概念板块列表 ──────────────────────────────────────────────────────
    _set_board_progress(0, 1, "正在获取概念板块列表...")
    concept_names = []
    industry_names = []

    try:
        import akshare as ak
        # 同花顺概念列表（稳定）
        df_concepts = await _asyncio.wait_for(
            _asyncio.to_thread(ak.stock_board_concept_name_ths),
            timeout=60
        )
        concept_names = df_concepts["name"].tolist() if df_concepts is not None else []
        logger.info(f"获取到 {len(concept_names)} 个概念板块")
    except Exception as e:
        logger.warning(f"同花顺概念列表获取失败: {e}")

    try:
        import akshare as ak
        # 东方财富行业列表
        df_industries = await _asyncio.wait_for(
            _asyncio.to_thread(ak.stock_board_industry_name_em),
            timeout=15
        )
        if df_industries is not None and not df_industries.empty:
            industry_names = df_industries["板块名称"].tolist()
            logger.info(f"获取到 {len(industry_names)} 个行业板块")
    except Exception as e:
        logger.warning(f"东方财富行业列表获取失败: {e}")

    total_boards = len(concept_names) + len(industry_names)
    if total_boards == 0:
        _set_board_progress(0, 0, "❌ 无法获取板块列表", done=True, error="无数据")
        return {"success": False, "error": "无法获取板块列表"}

    # ── 建立 股票→标签 映射 ──────────────────────────────────────────────────
    # stock_tags_map: {ts_code: {"concepts": [...], "industries": [...]}}
    stock_tags_map: dict[str, dict] = {}

    def add_tag(ts_code: str, tag: str, tag_type: str):
        if ts_code not in stock_tags_map:
            stock_tags_map[ts_code] = {"concepts": [], "industries": []}
        lst = stock_tags_map[ts_code][tag_type]
        if tag not in lst:
            lst.append(tag)

    # ── 处理概念板块 ─────────────────────────────────────────────────────────
    _set_board_progress(0, total_boards, f"处理概念板块（共{len(concept_names)}个）...")
    for i, concept in enumerate(concept_names):
        _set_board_progress(i, total_boards, f"概念板块 {i}/{len(concept_names)}: {concept}")
        members = await fetch_concept_members_em(concept)
        for ts_code in members:
            add_tag(ts_code, concept, "concepts")
        await _asyncio.sleep(0.2)  # 限速

    # ── 处理行业板块 ─────────────────────────────────────────────────────────
    for i, industry in enumerate(industry_names):
        idx = len(concept_names) + i
        _set_board_progress(idx, total_boards, f"行业板块 {i}/{len(industry_names)}: {industry}")
        members = await fetch_industry_members_em(industry)
        for ts_code in members:
            add_tag(ts_code, industry, "industries")
        await _asyncio.sleep(0.2)

    # ── 写入数据库 ───────────────────────────────────────────────────────────
    _set_board_progress(total_boards, total_boards, "写入数据库...")
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
                json.dumps(concepts,   ensure_ascii=False),
                json.dumps(industries, ensure_ascii=False),
                json.dumps(all_tags,   ensure_ascii=False),
            ))
            written += 1
        await db.commit()

    _set_board_progress(total_boards, total_boards,
        f"✅ 板块标签生成完成，{written} 只股票获得板块标签", done=True)
    logger.info(f"板块标签完成: {written} 只股票，{total_boards} 个板块")
    return {"success": True, "stocks": written, "boards": total_boards}


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
