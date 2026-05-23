"""股票数据服务 — 新浪行情接口（akshare 在沙盒/海外网络下不可用时的替代方案）"""
import json
import asyncio
import time
from typing import Optional
import httpx
from loguru import logger
from backend.database import get_db
from backend.services.config_service import get_config
from backend.services.llm_client import get_llm_client

# 全局进度状态
_progress = {
    "stage": "idle", "current": 0, "total": 0,
    "percent": 0.0, "message": "空闲", "done": True, "error": None
}


def get_progress() -> dict:
    return dict(_progress)


def _set_progress(stage, current, total, message, done=False, error=None):
    _progress.update({
        "stage": stage, "current": current, "total": total,
        "percent": round(current / total * 100, 1) if total > 0 else 0,
        "message": message, "done": done, "error": error,
    })


# ─── 新浪接口拉取 ──────────────────────────────────────────────────────────────

SINA_NODES = [
    ("hs_a", "沪主板"),
    ("sz_a", "深主板"),
    ("kcb",  "科创板"),
    ("cyb",  "创业板"),
]

SINA_URL = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"


async def _fetch_node(client: httpx.AsyncClient, node: str, market: str) -> list[dict]:
    """分页拉取单个板块，带重试"""
    stocks = []
    page = 1
    consecutive_empty = 0
    while True:
        for attempt in range(3):
            try:
                r = await client.get(SINA_URL, params={
                    "page": page, "num": 100, "sort": "symbol",
                    "asc": 1, "node": node, "_s_r_a": "page"
                }, timeout=12)
                if not r.text.strip():
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        return stocks
                    await asyncio.sleep(1)
                    break
                data = json.loads(r.text)
                if not data:
                    return stocks
                consecutive_empty = 0
                for item in data:
                    symbol = item.get("symbol", "")
                    code   = item.get("code", "")
                    # 转换为标准 ts_code
                    if symbol.startswith("sh") or symbol.startswith("bj"):
                        ts_code = f"{code}.SH"
                    else:
                        ts_code = f"{code}.SZ"
                    stocks.append({
                        "ts_code": ts_code,
                        "name":    item.get("name", ""),
                        "market":  market,
                        "industry": "",
                        "list_date": "",
                        "mktcap":  item.get("mktcap", 0),
                    })
                if len(data) < 100:
                    return stocks
                page += 1
                await asyncio.sleep(0.2)
                break
            except (json.JSONDecodeError, httpx.ReadTimeout):
                if attempt < 2:
                    await asyncio.sleep(1 + attempt)
                else:
                    logger.warning(f"{node} page={page} 解析失败，跳过")
                    return stocks
            except Exception as e:
                logger.warning(f"{node} page={page} 异常: {e}")
                return stocks
    return stocks


async def update_stock_list() -> dict:
    """更新 A 股全量股票列表
    优先级：Tushare（有 Token，含行业分类）→ 新浪行情（免费兜底）
    """
    _set_progress("stocks", 0, 1, "正在获取股票列表...")
    all_stocks = []

    # ── 优先：Tushare（需开关开启且有 Token）──────────────────────────────────
    tushare_token   = await get_config("tushare_token")
    tushare_enabled = await get_config("tushare_enabled", False)
    if tushare_enabled and tushare_token:
        try:
            import tushare as ts
            ts.set_token(tushare_token)
            pro = ts.pro_api()
            _set_progress("stocks", 0, 1, "Tushare 获取中（含行业分类）...")
            df = await asyncio.to_thread(
                pro.stock_basic,
                exchange="", list_status="L",
                fields="ts_code,name,market,industry,list_date"
            )
            for _, row in df.iterrows():
                mkt = str(row.get("market", "") or "")
                code = str(row["ts_code"])
                if code.endswith(".SH"):
                    mkt = "科创板" if code.startswith("688") else ("沪主板" if not mkt else mkt)
                elif code.endswith(".SZ"):
                    if code.startswith("3"):
                        mkt = "创业板"
                    elif not mkt:
                        mkt = "深主板"
                all_stocks.append({
                    "ts_code":   code,
                    "name":      str(row["name"]),
                    "market":    mkt,
                    "industry":  str(row.get("industry", "") or ""),
                    "list_date": str(row.get("list_date", "") or ""),
                })
            logger.info(f"Tushare 获取 {len(all_stocks)} 只股票（含行业分类）")
        except Exception as e:
            logger.warning(f"Tushare 失败，切换新浪接口: {e}")
            all_stocks = []

    # ── 兜底：新浪行情 ─────────────────────────────────────────────────────────
    if not all_stocks:
        _set_progress("stocks", 0, 4, "正在连接新浪行情接口...")
        try:
            async with httpx.AsyncClient() as client:
                for i, (node, market) in enumerate(SINA_NODES):
                    _set_progress("stocks", i, 4, f"正在获取{market}列表...")
                    stocks = await _fetch_node(client, node, market)
                    all_stocks.extend(stocks)
                    logger.info(f"{market}: {len(stocks)} 只（新浪）")
        except Exception as e:
            logger.error(f"新浪接口也失败: {e}")

    if not all_stocks:
        _set_progress("stocks", 0, 1, "获取失败，无数据", done=True, error="无数据")
        return {"success": False, "error": "无数据"}

    # ── 去重 + 写库 ────────────────────────────────────────────────────────────
    try:
        seen, unique = set(), []
        for s in all_stocks:
            if s["ts_code"] not in seen:
                seen.add(s["ts_code"])
                unique.append(s)

        _set_progress("stocks", 0, len(unique), f"正在写入 {len(unique)} 只股票...")
        async with get_db() as db:
            for i, s in enumerate(unique):
                await db.execute(
                    """INSERT INTO stocks(ts_code,name,market,industry,list_date,updated_at)
                       VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)
                       ON CONFLICT(ts_code) DO UPDATE SET
                       name=excluded.name, market=excluded.market,
                       industry=CASE WHEN excluded.industry!='' THEN excluded.industry ELSE industry END,
                       updated_at=CURRENT_TIMESTAMP""",
                    (s["ts_code"], s["name"], s["market"], s["industry"], s["list_date"]),
                )
                if i % 500 == 0:
                    _set_progress("stocks", i, len(unique), f"写入中 {i}/{len(unique)}...")
            await db.commit()

        _set_progress("stocks", len(unique), len(unique),
                      f"✅ 股票列表更新完成，共 {len(unique)} 只，正在补全主营业务...", done=False)
        logger.info(f"股票列表更新完成: {len(unique)} 只")
        return {"success": True, "count": len(unique)}
    except Exception as e:
        _set_progress("stocks", 0, 1, f"写入失败: {e}", done=True, error=str(e))
        logger.error(f"股票列表写入失败: {e}")
        return {"success": False, "error": str(e)}


async def update_stock_profiles(limit: int = 9999) -> dict:
    """为股票批量补全主营业务（10并发，后台持续运行）"""
    async with get_db() as db:
        async with db.execute(
            """SELECT s.ts_code, s.name, s.industry FROM stocks s
               LEFT JOIN stock_profile sp ON s.ts_code = sp.ts_code
               WHERE sp.ts_code IS NULL
                  OR sp.business_desc IS NULL
                  OR sp.business_desc = ''
                  OR (sp.llm_filled = 0 AND length(sp.business_desc) < 30)
               LIMIT ?""", (limit,)
        ) as cur:
            rows = await cur.fetchall()

    if not rows:
        _set_progress("profiles", 0, 0, "✅ 主营业务已全部补全", done=True)
        return {"success": True, "count": 0}

    total = len(rows)
    _set_progress("profiles", 0, total, f"准备补全 {total} 只股票主营业务...")
    client = await get_llm_client()

    filled = 0
    failed = 0
    done_count = 0
    lock = asyncio.Lock()

    CONCURRENCY = 10
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _process_one(row):
        nonlocal filled, failed, done_count
        async with sem:
            try:
                desc, domains, keywords, llm_filled, ind = await _get_stock_profile(
                    row["ts_code"], row["name"], row["industry"], client
                )
                if desc is not None:
                    async with get_db() as db:
                        await db.execute(
                            """INSERT OR REPLACE INTO stock_profile
                               (ts_code,business_desc,domains,keywords,llm_filled,updated_at)
                               VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)""",
                            (row["ts_code"], desc,
                             json.dumps(domains, ensure_ascii=False),
                             json.dumps(keywords, ensure_ascii=False),
                             int(llm_filled)),
                        )
                        if ind and not row["industry"]:
                            await db.execute(
                                "UPDATE stocks SET industry=? WHERE ts_code=?",
                                (ind, row["ts_code"])
                            )
                        await db.commit()
                    async with lock:
                        filled += 1
                else:
                    async with lock:
                        failed += 1
            except Exception as e:
                async with lock:
                    failed += 1
                logger.warning(f"股票 {row['ts_code']} 补全失败: {e}")
            finally:
                async with lock:
                    done_count += 1
                    if done_count % 50 == 0 or done_count == total:
                        _set_progress("profiles", done_count, total,
                                      f"补全中 {done_count}/{total}（✅{filled} ❌{failed}）")
            # 每个任务间隔 0.1s，避免 cninfo 瞬间并发过高
            await asyncio.sleep(0.1)

    await asyncio.gather(*[_process_one(row) for row in rows])

    _set_progress("profiles", total, total,
                  f"✅ 主营业务补全完成 {filled}/{total}（失败 {failed} 只）", done=True)
    logger.info(f"主营业务补全: {filled}/{total}，失败: {failed}")
    return {"success": True, "count": filled}


async def _get_stock_profile(ts_code, name, industry, client) -> tuple:
    """优先 stock_profile_cninfo，失败则大模型补全"""
    try:
        import akshare as ak
        import asyncio as _asyncio
        code = ts_code.split(".")[0]
        df = await _asyncio.wait_for(_asyncio.to_thread(ak.stock_profile_cninfo, symbol=code), timeout=10)
        if df is not None and not df.empty:
            row = df.iloc[0]
            desc = str(row.get("主营业务", "") or "").strip()
            ind  = str(row.get("所属行业", "") or industry or "").strip()
            if desc and len(desc) > 10:
                # 关键词：从主营业务文本提取
                import re as _re
                kws = list(dict.fromkeys(
                    [w for w in _re.split(r'[、，；。、 ]+', desc) if 2 <= len(w) <= 8]
                ))[:10]
                return desc, [ind] if ind else [], kws, False, ind
    except Exception as e:
        logger.debug(f"cninfo {ts_code} 失败: {e}")

    # 大模型补全
    if client:
        prompt = (f'请为A股上市公司"{name}"（行业：{industry or "未知"}）提供简短主营业务描述。'
                  f'返回JSON（不加markdown）：'
                  f'{{"business_desc":"主营业务（80字内）","domains":["领域1","领域2"],"keywords":["关键词1","关键词2","关键词3"]}}')
        try:
            resp = await client.chat([{"role": "user", "content": prompt}])
            text = resp.strip().lstrip("```json").lstrip("```").rstrip("```")
            data = json.loads(text)
            return data.get("business_desc",""), data.get("domains",[]), data.get("keywords",[]), True, industry or ""
        except Exception as e:
            logger.debug(f"大模型补全 {name} 失败: {e}")

    # 不写入兜底值，返回 None 让调用方跳过，下次更新时可以重试
    return None, None, None, None, None


async def get_stock_stats() -> dict:
    async with get_db() as db:
        async with db.execute("SELECT COUNT(*) as cnt FROM stocks") as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT COUNT(*) as cnt FROM stock_profile") as cur:
            with_profile = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT MAX(updated_at) as last FROM stocks") as cur:
            row = await cur.fetchone()
            last_updated = row["last"] if row else None
    return {
        "total": total,
        "with_profile": with_profile,
        "profile_coverage": round(with_profile / total * 100, 1) if total > 0 else 0,
        "last_updated": last_updated,
    }
