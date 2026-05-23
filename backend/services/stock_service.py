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
_profiles_running = False  # 防止 profiles 任务重复启动


def get_progress() -> dict:
    return dict(_progress)


def _set_progress(stage, current, total, message, done=False, error=None):
    _progress.update({
        "stage": stage, "current": current, "total": total,
        "percent": round(current / total * 100, 1) if total > 0 else 0,
        "message": message, "done": done, "error": error,
    })


# ─── 新浪接口拉取 ──────────────────────────────────────────────────────────────

# 新浪接口说明：
# hs_a 节点现已返回全市场所有A股（沪/深/北），无需再拉其他节点
# sz_a/kcb/cyb 已包含在 hs_a 内，重复拉取会导致股票数量虚高
SINA_NODES = [
    ("hs_a", "全市场"),
]

SINA_URL = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"


async def _fetch_node(client: httpx.AsyncClient, node: str, market: str,
                      progress_cb=None) -> list[dict]:
    """分页拉取单个板块，带重试。progress_cb(page, count) 每页回调更新进度。"""
    stocks = []
    page = 1
    consecutive_empty = 0
    _timeout = httpx.Timeout(connect=8.0, read=12.0, write=5.0, pool=5.0)
    while True:
        for attempt in range(3):
            try:
                r = await client.get(SINA_URL, params={
                    "page": page, "num": 100, "sort": "symbol",
                    "asc": 1, "node": node, "_s_r_a": "page"
                }, timeout=_timeout)
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
                    if symbol.startswith("sh"):
                        ts_code = f"{code}.SH"
                        mkt = "科创板" if code.startswith("688") else "沪主板"
                    elif symbol.startswith("bj"):
                        ts_code = f"{code}.BJ"
                        mkt = "北交所"
                    else:
                        ts_code = f"{code}.SZ"
                        mkt = "创业板" if code.startswith("3") else "深主板"
                    stocks.append({
                        "ts_code": ts_code,
                        "name":    item.get("name", ""),
                        "market":  mkt,
                        "industry": "",
                        "list_date": "",
                        "mktcap":  item.get("mktcap", 0),
                    })
                # 每页完成后回调更新进度
                if progress_cb:
                    progress_cb(page, len(stocks))
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


_stock_list_running = False  # 防止 update_stock_list 重复启动

async def update_stock_list() -> dict:
    """更新 A 股全量股票列表
    优先级：Tushare（有 Token，含行业分类）→ 新浪行情（免费兜底）
    """
    global _stock_list_running
    if _stock_list_running:
        logger.warning("update_stock_list 已在运行，跳过重复启动")
        return {"success": False, "error": "任务已在运行"}
    _stock_list_running = True
    try:
        return await _do_update_stock_list()
    finally:
        _stock_list_running = False


async def _do_update_stock_list() -> dict:
    _set_progress("stocks", 0, 1, "正在获取股票列表...")
    all_stocks = []

    # ── 优先：Tushare（需开关开启且有 Token）──────────────────────────────────
    tushare_token   = await get_config("tushare_token")
    stock_list_source = await get_config("stock_list_source", "")
    # 兼容旧字段：stock_list_source 未设置时，读 tushare_enabled 旧字段
    if not stock_list_source:
        old_enabled = await get_config("tushare_enabled", False)
        stock_list_source = "tushare" if (old_enabled == True or old_enabled == "true") else "sina"
    tushare_enabled = (stock_list_source == "tushare") and bool(tushare_token)
    if tushare_enabled and tushare_token:
        try:
            import tushare as ts
            ts.set_token(tushare_token)
            pro = ts.pro_api()
            _set_progress("stocks", 0, 1, "Tushare 获取中（含行业分类）...")
            df = await asyncio.wait_for(
                asyncio.to_thread(
                    pro.stock_basic,
                    exchange="", list_status="L",
                    fields="ts_code,name,market,industry,list_date"
                ),
                timeout=60
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
            _set_progress("stocks", 0, 100, f"Tushare 不可用，切换新浪接口...")
            all_stocks = []

    # ── 兜底：新浪行情 ─────────────────────────────────────────────────────────
    if not all_stocks:
        _set_progress("stocks", 0, 4, "正在连接新浪行情接口...")
        try:
            async with httpx.AsyncClient() as client:
                for i, (node, market) in enumerate(SINA_NODES):
                    _set_progress("stocks", 0, 100, f"正在拉取{market}股票列表（第1页）...")

                    def make_cb(mkt):
                        def cb(page, count):
                            # 新浪全市场约56页，用页数估算进度
                            est_total = max(page * 100, count + 100)
                            _set_progress("stocks", count, est_total,
                                          f"拉取{mkt}列表 第{page}页，已获取 {count} 只...")
                        return cb

                    try:
                        stocks = await asyncio.wait_for(
                            _fetch_node(client, node, market, progress_cb=make_cb(market)),
                            timeout=120
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"新浪 {market} 节点超时（120s），跳过")
                        stocks = []
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
    global _profiles_running
    if _profiles_running:
        logger.warning("update_stock_profiles 已在运行，跳过重复启动")
        return {"success": False, "error": "任务已在运行"}
    _profiles_running = True
    try:
        return await _do_update_profiles(limit)
    finally:
        _profiles_running = False


async def _do_update_profiles(limit: int = 9999) -> dict:
    """实际执行补全逻辑"""
    async with get_db() as db:
        async with db.execute(
            """SELECT s.ts_code, s.name, s.industry FROM stocks s
               LEFT JOIN stock_profile sp ON s.ts_code = sp.ts_code
               WHERE sp.ts_code IS NULL
                  OR sp.business_desc IS NULL
                  OR sp.business_desc = ''
                  OR (sp.llm_filled = 0 AND length(sp.business_desc) < 5)
                  OR sp.llm_filled = 1
               LIMIT ?""", (limit,)
        ) as cur:
            rows = await cur.fetchall()

    if not rows:
        _set_progress("profiles", 0, 0, "✅ 主营业务已全部补全", done=True)
        return {"success": True, "count": 0}

    total = len(rows)
    _set_progress("profiles", 0, total, f"准备补全 {total} 只股票主营业务...")

    filled = 0
    failed = 0
    done_count = 0
    lock = asyncio.Lock()

    CONCURRENCY = 5  # 纯数据接口，无需限速
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _process_one(row):
        nonlocal filled, failed, done_count
        async with sem:
            try:
                desc, domains, keywords, llm_filled, ind = await _get_stock_profile(
                    row["ts_code"], row["name"], row["industry"]
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
                    if done_count % 10 == 0 or done_count == total:
                        _set_progress("profiles", done_count, total,
                                      f"补全中 {done_count}/{total}（✅{filled} ❌{failed}）")
            # 每个任务间隔 0.1s，避免 cninfo 瞬间并发过高
            await asyncio.sleep(0.1)

    await asyncio.gather(*[_process_one(row) for row in rows])

    _set_progress("profiles", total, total,
                  f"✅ 主营业务补全完成 {filled}/{total}（失败 {failed} 只）", done=True)
    logger.info(f"主营业务补全: {filled}/{total}，失败: {failed}")
    return {"success": True, "count": filled}


async def _get_stock_profile(ts_code, name, industry, client=None) -> tuple:
    """公司简介 + 行业补全。

    问题根源与修复：
    ① AKShare stock_zyjs_ths  → 同花顺主营介绍，文字描述质量好
    ② 巨潮 stock_profile_cninfo → 主营业务(精简) + 机构简介(完整背景) + 所属行业(标准分类)
       - 之前只取"主营业务"字段，7-80字，信息量不足
       - 现在同时取"机构简介"补充完整背景，行业用巨潮标准分类
    ③ LLM 兜底（所有数据接口失败时）

    行业字段优先级：巨潮所属行业 > AKShare产品类型 > 原有行业值
    """
    import akshare as ak
    import asyncio as _asyncio
    import re as _re
    from backend.services.config_service import get_config as _gc

    profile_source = await _gc("profile_source", "both")
    code_only = ts_code.split(".")[0]
    found_desc = None
    found_ind  = industry or ""

    # ── ① AKShare stock_zyjs_ths（同花顺主营介绍）────────────────────────────
    if profile_source in ("eastmoney", "both"):
        try:
            df = await _asyncio.wait_for(
                _asyncio.to_thread(ak.stock_zyjs_ths, symbol=code_only), timeout=12
            )
            if df is not None and not df.empty:
                row  = df.iloc[0]
                desc = str(row.get("主营业务", "") or "").strip()
                prod = str(row.get("产品类型", "") or "").strip()
                if desc and len(desc) > 5:
                    found_desc = desc
                    if prod and not found_ind:
                        found_ind = prod.split("、")[0].strip()[:20]
        except Exception as e:
            logger.debug(f"AKShare zyjs_ths {ts_code} 失败: {e}")

    # ── ② 巨潮 stock_profile_cninfo（主营业务 + 机构简介 + 标准行业分类）─────
    # 无论①是否成功都调：①只给简介，②额外提供标准行业分类和机构背景
    if profile_source in ("cninfo", "both"):
        try:
            df2 = await _asyncio.wait_for(
                _asyncio.to_thread(ak.stock_profile_cninfo, symbol=code_only), timeout=12
            )
            if df2 is not None and not df2.empty:
                row2 = df2.iloc[0]

                # 行业：巨潮所属行业是国家标准行业分类，比东方财富更准确
                ind2 = str(row2.get("所属行业", "") or "").strip()
                if ind2:
                    found_ind = ind2  # 巨潮行业优先覆盖

                # 简介：主营业务 + 机构简介合并，提供更丰富的语义信息
                if not found_desc:
                    main_biz = str(row2.get("主营业务", "") or "").strip()
                    intro    = str(row2.get("机构简介", "") or "").strip()
                    # 机构简介取前200字（含公司背景、核心业务等关键信息）
                    intro = intro[:200] if intro else ""
                    if main_biz and intro:
                        found_desc = main_biz + " " + intro
                    else:
                        found_desc = main_biz or intro
                else:
                    # ①已有简介，用机构简介补充背景（追加，不替换）
                    intro = str(row2.get("机构简介", "") or "").strip()[:150]
                    if intro and len(found_desc) < 50:
                        # 简介太短时用机构简介补充
                        found_desc = found_desc + " " + intro

        except Exception as e:
            logger.debug(f"巨潮 {ts_code} 失败: {e}")

    # 有简介就返回
    if found_desc and len(found_desc) > 5:
        kws = list(dict.fromkeys(
            [w for w in _re.split(r"[、，；。 ]+", found_desc) if 2 <= len(w) <= 8]
        ))[:10]
        return found_desc, [found_ind] if found_ind else [], kws, False, found_ind

    # ── ③ LLM 兜底（仅在所有数据接口全部失败时）────────────────────────────
    try:
        from backend.services.llm_client import get_llm_client as _get_llm
        _client = await _get_llm()
    except Exception:
        _client = None

    if _client:
        prompt = (f'请为A股上市公司"{name}"（行业：{found_ind or "未知"}）提供简短主营业务描述。'
                  f'返回JSON（不加markdown）：'
                  f'{{"business_desc":"主营业务（80字内）","domains":["领域1","领域2"],"keywords":["关键词1","关键词2","关键词3"]}}')
        try:
            resp = await _client.chat([{"role": "user", "content": prompt}])
            text = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(text)
            desc = data.get("business_desc", "")
            if desc:
                return desc, data.get("domains", []), data.get("keywords", []), True, found_ind
        except Exception as e:
            logger.debug(f"LLM兜底 {name} 失败: {e}")

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
