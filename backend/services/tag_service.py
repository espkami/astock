"""股票标签服务 — 用 LLM 批量生成结构化标签，提升匹配精准度"""
import json
import asyncio
from loguru import logger
from backend.database import get_db

# 全局进度
_tag_progress = {"stage": "idle", "current": 0, "total": 0,
                 "percent": 0.0, "message": "空闲", "done": True, "error": None}

def get_tag_progress() -> dict:
    return dict(_tag_progress)

def _set_tag_progress(current, total, message, done=False, error=None):
    _tag_progress.update({
        "stage": "tagging",
        "current": current, "total": total,
        "percent": round(current / total * 100, 1) if total > 0 else 0,
        "message": message, "done": done, "error": error,
    })


async def init_tags_table():
    """建 stock_tags 表"""
    async with get_db() as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS stock_tags (
                ts_code     TEXT PRIMARY KEY,
                products    TEXT DEFAULT '[]',  -- 核心产品/服务
                techs       TEXT DEFAULT '[]',  -- 技术/工艺
                sectors     TEXT DEFAULT '[]',  -- 细分行业
                chain_pos   TEXT DEFAULT '[]',  -- 产业链位置（上/中/下游）
                themes      TEXT DEFAULT '[]',  -- 热点主题（AI算力/新能源/国产替代等）
                all_tags    TEXT DEFAULT '[]',  -- 所有标签合并（用于快速搜索）
                updated_at  TEXT
            )
        """)
        await db.commit()
    logger.info("stock_tags 表初始化完成")


async def _fetch_em_mainbz(ts_code: str) -> str:
    """用 AKShare stock_zygc_em 获取主营构成（东方财富数据源，免费）"""
    import asyncio as _asyncio
    # 转换代码格式：600127.SH → SH600127
    parts = ts_code.split(".")
    if len(parts) != 2:
        return ""
    em_code = parts[1] + parts[0]

    try:
        import akshare as ak
        df = await _asyncio.wait_for(
            _asyncio.to_thread(ak.stock_zygc_em, symbol=em_code),
            timeout=12
        )
        if df is None or df.empty:
            return ""

        # 取最新年报
        df_sorted = df.sort_values("报告日期", ascending=False)
        latest_date = df_sorted["报告日期"].iloc[0]
        latest = df_sorted[df_sorted["报告日期"] == latest_date]

        NOISE = {"其他", "其他主营业务", "其他产品", "其他业务", "综合", "其他(补充)", "其它"}

        # 优先按产品分类（具体产品名，如"动力电池系统"），再按行业分类（如"电气机械及器材制造业"）
        # 按产品分类更适合做标签，行业分类是国民经济大类，粒度太粗
        for cat in ["按产品分类", "按行业分类"]:
            items = latest[latest["分类类型"] == cat].sort_values("收入比例", ascending=False)
            if not items.empty:
                lines = []
                for _, row in items.head(8).iterrows():
                    name = str(row.get("主营构成", "")).strip()
                    ratio = float(row.get("收入比例", 0)) * 100
                    for suf in ["分部", "业务", "板块", "行业", "产业"]:
                        if name.endswith(suf) and len(name) > len(suf) + 1:
                            name = name[:-len(suf)]
                    if name and name not in NOISE and ratio > 1 and len(name) <= 15:
                        lines.append(f"{name}({ratio:.0f}%)")
                if lines:
                    return "；".join(lines)
        return ""
    except Exception as e:
        logger.debug(f"AKShare 主营构成获取失败 {ts_code}: {e}")
        return ""


# 行业词扩展映射：东方财富分类名 → 更通用的匹配词
_INDUSTRY_EXPAND = {
    # ── 军工/国防（雪球行业词 + 东方财富分类名）────────────────────────────
    "军工装备":   ["军工", "国防", "航空", "武器", "军工装备"],
    "航空制造业": ["航空", "军工", "国防", "航空制造"],
    "航空产品":   ["航空", "军工", "国防"],
    "船舶制造":   ["船舶", "军工", "国防", "造船"],
    "兵器制造":   ["军工", "国防", "兵器"],
    "军工":       ["军工", "国防", "武器"],
    "军事":       ["军工", "国防", "军事"],
    "国防":       ["国防", "军工"],

    # ── 贵金属/黄金 ──────────────────────────────────────────────────────
    "贵金属":     ["黄金", "贵金属", "避险", "白银"],
    "黄金":       ["黄金", "贵金属", "避险"],
    "黄金行业":   ["黄金", "贵金属", "避险"],

    # ── 能源 ─────────────────────────────────────────────────────────────
    "石油加工贸易": ["石油", "能源", "油气", "炼化"],
    "石油":       ["石油", "能源", "油气"],
    "天然气":     ["天然气", "能源", "油气"],
    "煤炭":       ["煤炭", "能源"],
    "煤炭开采加工": ["煤炭", "能源"],
    "电力":       ["电力", "能源"],
    "新能源":     ["新能源", "光伏", "储能"],
    "光伏":       ["光伏", "新能源", "太阳能"],
    "光伏设备":   ["光伏", "新能源", "太阳能"],
    "储能":       ["储能", "新能源", "电池"],
    "电池":       ["电池", "储能", "新能源", "锂电池"],

    # ── 半导体/科技 ──────────────────────────────────────────────────────
    "半导体":     ["半导体", "芯片", "集成电路"],
    "芯片":       ["芯片", "半导体", "集成电路"],
    "光学光电子": ["光学", "半导体", "光电"],
    "通信设备":   ["通信", "5G", "通信设备"],
    "通信服务":   ["通信", "运营商"],
    "软件开发":   ["软件", "IT", "互联网"],
    "IT服务":     ["IT", "软件", "信息技术"],
    "人工智能":   ["人工智能", "AI", "大模型"],
    "大模型":     ["大模型", "人工智能", "AI"],

    # ── 有色金属/大宗商品 ────────────────────────────────────────────────
    "铜":         ["铜", "有色金属", "大宗商品"],
    "铝":         ["铝", "有色金属", "大宗商品"],
    "锂":         ["锂", "锂电池", "新能源"],
    "小金属":     ["稀有金属", "有色金属", "大宗商品"],
    "钢铁":       ["钢铁", "黑色金属", "大宗商品"],

    # ── 化工/材料 ────────────────────────────────────────────────────────
    "化工":       ["化工", "化学品"],
    "化学制品":   ["化工", "化学品"],
    "化学制药":   ["医药", "制药", "化学药"],
    "农化制品":   ["农药", "化肥", "农化"],

    # ── 医药/医疗 ────────────────────────────────────────────────────────
    "医疗器械":   ["医疗器械", "医疗", "医药"],
    "医疗服务":   ["医疗", "医院", "医疗服务"],
    "化学制药":   ["医药", "制药", "化学药"],
    "中药":       ["中药", "医药", "中医"],
    "生物药":     ["生物药", "医药", "制药"],
    "医药":       ["医药", "医疗", "制药"],
    "医药商业":   ["医药", "医疗", "药品流通"],

    # ── 消费/食品 ────────────────────────────────────────────────────────
    "白酒":       ["白酒", "酒类", "消费"],
    "食品加工制造": ["食品", "消费"],

    # ── 汽车 ─────────────────────────────────────────────────────────────
    "汽车零部件": ["汽车", "零部件", "汽车零部件"],
    "汽车整车":   ["汽车", "新能源汽车"],
}

def _parse_em_tags(mainbz: str, industry: str) -> dict:
    """直接把东方财富主营构成解析为结构化标签（不调LLM）
    输入: "大米加工与销售分部(36%)；种植业分部(36%)；农资服务(21%)"
    同时扩展行业词，确保通用关键词也能被匹配到。
    """
    import re

    raw_items = [item.strip() for item in mainbz.split("；") if item.strip()]

    NOISE_WORDS = {"其他", "其他主营业务", "其他产品", "其他业务", "综合", "合并抵销", "其他(补充)"}
    SUFFIX_STRIP = ["分部", "业务", "板块", "事业部", "行业", "产业"]

    products = []
    for item in raw_items:
        name = re.sub(r'[(]\d+%[)]', '', item).strip()
        for suffix in SUFFIX_STRIP:
            if name.endswith(suffix) and len(name) > len(suffix) + 1:
                name = name[:-len(suffix)].strip()
        if name and name not in NOISE_WORDS and len(name) <= 15:
            products.append(name)

    products = list(dict.fromkeys(products))[:8]

    # 行业词扩展：把东方财富分类名映射到通用词
    # 使用精确匹配或前缀匹配，避免"制造业"误触发"航空制造"
    def _match_key(text, key):
        """精确匹配：text==key，或 text 以 key 开头（如"黄金行业"匹配"黄金"）"""
        return text == key or text.startswith(key) or key == text

    expanded = list(products)
    for p in products:
        for key, synonyms in _INDUSTRY_EXPAND.items():
            if _match_key(p, key) or _match_key(key, p):
                expanded.extend(synonyms)
    # 也对 industry 字段做扩展
    if industry:
        for key, synonyms in _INDUSTRY_EXPAND.items():
            if _match_key(industry, key) or _match_key(key, industry):
                expanded.extend(synonyms)

    sectors = [industry] if industry else []
    all_tags = list(dict.fromkeys(expanded + sectors))[:15]

    return {
        "products":  products,
        "techs":     [],
        "sectors":   sectors,
        "chain_pos": [],
        "themes":    [],
        "all_tags":  all_tags,
    }


TAG_PROMPT = """你是 A 股投研专家。根据以下股票信息，生成结构化标签。

股票名称：{name}
申万行业：{industry}
主营业务描述：{business_desc}
主营产品构成（按收入占比）：{mainbz}

请返回严格 JSON（不加 markdown），字段说明：
- products: 核心产品/服务（最细粒度，如"大米""锂电池""光刻机""CAD软件"，3-6个）
- techs: 核心技术/工艺（如"IGBT""HBM""mRNA""射频"，0-4个，无则空数组）
- sectors: 细分行业（如"水稻种植""动力电池""半导体设备"，2-4个）
- chain_pos: 产业链位置（从["上游原材料","中游制造","下游应用","流通服务","综合"]中选1-2个）
- themes: 相关热点主题（如"AI算力""新能源""国产替代""消费复苏""出海"，0-3个，严格相关才填）

示例输出：
{{"products":["大米","粮油制品"],"techs":[],"sectors":["水稻种植","粮食加工"],"chain_pos":["中游制造"],"themes":["粮食安全"]}}

注意：
- products 必须是具体产品名，不能是"产品""服务"等泛词
- 不相关的 themes 宁可不填，不要强行凑数
- 所有标签用中文"""


async def generate_tags_for_stock(ts_code: str, name: str, industry: str,
                                   business_desc: str, client) -> dict | None:
    """方案C：有东方财富数据直接解析为标签，无数据才调LLM兜底"""
    if not business_desc or len(business_desc) < 5:
        return None

    # ── 优先：东方财富主营构成（免费，产品级别，直接解析） ──────────────────
    mainbz = await _fetch_em_mainbz(ts_code)
    if mainbz:
        return _parse_em_tags(mainbz, industry)

    # ── 兜底：调 LLM（约20%无东方财富数据的股票）──────────────────────────
    if not client:
        return None
    prompt = TAG_PROMPT.format(
        name=name, industry=industry or "未知",
        business_desc=business_desc[:200],
        mainbz="（暂无数据）",
    )
    try:
        resp = await client.chat([{"role": "user", "content": prompt}], timeout=20)
        text = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(text)
        # 合并所有标签
        all_tags = list(dict.fromkeys(
            data.get("products", []) +
            data.get("techs", []) +
            data.get("sectors", []) +
            data.get("themes", [])
        ))
        return {
            "products":  data.get("products", []),
            "techs":     data.get("techs", []),
            "sectors":   data.get("sectors", []),
            "chain_pos": data.get("chain_pos", []),
            "themes":    data.get("themes", []),
            "all_tags":  all_tags,
        }
    except Exception as e:
        logger.debug(f"标签生成失败 {ts_code}: {e}")
        return None


async def generate_all_tags(force: bool = False) -> dict:
    """批量生成所有股票标签"""
    from backend.services.llm_client import get_llm_client
    await init_tags_table()

    client = await get_llm_client()
    if not client:
        return {"success": False, "error": "未配置分析匹配模型"}

    async with get_db() as db:
        if force:
            # 强制重新生成：清空旧标签
            await db.execute("DELETE FROM stock_tags")
            await db.commit()
            async with db.execute("""
                SELECT s.ts_code, s.name, s.industry, COALESCE(sp.business_desc,'') as business_desc
                FROM stocks s LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                WHERE sp.business_desc IS NOT NULL AND length(sp.business_desc) >= 5
            """) as cur:
                rows = await cur.fetchall()
        else:
            # 只处理没有标签的
            async with db.execute("""
                SELECT s.ts_code, s.name, s.industry, COALESCE(sp.business_desc,'') as business_desc
                FROM stocks s
                LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                LEFT JOIN stock_tags st ON s.ts_code=st.ts_code
                WHERE st.ts_code IS NULL
                  AND sp.business_desc IS NOT NULL
                  AND length(sp.business_desc) >= 5
            """) as cur:
                rows = await cur.fetchall()

    if not rows:
        _set_tag_progress(0, 0, "✅ 所有股票标签已是最新", done=True)
        return {"success": True, "count": 0}

    total = len(rows)
    _set_tag_progress(0, total, f"准备为 {total} 只股票生成标签...")
    logger.info(f"开始生成标签: {total} 只")

    filled = 0
    failed = 0
    done_count = 0
    lock = asyncio.Lock()

    # 东方财富接口需要限速，降低并发避免被封
    CONCURRENCY = 5
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _process(row):
        nonlocal filled, failed, done_count
        async with sem:
            tags = await generate_tags_for_stock(
                row["ts_code"], row["name"],
                row["industry"] or "", row["business_desc"],
                client
            )
            async with lock:
                done_count += 1
                if tags:
                    async with get_db() as db2:
                        await db2.execute("""
                            INSERT OR REPLACE INTO stock_tags
                            (ts_code, products, techs, sectors, chain_pos, themes, all_tags, updated_at)
                            VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                        """, (
                            row["ts_code"],
                            json.dumps(tags["products"],  ensure_ascii=False),
                            json.dumps(tags["techs"],     ensure_ascii=False),
                            json.dumps(tags["sectors"],   ensure_ascii=False),
                            json.dumps(tags["chain_pos"], ensure_ascii=False),
                            json.dumps(tags["themes"],    ensure_ascii=False),
                            json.dumps(tags["all_tags"],  ensure_ascii=False),
                        ))
                        await db2.commit()
                    filled += 1
                else:
                    failed += 1

                if done_count % 50 == 0 or done_count == total:
                    _set_tag_progress(done_count, total,
                        f"生成中 {done_count}/{total}（✅{filled} ❌{failed}）")
            await asyncio.sleep(0.3)  # 东方财富限速

    await asyncio.gather(*[_process(row) for row in rows])

    _set_tag_progress(total, total,
        f"✅ 标签生成完成 {filled}/{total}（失败 {failed} 只）", done=True)
    logger.info(f"标签生成完成: {filled}/{total}")
    return {"success": True, "count": filled, "failed": failed}


async def get_stock_tags(ts_code: str) -> dict:
    """获取单只股票的标签"""
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM stock_tags WHERE ts_code=?", (ts_code,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {}
    return {
        "products":  json.loads(row["products"]  or "[]"),
        "techs":     json.loads(row["techs"]     or "[]"),
        "sectors":   json.loads(row["sectors"]   or "[]"),
        "chain_pos": json.loads(row["chain_pos"] or "[]"),
        "themes":    json.loads(row["themes"]    or "[]"),
        "all_tags":  json.loads(row["all_tags"]  or "[]"),
    }
