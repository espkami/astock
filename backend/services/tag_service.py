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
        logger.warning(f"AKShare 主营构成获取失败 {ts_code}: {e}")
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

    # ── 快递/物流/即时配送 ───────────────────────────────────────────────
    "快递":       ["快递", "物流", "配送", "速运", "快运", "同城配送", "即时配送"],
    "快运":       ["快运", "快递", "物流", "货运"],
    "物流":       ["物流", "快递", "配送", "供应链"],
    "供应链":     ["供应链", "物流", "仓储", "配送"],
    "仓储":       ["仓储", "物流", "供应链"],
    "航运":       ["航运", "航海", "海运", "港口"],
    "港口":       ["港口", "航运", "海运"],
    "即时配送":   ["即时配送", "同城配送", "快递", "外卖配送", "配送平台"],
    "同城配送":   ["同城配送", "即时配送", "快递", "配送"],
    "外卖":       ["外卖", "即时配送", "同城配送", "餐饮"],

    # ── 零售/电商/消费 ───────────────────────────────────────────────────
    "零售":       ["零售", "商超", "百货", "消费"],
    "电商":       ["电商", "零售", "互联网", "网购"],
    "商超":       ["商超", "零售", "百货"],
    "百货":       ["百货", "零售", "商超"],
    "餐饮":       ["餐饮", "食品", "消费"],
    "旅游":       ["旅游", "酒店", "消费"],
    "酒店":       ["酒店", "旅游", "消费"],

    # ── 房地产/建筑 ──────────────────────────────────────────────────────
    "房地产":     ["房地产", "地产", "住宅", "商业地产"],
    "建筑":       ["建筑", "工程", "基建"],
    "建材":       ["建材", "水泥", "玻璃", "建筑"],
    "水泥":       ["水泥", "建材", "建筑"],

    # ── 农业/粮食 ────────────────────────────────────────────────────────
    "农业":       ["农业", "种植", "粮食", "农产品"],
    "种植业":     ["种植", "农业", "粮食"],
    "畜牧":       ["畜牧", "养殖", "农业"],
    "养殖":       ["养殖", "畜牧", "农业"],
    "渔业":       ["渔业", "水产", "农业"],

    # ── 银行/金融 ────────────────────────────────────────────────────────
    "银行":       ["银行", "金融", "存贷款"],
    "证券":       ["证券", "金融", "投行", "经纪"],
    "保险":       ["保险", "金融"],
    "信托":       ["信托", "金融", "资管"],

    # ── 传媒/游戏/教育 ───────────────────────────────────────────────────
    "传媒":       ["传媒", "媒体", "广告", "影视"],
    "游戏":       ["游戏", "互联网", "娱乐"],
    "教育":       ["教育", "培训", "在线教育"],
    "广告":       ["广告", "传媒", "营销"],
    "影视":       ["影视", "传媒", "娱乐"],

    # ── 纺织/服装 ────────────────────────────────────────────────────────
    "纺织":       ["纺织", "服装", "面料"],
    "服装":       ["服装", "纺织", "品牌消费"],

    # ── 机械/装备 ────────────────────────────────────────────────────────
    "工程机械":   ["工程机械", "机械", "设备"],
    "机械":       ["机械", "设备", "制造"],
    "工业自动化": ["工业自动化", "机器人", "智能制造"],
    "机器人":     ["机器人", "工业自动化", "智能制造"],
    "智能制造":   ["智能制造", "工业自动化", "机器人"],

    # ── IT基础设施/数据中心（互联网扩张受益场景）────────────────────────────
    "IDC":        ["IDC", "数据中心", "算力", "机房", "互联网基础设施"],
    "AIDC":       ["AIDC", "IDC", "数据中心", "AI算力", "智算中心"],
    "数据中心":   ["数据中心", "IDC", "算力", "机房"],
    "云计算":     ["云计算", "IDC", "数据中心", "SaaS"],
    "算力":       ["算力", "数据中心", "IDC", "AI算力", "GPU"],

    # ── 物联网/通信模组（配送/智能设备受益场景）────────────────────────────
    "物联网":     ["物联网", "IoT", "通信模组", "传感器", "智能终端"],
    "通信模组":   ["通信模组", "物联网", "IoT", "无线通信", "4G模组", "5G模组"],
    "IoT":        ["IoT", "物联网", "通信模组", "传感器"],
    "传感器":     ["传感器", "物联网", "IoT", "智能硬件"],
    "智能终端":   ["智能终端", "物联网", "IoT", "手持设备"],

    # ── ERP/数字化系统（企业扩张受益场景）──────────────────────────────────
    "ERP":        ["ERP", "数字化", "企业管理软件", "信息化", "供应链管理系统"],
    "数字化":     ["数字化", "ERP", "信息化", "企业软件", "SaaS"],
    "SaaS":       ["SaaS", "云计算", "数字化", "企业软件"],
    "信息化":     ["信息化", "数字化", "ERP", "IT服务"],

    # ── 激光雷达/自动驾驶（智能配送受益场景）───────────────────────────────
    "激光雷达":   ["激光雷达", "自动驾驶", "无人配送", "智能感知", "LIDAR"],
    "自动驾驶":   ["自动驾驶", "激光雷达", "无人驾驶", "智能汽车"],
    "无人配送":   ["无人配送", "激光雷达", "自动驾驶", "配送机器人"],
    "无人机":     ["无人机", "航空", "无人配送", "低空经济"],

    # ── GIS/地图/导航（配送平台受益场景）───────────────────────────────────
    "GIS":        ["GIS", "地图", "导航", "位置服务", "LBS"],
    "地图":       ["地图", "GIS", "导航", "位置服务"],

    # ── 包装材料（快递/电商扩张受益场景）───────────────────────────────────
    "包装":       ["包装", "纸箱", "快递包装", "塑料包装"],
    "纸箱":       ["纸箱", "包装", "快递包装", "瓦楞纸"],
    "快递包装":   ["快递包装", "包装", "纸箱"],

    # ── 冷链/温控（生鲜配送受益场景）───────────────────────────────────────
    "冷链":       ["冷链", "冷库", "冷藏运输", "温控物流"],

    # ── 电动两轮车/运力工具（即时配送/外卖扩张受益场景）──────────────────
    "电动两轮车": ["电动两轮车", "电动自行车", "电动摩托车", "两轮电动车", "配送车辆"],
    "电动自行车": ["电动自行车", "电动两轮车", "电动摩托车", "两轮电动车"],
    "电动摩托车": ["电动摩托车", "电动两轮车", "电动自行车"],
    "电动三轮车": ["电动三轮车", "电动两轮车", "配送车辆", "快递三轮"],
    "充换电":     ["充换电", "换电站", "电动两轮车", "骑手配套"],
    "配送车辆":   ["配送车辆", "电动两轮车", "电动三轮车", "物流车"],
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


TAG_PROMPT = """你是一位年化30%的A股高手兼投研专家。根据以下股票信息，生成结构化标签。

股票名称：{name}
申万行业：{industry}
主营业务描述：{business_desc}
主营产品构成（按收入占比）：{mainbz}

请返回严格 JSON（不加 markdown），字段说明：
- products: 核心产品/服务（市场通俗词，3-6个）
  【重要】必须用市场通俗词，不能用财务报表分类词：
  - 快递公司 → ["快递","快运","物流配送","同城配送","即时配送"]，非["派费收入","面单销售收入"]
  - 数据中心 → ["IDC","数据中心","算力","机房","服务器托管"]，非["机柜租赁收入"]
  - 通信模组 → ["通信模组","物联网","IoT","无线通信"]，非["模组产品收入"]
  - 银行 → ["存贷款","银行","金融服务"]，非["利息净收入"]
- techs: 核心技术/工艺（如"IGBT""HBM""激光雷达""5G模组"，0-4个）
- sectors: 细分行业（如"快递物流""数据中心""半导体设备"，2-4个）
- chain_pos: 产业链位置（从["上游原材料","中游制造","下游应用","流通服务","综合"]中选1-2个）
- themes: 受益场景标签（0-5个）
  【核心】按"三层受益框架"填写该股票在哪类新闻事件下受益：
  运力/工具层受益：
  - 电动两轮车/三轮车股 → ["即时配送扩张受益","外卖骑手增加受益","配送运力提升受益","共享电单车扩张受益"]
  - 工业机器人/AGV股 → ["工厂自动化受益","智能仓储受益","物流自动化受益"]
  - 无人车/激光雷达股 → ["无人配送受益","自动驾驶受益","智能物流受益"]
  基础设施层受益：
  - 数据中心/IDC股 → ["互联网扩张受益","AI算力受益","数字化投资受益"]
  - 通信模组/IoT股 → ["物联网扩张受益","智能设备普及受益","配送数字化受益"]
  - ERP/SaaS软件股 → ["企业数字化受益","互联网公司扩张受益","供应链升级受益"]
  消耗品层受益：
  - 快递包装/纸箱股 → ["电商扩张受益","快递量增长受益","即时配送受益"]
  - 锂电池/充电股 → ["电动车销量受益","配送运力扩张受益","新能源渗透受益"]
  - 冷链物流股 → ["生鲜配送受益","外卖扩张受益"]

示例输出（润泽科技）：
{{"products":["IDC","AIDC","数据中心","算力","机房"],"techs":[],"sectors":["数据中心","IDC"],"chain_pos":["流通服务"],"themes":["互联网扩张受益","AI算力受益","数字化投资受益","企业IT建设受益"]}}

示例输出（移远通信）：
{{"products":["通信模组","物联网","IoT","无线通信","4G模组","5G模组"],"techs":["5G","4G","LPWA"],"sectors":["通信模组","物联网"],"chain_pos":["中游制造"],"themes":["物联网扩张受益","智能设备普及受益","配送数字化受益","工业互联网受益"]}}

注意：
- products 必须是市场通俗词，让新闻 beneficiary_chain 能直接命中
- themes 是受益场景，填"XX受益"格式，让系统知道该股在什么情况下会涨
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
        logger.warning(f"标签生成失败 {ts_code}: {e}")
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
