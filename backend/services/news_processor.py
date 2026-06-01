"""新闻处理模块 — 批量大模型精分类（多条合并一次调用，大幅节省 Token）"""
import json
import asyncio
from loguru import logger
from backend.database import get_db
from backend.services.config_service import get_config

# 全局分类进度
_classify_progress = {
    "running": False, "current": 0, "total": 0,
    "percent": 0.0, "message": "空闲", "done": True, "error": None
}

def get_classify_progress() -> dict:
    return dict(_classify_progress)

def _set_classify_progress(current: int, total: int, message: str,
                           running: bool = True, done: bool = False, error=None):
    _classify_progress.update({
        "running": running,
        "current": current, "total": total,
        "percent": round(current / total * 100, 1) if total > 0 else 0,
        "message": message, "done": done, "error": error,
    })

# 各分析偏好的 Prompt 模板
FOCUS_PROMPTS = {
    "stock": "你是 A 股投资分析师，专注判断新闻对个股和板块的直接影响。",
    "industry": "你是产业研究分析师，专注行业趋势和产业链变化。",
    "macro": "你是宏观经济分析师，专注政策面和宏观经济对 A 股的影响。",
    "balanced": "你是 A 股新闻分析师，全面提取行业、情感和关键词。",
}

SENTIMENT_ADDONS = {
    "strict":    "\n注意：情感判断要严格，只有新闻明确提到对股市/个股有直接正面或负面影响时才标注 positive/negative，其余一律 neutral。",
    "normal":    "",
    "sensitive": "\n注意：情感判断要敏感，潜在的、间接的影响也应体现在 positive/negative 判断中，尽量减少 neutral 标注。",
}

DEFAULT_CLASSIFY_PROMPT = FOCUS_PROMPTS["balanced"]  # 向后兼容

# ══════════════════════════════════════════════════════
# 方法论框架常量（年化30%高手决策逻辑）
# ══════════════════════════════════════════════════════

# 新闻级别判断标准
_NEWS_LEVEL_GUIDE = """
新闻级别判断：
- 高：领导调研/政策定调、企业大额投资/扩产(>10亿)、重大并购重组、技术突破/重大订单
- 中：工商变更/注册资本增减、中小额投资、行业公告/声明、融资计划
- 低：个人言论/宣传、无量级普通消息、行业内日常动态"""

# 产业链受益词提取规则（年化30%高手选股框架）
_BENEFICIARY_RULES = """
beneficiary_chain 提取规则：找"钱最终花到哪里"的受益方

【核心思维框架：三层受益拆解】

第一层：运力/工具层（最直接，最容易被忽视）
  - 配送平台扩张/外卖增长 → 电动两轮车、电动三轮车（骑手的核心工具）
  - 无人配送布局 → 无人车、配送机器人、激光雷达
  - 网约车/出行扩张 → 汽车、新能源车
  - 工厂自动化 → 工业机器人、AGV、机械臂

第二层：基础设施层（IT/能源/仓储）
  - 互联网平台扩张 → IDC数据中心、服务器、云计算
  - 配送量增加 → 智能仓储、AGV机器人、分拣设备
  - 数字化升级 → ERP系统、SaaS软件、物联网模组
  - 新能源扩张 → 充换电站、储能设备

第三层：消耗品/耗材层（量价齐升逻辑）
  - 配送量增加 → 快递包装、纸箱、胶带、包装耗材
  - 骑手增加 → 头盔、雨衣、配送箱（保温箱）
  - 电动车增加 → 锂电池、充电器、电机

【典型场景示例】
场景1：配送平台增资/扩张（蜂鸟即配/美团/饿了么）
  → 运力层：[电动两轮车, 电动三轮车, 电动摩托车]  ← 最直接，骑手工具
  → 基础设施：[IDC数据中心, 物联网模组, AGV机器人, 智能仓储, SaaS软件]
  → 消耗品：[快递包装, 锂电池, 保温箱]

场景2：新能源汽车扩产
  → 上游材料：[锂电池, 碳酸锂, 铜箔, 隔膜, 正极材料]
  → 制造设备：[工业机器人, 激光焊接, 锂电设备]
  → 配套：[汽车零部件, 充电桩]

场景3：AI/大模型公司融资/扩张
  → 算力层：[IDC数据中心, GPU服务器, 液冷设备, 光模块]
  → 软件层：[AI芯片, EDA工具, AI软件]
  → 能源：[UPS电源, 变压器, 电力设备]

场景4：地产政策松绑/基建投资
  → 建材：[水泥, 钢铁, 玻璃, 防水材料]
  → 家居消费：[家电, 家具, 装修材料]
  → 工程机械：[挖掘机, 起重机]

场景5：消费品牌出海/扩张
  → 供应链：[ODM制造, 包装印刷, 跨境物流]
  → 营销：[广告投放, 社媒平台]

【排除规则】
- 不填新闻主体的同行竞争对手
- 不填泛行业词（如"互联网""消费"）
- 优先填具体产品词，能让股票标签直接命中"""

# 时间维度判断
_TIME_HORIZON_GUIDE = """
time_horizon 判断：
- short（短线1-4周）：政策概念/题材热度、高级别调研、地区概念炒作
- medium（中线1-6月）：企业扩产落地/订单兑现、融资扩张、合作签约
- long（长线1-3年）：技术突破、产业链战略扩张、政策正式落地"""

# keywords 规则
_KEYWORDS_RULES = """
keywords 规则（严格遵守）：
【核心原则】keywords 反映新闻对A股的影响方向，而不是公司经营范围描述
  - 正确：蜂鸟即配增资 → ["即时配送","同城配送","快递物流","配送平台"]
  - 错误：不要提取工商登记里的"广告制作""信息系统集成"等次要经营范围
① 只提取能直接匹配A股公司主营业务的名词词根，2-6字
② 同一概念拆成多个词根：即时配送→["即时配送","同城配送","快递物流","配送平台"]
③ 禁止从公司工商登记经营范围提取词汇，要从新闻核心事件和影响提取
④ 禁止通用词：禁止"上市公司""市场""发展""创新""政策""行业""企业""信息技术""广告"
⑤ 与A股无直接关联则返回[]"""

# ── 批量分类 Prompt（注入方法论七步框架）────────────────────────────────
BATCH_PROMPT_TEMPLATE = """{role_desc}

你是一位年化30%的A股高手，用以下七步框架分析新闻：
1. 新闻级别判断（高/中/低）
2. 提炼核心关键词（行业+技术+地区+动作+量化）
3. 资金流向分析（钱最终花到哪里）
4. 产业链受益词（直接受益方的行业词）
5. 排除竞争方（新闻主体的同行不是受益方）
6. 时间维度（短/中/长线）
7. 情感判断（对受益方是positive/negative/neutral）

分析以下{count}条新闻，返回JSON数组，长度必须={count}，不加markdown：

{news_list}

每条格式：{{"id":序号,"summary":"不超过60字","sentiment":"positive/negative/neutral","industries":["受益行业"],"event_type":"政策利好/技术突破/业绩超预期/利空消息/行业动态/其他","keywords":["受益方行业词1","词2"],"news_level":"高/中/低","beneficiary_chain":["产业链受益词1","词2","词3"],"time_horizon":"short/medium/long"}}{sentiment_addon}
{keywords_rules}
{beneficiary_rules}
{time_horizon_guide}"""

# content 截取限制
_CONTENT_LIMIT = 500


async def process_pending_news(batch_size: int = 10, _skip_reset: bool = False) -> int:
    """处理未分类的新闻，批量调用大模型，返回处理条数"""
    from backend.services.llm_client import get_llm_client

    client = await get_llm_client()
    if not client:
        logger.warning("新闻处理：未配置大模型，跳过")
        _set_classify_progress(0, 0, "未配置大模型", running=False, done=True, error="未配置大模型")
        return 0

    focus         = await get_config("analysis_focus", "balanced")
    sentiment_std = await get_config("sentiment_standard", "normal")
    role_desc     = FOCUS_PROMPTS.get(focus, FOCUS_PROMPTS["balanced"])
    sentiment_addon = SENTIMENT_ADDONS.get(sentiment_std, "")

    async with get_db() as db:
        async with db.execute(
            "SELECT id, title, content FROM news WHERE summary IS NULL LIMIT ?",
            (batch_size,),
        ) as cur:
            rows = await cur.fetchall()

    if not rows:
        if not _skip_reset:
            _set_classify_progress(0, 0, "✅ 最新新闻均已分类完毕", running=False, done=True)
        return 0

    total = len(rows)
    if not _skip_reset:
        _set_classify_progress(0, total, f"准备分类 {total} 条新闻...")

    # ── 批量处理：每次最多 20 条合并为一个请求 ──────────────────────────────
    BATCH = 10
    processed = 0
    for chunk_start in range(0, total, BATCH):
        chunk = rows[chunk_start: chunk_start + BATCH]
        try:
            n = await _classify_batch(client, role_desc, sentiment_addon, chunk)
            processed += n
        except Exception as e:
            err_str = str(e)
            logger.warning(f"批量分类失败（条目 {chunk_start}-{chunk_start+len(chunk)}）: {err_str[:80]}，降级为单条处理")
            for row in chunk:
                try:
                    ok = await _classify_one_fallback(client, role_desc, sentiment_addon,
                                                       row["id"], row["title"], row["content"] or "")
                    if ok:
                        processed += 1
                except Exception as e2:
                    err_str2 = str(e2)
                    if "400" in err_str2:
                        await _mark_blocked(row["id"])
                    else:
                        logger.warning(f"单条分类跳过 {row['id']}: {err_str2[:60]}")
                await asyncio.sleep(2)  # 单条请求小，2秒间隔足够
        if not _skip_reset:
            done_so_far = min(chunk_start + BATCH, total)
            _set_classify_progress(done_so_far, total,
                f"分类中 {done_so_far}/{total}（已完成 {processed} 条）")
        if chunk_start + BATCH < total:
            await asyncio.sleep(5)  # 有多模型故障转移，5秒足够

    if not _skip_reset:
        _set_classify_progress(total, total,
            f"✅ 分类完成，共处理 {processed} 条" if processed > 0 else "✅ 最新新闻均已分类完毕", running=False, done=True)
    logger.info(f"新闻分类完成: {processed}/{total}")
    return total


async def _classify_batch(client, role_desc: str, sentiment_addon: str, rows: list) -> int:
    """批量分类：多条新闻合并为一次 LLM 调用"""
    count = len(rows)
    # 修复点②：content 截取提升到 _CONTENT_LIMIT（500字）
    news_list = "\n\n".join([
        f"【新闻{i+1}】标题：{row['title']}\n内容：{(row['content'] or '')[:1000]}"
        for i, row in enumerate(rows)
    ])

    prompt = BATCH_PROMPT_TEMPLATE.format(
        role_desc=role_desc,
        count=count,
        news_list=news_list,
        sentiment_addon=sentiment_addon,
        keywords_rules=_KEYWORDS_RULES,
        beneficiary_rules=_BENEFICIARY_RULES,
        time_horizon_guide=_TIME_HORIZON_GUIDE,
    )

    resp = await client.chat([
        {"role": "system", "content": "只返回 JSON 数组，不加任何说明。"},
        {"role": "user",   "content": prompt},
    ], json_mode=False, timeout=180)

    # 解析响应
    text = resp.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    results = json.loads(text)
    if not isinstance(results, list):
        raise ValueError(f"期望 JSON 数组，实际: {type(results)}")

    # 写入数据库
    saved = 0
    async with get_db() as db:
        for item in results:
            idx = int(item.get("id", 0)) - 1
            if 0 <= idx < len(rows):
                news_id = rows[idx]["id"]
                await db.execute(
                    """UPDATE news SET
                       summary=?, sentiment=?, industries=?, event_type=?, keywords=?, confidence=?,
                       news_level=?, beneficiary_chain=?, time_horizon=?
                       WHERE id=?""",
                    (
                        item.get("summary", ""),
                        item.get("sentiment", "neutral"),
                        json.dumps(item.get("industries", []), ensure_ascii=False),
                        item.get("event_type", ""),
                        json.dumps(item.get("keywords", []), ensure_ascii=False),
                        0.9,
                        item.get("news_level", "medium"),
                        json.dumps(item.get("beneficiary_chain", []), ensure_ascii=False),
                        item.get("time_horizon", "short"),
                        news_id,
                    ),
                )
                saved += 1
        await db.commit()

    logger.info(f"批量分类 {count} 条 → 成功写入 {saved} 条（1次 LLM 调用）")
    return saved


async def _classify_one_fallback(client, role_desc: str, sentiment_addon: str,
                                  news_id: int, title: str, content: str) -> bool:
    """单条分类 fallback（批量失败时使用）"""
    prompt = f"""{role_desc}

你是一位年化30%的A股高手，用资金流向思维分析新闻受益方。
分析新闻，返回JSON（不加markdown）：
标题：{title}
内容：{content}

格式：{{"summary":"不超过60字","sentiment":"positive/negative/neutral","industries":["受益行业"],"event_type":"政策利好/技术突破/业绩超预期/利空消息/行业动态/其他","keywords":["受益方行业词"],"news_level":"高/中/低","beneficiary_chain":["产业链受益词1","词2","词3"],"time_horizon":"short/medium/long"}}{sentiment_addon}
{_KEYWORDS_RULES}
{_BENEFICIARY_RULES}
{_TIME_HORIZON_GUIDE}"""

    resp = await client.chat([
        {"role": "system", "content": "只返回 JSON，不加任何说明。"},
        {"role": "user",   "content": prompt},
    ], json_mode=False, timeout=180)

    text = resp.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    data = json.loads(text)
    async with get_db() as db:
        await db.execute(
            """UPDATE news SET summary=?, sentiment=?, industries=?, event_type=?, keywords=?, confidence=?,
               news_level=?, beneficiary_chain=?, time_horizon=? WHERE id=?""",
            (data.get("summary",""), data.get("sentiment","neutral"),
             json.dumps(data.get("industries",[]), ensure_ascii=False),
             data.get("event_type",""),
             json.dumps(data.get("keywords",[]), ensure_ascii=False),
             float(data.get("confidence", 0.9)),
             data.get("news_level", "medium"),
             json.dumps(data.get("beneficiary_chain", []), ensure_ascii=False),
             data.get("time_horizon", "short"),
             news_id),
        )
        await db.commit()
    return True


async def _mark_blocked(news_id: int):
    """400 内容审核：写入占位，避免反复重试"""
    async with get_db() as db:
        await db.execute(
            "UPDATE news SET summary=?, sentiment=?, industries=?, keywords=?, confidence=? WHERE id=?",
            ("[内容审核限制]", "neutral", "[]", "[]", 0.0, news_id)
        )
        await db.commit()
