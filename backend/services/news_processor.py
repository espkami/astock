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

# ── keywords 规则（抽取为常量，批量和单条 fallback 共用，保持一致）────────────
# 修复点①：原来 fallback 单条 prompt 里没有 keywords 规则，导致 fallback 质量急剧下降
_KEYWORDS_RULES = """
keywords 规则（严格遵守）：
① 只提取能直接匹配A股公司主营业务的**名词词根**，2-6字
② 同一概念必须拆成多个词根：大米相关→["大米","水稻","稻谷","粮食","种子"]；芯片相关→["芯片","半导体","集成电路","晶圆"]
③ 禁止描述性词组：禁止"风味下降""品种变化""选育方向""口感变化""品质问题"等
④ 禁止通用词：禁止"上市公司""市场""发展""创新""政策""行业""企业"
⑤ 与A股无直接关联则返回[]
示例：新闻"大米越来越难吃，水稻品种选育问题"→keywords:["大米","水稻","稻谷","粮食","种子","粮食加工"]"""

# ── 批量分类 Prompt ────────────────────────────────────────────────────────────
BATCH_PROMPT_TEMPLATE = """{role_desc}

分析以下{count}条新闻，返回JSON数组，长度必须={count}，不加markdown：

{news_list}

每条格式：{{"id":序号,"summary":"不超过60字","sentiment":"positive/negative/neutral","industries":["行业"],"event_type":"政策利好/技术突破/业绩超预期/利空消息/行业动态/其他","keywords":["词1","词2","词3"]}}{sentiment_addon}
{keywords_rules}"""

# 修复点②：content 截取从 150 字提升到 500 字
# 原来 150 字严重不够，产业链受益逻辑（如"赛力斯制造/宁德配套"）全在正文里，截断后 LLM 只能猜
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
                await asyncio.sleep(5)
        if not _skip_reset:
            done_so_far = min(chunk_start + BATCH, total)
            _set_classify_progress(done_so_far, total,
                f"分类中 {done_so_far}/{total}（已完成 {processed} 条）")
        if chunk_start + BATCH < total:
            await asyncio.sleep(20)

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
        f"【新闻{i+1}】标题：{row['title']}\n内容：{(row['content'] or '')[:_CONTENT_LIMIT]}"
        for i, row in enumerate(rows)
    ])

    prompt = BATCH_PROMPT_TEMPLATE.format(
        role_desc=role_desc,
        count=count,
        news_list=news_list,
        sentiment_addon=sentiment_addon,
        keywords_rules=_KEYWORDS_RULES,  # 修复点①：规则抽出为常量复用
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
                       summary=?, sentiment=?, industries=?, event_type=?, keywords=?, confidence=?
                       WHERE id=?""",
                    (
                        item.get("summary", ""),
                        item.get("sentiment", "neutral"),
                        json.dumps(item.get("industries", []), ensure_ascii=False),
                        item.get("event_type", ""),
                        json.dumps(item.get("keywords", []), ensure_ascii=False),
                        0.9,
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
    # 修复点①②：content 提升至 500 字，且加入与批量版相同的 keywords 规则
    prompt = f"""{role_desc}

分析新闻，返回JSON（不加markdown）：
标题：{title}
内容：{content[:_CONTENT_LIMIT]}

格式：{{"summary":"不超过60字","sentiment":"positive/negative/neutral","industries":["行业"],"event_type":"政策利好/技术突破/业绩超预期/利空消息/行业动态/其他","keywords":["词1","词2","词3"]}}{sentiment_addon}
{_KEYWORDS_RULES}"""

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
            """UPDATE news SET summary=?, sentiment=?, industries=?, event_type=?, keywords=?, confidence=? WHERE id=?""",
            (data.get("summary",""), data.get("sentiment","neutral"),
             json.dumps(data.get("industries",[]), ensure_ascii=False),
             data.get("event_type",""),
             json.dumps(data.get("keywords",[]), ensure_ascii=False),
             float(data.get("confidence", 0.9)), news_id),
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
