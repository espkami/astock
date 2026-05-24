"""新闻处理模块 — 批量大模型精分类（多条合并一次调用，大幅节省 Token）"""
import json
import asyncio
from loguru import logger
from backend.database import get_db
from backend.services.config_service import get_config

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

# ── 批量分类 Prompt ────────────────────────────────────────────────────────────
BATCH_PROMPT_TEMPLATE = """{role_desc}

分析以下{count}条新闻，返回JSON数组，长度必须={count}，不加markdown：

{news_list}

每条格式：{{"id":序号,"summary":"不超过60字","sentiment":"positive/negative/neutral","industries":["行业"],"event_type":"政策利好/技术突破/业绩超预期/利空消息/行业动态/其他","keywords":["词1","词2","词3"]}}{sentiment_addon}

keywords 规则（严格遵守）：
① 只提取能直接匹配A股公司主营业务的**名词词根**，2-6字
② 同一概念必须拆成多个词根：大米相关→["大米","水稻","稻谷","粮食","种子"]；芯片相关→["芯片","半导体","集成电路","晶圆"]
③ 禁止描述性词组：禁止"风味下降""品种变化""选育方向""口感变化""品质问题"等
④ 禁止通用词：禁止"上市公司""市场""发展""创新""政策""行业""企业"
⑤ 与A股无直接关联则返回[]
示例：新闻"大米越来越难吃，水稻品种选育问题"→keywords:["大米","水稻","稻谷","粮食","种子","粮食加工"]"""


async def process_pending_news(batch_size: int = 20) -> int:
    """处理未分类的新闻，批量调用大模型，返回处理条数"""
    from backend.services.llm_client import get_llm_client

    client = await get_llm_client()
    if not client:
        logger.warning("新闻处理：未配置大模型，跳过")
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
        return 0

    # ── 批量处理：每次最多 10 条合并为一个请求 ──────────────────────────────
    BATCH = 20  # 每批20条合并1次调用，比10条节省~3% prompt overhead
    processed = 0
    for chunk_start in range(0, len(rows), BATCH):
        chunk = rows[chunk_start: chunk_start + BATCH]
        try:
            n = await _classify_batch(client, role_desc, sentiment_addon, chunk)
            processed += n
        except Exception as e:
            logger.error(f"批量分类失败（条目 {chunk_start}-{chunk_start+len(chunk)}）: {e}")
            # fallback：逐条处理
            for row in chunk:
                try:
                    ok = await _classify_one_fallback(client, role_desc, sentiment_addon,
                                                       row["id"], row["title"], row["content"] or "")
                    if ok:
                        processed += 1
                except Exception as e2:
                    if "400" in str(e2):
                        await _mark_blocked(row["id"])
                    logger.warning(f"单条分类跳过 {row['id']}: {type(e2).__name__}")
        # 批次间短暂等待，避免 429
        if chunk_start + BATCH < len(rows):
            await asyncio.sleep(2)

    logger.info(f"新闻分类完成: {processed}/{len(rows)}")

    return processed


async def _classify_batch(client, role_desc: str, sentiment_addon: str, rows: list) -> int:
    """批量分类：多条新闻合并为一次 LLM 调用"""
    count = len(rows)
    news_list = "\n\n".join([
        f"【新闻{i+1}】标题：{row['title']}\n内容：{(row['content'] or '')[:150]}"
        for i, row in enumerate(rows)
    ])

    prompt = BATCH_PROMPT_TEMPLATE.format(
        role_desc=role_desc,
        count=count,
        news_list=news_list,
        sentiment_addon=sentiment_addon,
    )

    resp = await client.chat([
        {"role": "system", "content": "只返回 JSON 数组，不加任何说明。"},
        {"role": "user",   "content": prompt},
    ], json_mode=False)

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
                        0.9,  # confidence 固定值，不再从 LLM output 解析
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

分析新闻，返回JSON（不加markdown）：
标题：{title}
内容：{content[:150]}

格式：{{"summary":"不超过60字","sentiment":"positive/negative/neutral","industries":["行业"],"event_type":"政策利好/技术突破/业绩超预期/利空消息/行业动态/其他","keywords":["词1","词2","词3"]}}{sentiment_addon}"""

    resp = await client.chat([
        {"role": "system", "content": "只返回 JSON，不加任何说明。"},
        {"role": "user",   "content": prompt},
    ], json_mode=False)

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
             float(data.get("confidence",0.0)), news_id),
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


