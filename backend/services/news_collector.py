"""新闻采集模块 — NewsAPI.ai / RSS / 大模型搜索"""
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Optional
import feedparser
import httpx
from loguru import logger
from backend.database import get_db
from backend.services.config_service import get_config
from backend.services.llm_client import get_llm_client


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


async def _is_duplicate(url: Optional[str], title: str) -> bool:
    key = url or f"title:{title}"
    async with get_db() as db:
        async with db.execute("SELECT 1 FROM news WHERE url=? LIMIT 1", (key,)) as cur:
            return await cur.fetchone() is not None


async def _save_raw_news(items: list[dict]) -> int:
    """保存原始新闻，返回新增条数（URL + 标题双重去重）"""
    saved = 0
    async with get_db() as db:
        for item in items:
            title = (item.get("title") or "").strip()
            url = item.get("url") or ""
            if not url:
                url = f"hash:{_url_hash((title + item.get('source','')))}"
            try:
                # URL 去重
                async with db.execute("SELECT 1 FROM news WHERE url=? LIMIT 1", (url,)) as cur:
                    if await cur.fetchone():
                        continue
                # 标题去重（同标题不同URL也跳过，防止转载重复）
                if title:
                    async with db.execute("SELECT 1 FROM news WHERE title=? LIMIT 1", (title,)) as cur:
                        if await cur.fetchone():
                            continue
                await db.execute(
                    """INSERT INTO news
                       (url,title,source,published_at,content,raw_source,created_at)
                       VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
                    (url, title, item.get("source",""),
                     item.get("published_at",""), item.get("content",""),
                     item.get("raw_source","unknown")),
                )
                saved += 1
            except Exception as e:
                logger.warning(f"保存新闻失败: {e}")
        await db.commit()
    return saved

async def collect_newsapi() -> list[dict]:
    enabled = await get_config("newsapi_enabled", False)
    if not enabled:
        return []
    keys = await get_config("newsapi_keys", [])
    active_keys = [k for k in keys if k.get("enabled") and k.get("key")]
    if not active_keys:
        logger.info("NewsAPI: 无可用 Key")
        return []

    keywords = await get_config("newsapi_keywords", ["A股", "中国股市"])
    page_size = await get_config("newsapi_page_size", 50)
    # 使用多值 list + keywordOper=or + phrase 模式，避免 OR 字符串在 simple 模式下失效
    kw_list = keywords if keywords else ["A股"]

    results = []
    for key_item in active_keys:
        api_key = key_item["key"]
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "https://eventregistry.org/api/v1/article/getArticles",
                    json={
                        "apiKey": api_key,
                        "keyword": kw_list,
                        "keywordOper": "or",
                        "keywordSearchMode": "phrase",
                        "lang": "zho",
                        "articlesPage": 1,
                        "articlesCount": page_size,
                        "articlesSortBy": "date",
                        "resultType": "articles",
                        "dataType": ["news"],
                        "includeArticleSentiment": True,
                        "articleBodyLen": 2000,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    logger.error(f"NewsAPI 返回错误: {data['error']}")
                    break
                articles = data.get("articles", {}).get("results", [])
                for a in articles:
                    raw_title = a.get("title", "")
                    # NewsAPI 偶尔将正文塞入 title，超过 100 字截断
                    title = raw_title[:100] + "…" if len(raw_title) > 100 else raw_title
                    results.append({
                        "url": a.get("url", ""),
                        "title": title,
                        "source": a.get("source", {}).get("title", ""),
                        "published_at": a.get("dateTime", ""),
                        "content": a.get("body", "")[:2000],
                        "raw_source": "newsapi",
                    })
            logger.info(f"NewsAPI 采集 {len(articles)} 条")
            break  # 成功后不再轮换
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"NewsAPI Key 限速，切换下一个")
                continue
            logger.error(f"NewsAPI 请求失败: {e}")
            break
        except Exception as e:
            logger.error(f"NewsAPI 异常: {e}")
            break
    return results


# ─── 路二：RSS ────────────────────────────────────────────────────────────────

async def collect_rss() -> list[dict]:
    enabled = await get_config("rss_enabled", False)
    if not enabled:
        return []
    feeds = await get_config("rss_feeds", [])
    active_feeds = [f for f in feeds if f.get("enabled") and f.get("url")]
    if not active_feeds:
        return []

    results = []
    for feed_item in active_feeds:
        try:
            parsed = await asyncio.to_thread(feedparser.parse, feed_item["url"])
            for entry in parsed.entries[:50]:
                results.append({
                    "url": entry.get("link", ""),
                    "title": entry.get("title", ""),
                    "source": feed_item.get("name", feed_item["url"]),
                    "published_at": entry.get("published", ""),
                    "content": entry.get("summary", "")[:2000],
                    "raw_source": "rss",
                })
            logger.info(f"RSS [{feed_item.get('name','')}] 采集 {len(parsed.entries)} 条")
        except Exception as e:
            logger.error(f"RSS 解析失败 {feed_item['url']}: {e}")
    return results


# ─── 搜索专用模型客户端 ──────────────────────────────────────────────────────

async def _get_search_llm_client():
    """搜索专用模型：支持多模型轮询，fallback 到分析模型"""
    from backend.services.llm_client import LLMClient, _key_index

    # 优先读取 search_llm_models 列表
    models_cfg = await get_config("search_llm_models", [])
    active = [m for m in models_cfg if m.get("enabled") is not False
              and m.get("provider") and m.get("api_key") and m.get("model")]

    # fallback：兼容旧的单模型配置
    if not active:
        provider = await get_config("search_llm_provider")
        api_key  = await get_config("search_llm_api_key")
        model    = await get_config("search_llm_model")
        base_url = await get_config("search_llm_base_url")
        if provider and api_key and model:
            active = [{"provider": provider, "api_key": api_key,
                       "model": model, "base_url": base_url or ""}]

    if not active:
        # 最终 fallback：使用分析模型
        return await get_llm_client()

    # 轮询索引
    idx_key = "search_llm_models"
    current_idx = _key_index.get(idx_key, 0) % len(active)
    _key_index[idx_key] = (current_idx + 1) % len(active)
    m = active[current_idx]

    return LLMClient(
        provider=m["provider"],
        api_key=m["api_key"],
        model=m["model"],
        base_url=m.get("base_url") or LLMClient._default_base_url(m["provider"]),
    )


# ─── 路三：大模型搜索 ──────────────────────────────────────────────────────────

async def collect_llm_search() -> list[dict]:
    enabled = await get_config("llm_search_enabled", False)
    if not enabled:
        return []

    client = await _get_search_llm_client()
    if not client:
        logger.warning("大模型搜索：未配置搜索模型")
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results = []

    # ── 三个预设模式 ──────────────────────────────────────────────────────────
    MODES = [
        {
            "key":     "general",
            "name":    "时事热点",
            "enabled": await get_config("llm_general_enabled", True),
            "count":   int(await get_config("llm_general_count", 10) or 10),
            "prompt":  f"""今天是 {today}。
请搜索并整理当前最新的时事热点新闻（最近 24 小时内）。
要求：
- 内容范围：国内外重要新闻、科技、财经、社会热点均可
- 优先选取有实质内容的新闻，排除标题党和重复内容
- 返回最多 {{count}} 条，按重要性排序

返回严格 JSON 数组（不加 markdown 代码块）：
[{{"title":"新闻标题","source":"来源媒体","published_at":"{today}","content":"一句话摘要不超过150字","url":"原文链接或空字符串"}}]""",
        },
        {
            "key":     "tech",
            "name":    "科技/AI",
            "enabled": await get_config("llm_tech_enabled", True),
            "count":   int(await get_config("llm_tech_count", 10) or 10),
            "prompt":  f"""今天是 {today}。
请搜索过去 24 小时内科技与 AI 领域的重要动态。
要求：
- 聚焦：AI 模型发布、大厂动向（OpenAI/Google/Meta/百度/阿里等）、融资并购、重要产品发布
- 剔除营销软文、低质量资讯、重复内容
- 返回最多 {{count}} 条，按影响力排序

返回严格 JSON 数组（不加 markdown 代码块）：
[{{"title":"新闻标题","source":"来源媒体","published_at":"{today}","content":"一句话摘要不超过150字","url":"原文链接或空字符串"}}]""",
        },
        {
            "key":     "finance",
            "name":    "财经市场",
            "enabled": await get_config("llm_finance_enabled", True),
            "count":   int(await get_config("llm_finance_count", 10) or 10),
            "prompt":  f"""今天是 {today}。
请整理今日财经市场的重要资讯。
要求：
- 涉及市场：A股、美股、加密货币、大宗商品、宏观政策
- 每条新闻摘要格式：发生了什么 + 可能影响（合计不超过 50 字）
- 剔除无实质内容的行情播报，聚焦有影响力的事件
- 返回最多 {{count}} 条，按市场影响力排序

返回严格 JSON 数组（不加 markdown 代码块）：
[{{"title":"新闻标题","source":"来源媒体","published_at":"{today}","content":"发生了什么+可能影响（50字内）","url":"原文链接或空字符串"}}]""",
        },
    ]

    for mode in MODES:
        if not mode["enabled"]:
            continue
        prompt = mode["prompt"].replace("{count}", str(mode["count"]))
        try:
            resp_text = await client.chat([
                {"role": "system", "content": "你是新闻搜索助手，只返回 JSON 数组，不加任何说明。"},
                {"role": "user", "content": prompt},
            ], json_mode=False, timeout=20)
            text = resp_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            items = json.loads(text)
            for item in items:
                item["raw_source"] = "llm"
            results.extend(items[:mode["count"]])
            logger.info(f"大模型搜索[{mode['name']}] 返回 {len(items)} 条")
        except Exception as e:
            logger.error(f"大模型搜索[{mode['name']}]失败: {e}")
        await asyncio.sleep(1)  # 避免触发限速

    return results



# ─── 路四：百度热搜 + 微博热搜 ──────────────────────────────────────────────

async def collect_trending() -> list[dict]:
    """采集百度热搜和微博热搜"""
    enabled = await get_config("trending_enabled", False)
    if not enabled:
        return []

    results = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── 百度热搜 ──
    baidu_enabled = await get_config("baidu_trending_enabled", True)
    if baidu_enabled:
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                resp = await client.get(
                    "https://top.baidu.com/board?tab=realtime",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                import re as _re
                html = resp.text
                words  = _re.findall(r'"word":"([^"]+)"', html)
                scores = _re.findall(r'"hotScore":"([^"]+)"', html)
                descs  = _re.findall(r'"desc":"([^"]*)"', html)
                count = 0
                for i, word in enumerate(words[:20]):
                    score = scores[i] if i < len(scores) else "0"
                    desc  = descs[i]  if i < len(descs)  else ""
                    results.append({
                        "url": f"https://www.baidu.com/s?wd={word}",
                        "title": word,
                        "source": "百度热搜",
                        "published_at": today,
                        "content": desc[:200] if desc else f"百度热搜第{i+1}位，热度{score}",
                        "raw_source": "trending",
                    })
                    count += 1
                logger.info(f"百度热搜采集 {count} 条")
        except Exception as e:
            logger.error(f"百度热搜采集失败: {e}")

    # ── 微博热搜 ──
    weibo_enabled = await get_config("weibo_trending_enabled", True)
    if weibo_enabled:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://weibo.com/ajax/side/hotSearch",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Referer": "https://weibo.com/",
                        "Accept": "application/json",
                    },
                )
                d = resp.json()
                realtime = d.get("data", {}).get("realtime", [])
                count = 0
                for i, item in enumerate(realtime[:20]):
                    word  = item.get("word", "").strip("#")
                    num   = item.get("num", 0)
                    label = item.get("label_name", "")
                    if not word:
                        continue
                    results.append({
                        "url": f"https://s.weibo.com/weibo?q=%23{word}%23",
                        "title": word,
                        "source": "微博热搜",
                        "published_at": today,
                        "content": f"微博热搜第{i+1}位，热度{num}" + (f"，{label}" if label else ""),
                        "raw_source": "trending",
                    })
                    count += 1
                logger.info(f"微博热搜采集 {count} 条")
        except Exception as e:
            logger.error(f"微博热搜采集失败: {e}")

    # ── 抖音热搜 ──
    douyin_enabled = await get_config("douyin_trending_enabled", True)
    if douyin_enabled:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://www.iesdouyin.com/web/api/v2/hotsearch/billboard/word/",
                    headers={
                        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
                        "Referer": "https://www.douyin.com/",
                    },
                )
                d = resp.json()
                word_list = d.get("word_list", [])
                label_map = {1: "热", 2: "爆", 3: "新"}
                count = 0
                for i, w in enumerate(word_list[:20]):
                    word  = w.get("word", "").strip()
                    hot   = w.get("hot_value", 0)
                    label = label_map.get(w.get("label", 0), "")
                    if not word:
                        continue
                    results.append({
                        "url": f"https://www.douyin.com/search/{word}",
                        "title": word,
                        "source": "抖音热搜",
                        "published_at": today,
                        "content": f"抖音热搜第{i+1}位，热度{hot:,}" + (f"，{label}" if label else ""),
                        "raw_source": "trending",
                    })
                    count += 1
                logger.info(f"抖音热搜采集 {count} 条")
        except Exception as e:
            logger.error(f"抖音热搜采集失败: {e}")

    return results

# ─── 主入口 ───────────────────────────────────────────────────────────────────

# 全局采集进度
_collect_progress = {
    "running": False, "stage": "idle", "message": "空闲", "done": True,
    "collected": 0, "saved": 0, "error": None
}

def get_collect_progress() -> dict:
    return dict(_collect_progress)

def _set_collect_progress(stage: str, message: str, running: bool = True,
                           done: bool = False, collected: int = 0, saved: int = 0, error=None):
    _collect_progress.update({
        "running": running, "stage": stage, "message": message,
        "done": done, "collected": collected, "saved": saved, "error": error,
    })


async def run_collection(sources: Optional[list[str]] = None) -> dict:
    """执行采集，返回统计"""
    sources = sources or ["newsapi", "rss", "llm", "trending"]
    _set_collect_progress("collecting", "正在采集新闻...", running=True, done=False)

    tasks = []
    if "newsapi" in sources:
        tasks.append(collect_newsapi())
    if "rss" in sources:
        tasks.append(collect_rss())
    if "llm" in sources:
        tasks.append(collect_llm_search())
    if "trending" in sources:
        tasks.append(collect_trending())

    # 总超时60秒，防止某个源（尤其是LLM搜索）永久挂起
    try:
        all_results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=60
        )
    except asyncio.TimeoutError:
        logger.warning("采集总超时(60s)，已获取部分结果")
        all_results = []
    combined = []
    for r in all_results:
        if isinstance(r, list):
            combined.extend(r)
        elif isinstance(r, Exception):
            logger.error(f"采集任务异常: {r}")

    _set_collect_progress("saving", f"采集到 {len(combined)} 条，正在入库...",
                           running=True, done=False, collected=len(combined))
    saved = await _save_raw_news(combined)
    logger.info(f"本次采集: 原始 {len(combined)} 条，新增 {saved} 条")
    done_msg = f"✅ 采集完成，新增 {saved} 条" if saved > 0 else "✅ 最新新闻已采集完毕"
    _set_collect_progress("done", done_msg,
                           running=False, done=True, collected=len(combined), saved=saved)

    # 采集完成后是否自动分类，由「新闻采集」页「采集后自动分类」开关控制
    if saved > 0:
        auto_classify = await get_config("auto_classify_enabled", True)
        if auto_classify is not False:
            asyncio.create_task(_trigger_classify())
        else:
            logger.info("自动分类已关闭，跳过分类（可手动点击「立即分类」）")

    return {"collected": len(combined), "saved": saved}


async def _trigger_classify():
    """采集后触发新闻分类（提取关键词/情感），不做龙头股匹配"""
    try:
        from backend.services.news_processor import process_pending_news
        await process_pending_news()
    except Exception as e:
        logger.error(f"触发分类失败: {e}")
