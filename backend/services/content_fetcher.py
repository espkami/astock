"""
新闻正文回源补全模块
基于 news_scraper.py 的四级降级链：trafilatura → newspaper3k → bs4 → playwright
集成到采集流程，在存库前补全 content 不足的新闻正文
"""
import asyncio
import logging
from typing import Optional

log = logging.getLogger(__name__)

# content 低于此长度时触发回源补全
CONTENT_MIN_LENGTH = 200


def _fetch_trafilatura(url: str) -> Optional[str]:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if result and len(result.strip()) > 100:
            return result.strip()
    except Exception as e:
        log.debug(f"trafilatura 失败 {url}: {e}")
    return None


def _fetch_newspaper(url: str) -> Optional[str]:
    try:
        from newspaper import Article as NpArticle
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        cn_keywords = ["sina", "163", "sohu", "qq", "baidu", "xinhua",
                       "people", "ifeng", "caixin", "36kr", "eastmoney",
                       "cls", "yicai", "jrj", ".cn", ".com.cn"]
        lang = "zh" if any(kw in domain for kw in cn_keywords) else "en"
        art = NpArticle(url, language=lang, request_timeout=15)
        art.download()
        art.parse()
        if art.text and len(art.text.strip()) > 100:
            return art.text.strip()
    except Exception as e:
        log.debug(f"newspaper3k 失败 {url}: {e}")
    return None


def _fetch_bs4(url: str) -> Optional[str]:
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "iframe"]):
            tag.decompose()
        body = (
            soup.find("article")
            or soup.find(attrs={"class": lambda c: c and any(
                kw in " ".join(c).lower()
                for kw in ["article", "content", "post", "body", "entry", "story"]
            )})
            or soup.find("main")
            or soup.body
        )
        paragraphs = body.find_all("p") if body else soup.find_all("p")
        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 20
        )
        if text and len(text) > 100:
            return text
    except Exception as e:
        log.debug(f"bs4 失败 {url}: {e}")
    return None


def _fetch_full_content(url: str) -> Optional[str]:
    """同步降级链：trafilatura → newspaper3k → bs4"""
    for fn in [_fetch_trafilatura, _fetch_newspaper, _fetch_bs4]:
        result = fn(url)
        if result:
            # 剥离 HTML 标签，只保留纯文字
            try:
                from bs4 import BeautifulSoup
                result = BeautifulSoup(result, "html.parser").get_text(separator="\n", strip=True)
            except Exception:
                pass
            log.info(f"回源补全成功 [{fn.__name__}] {url[:60]} ({len(result)}字)")
            return result
    log.warning(f"回源补全全部失败: {url[:60]}")
    return None


async def enrich_content(items: list[dict]) -> list[dict]:
    """
    对 content 不足的新闻异步回源补全正文。
    items: 采集到的原始新闻列表（含 url / content 字段）
    返回补全后的列表
    """
    needs_enrich = [
        i for i, item in enumerate(items)
        if item.get("url")
        and not item["url"].startswith("hash:")
        # 热搜类没有真实文章URL，跳过
        and item.get("raw_source") not in ("trending",)
        # LLM搜索本身就是摘要，跳过
        and item.get("raw_source") not in ("llm",)
    ]

    log.info(f"enrich_content 收到 {len(items)} 条，needs_enrich: {len(needs_enrich)} 条")
    log.info(f"raw_sources: {set(i.get('raw_source') for i in items)}")
    if not needs_enrich:
        log.info("needs_enrich 为空，跳过补全")
        return items

    log.info(f"需要回源补全: {len(needs_enrich)} 条（共 {len(items)} 条）")

    def _do_enrich(idx: int):
        item = items[idx]
        full = _fetch_full_content(item["url"])
        if full:
            items[idx]["content"] = full
            items[idx]["content_enriched"] = True

    # 并发回源，最多 5 个同时进行，避免被封
    sem = asyncio.Semaphore(5)

    async def _enrich_one(idx: int):
        async with sem:
            await asyncio.to_thread(_do_enrich, idx)

    await asyncio.gather(*[_enrich_one(i) for i in needs_enrich])

    enriched = sum(1 for i in needs_enrich if items[i].get("content_enriched"))
    log.info(f"回源补全完成: {enriched}/{len(needs_enrich)} 条成功")
    return items
