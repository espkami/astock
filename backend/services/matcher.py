"""新闻-股票匹配引擎 — 四阶段流水线"""
import json
import asyncio
import numpy as np
from typing import Optional
from loguru import logger
from backend.database import get_db
from backend.services.config_service import get_config
from backend.services.llm_client import get_llm_client


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _tfidf_similarity(query: str, docs: list[str]) -> list[float]:
    """TF-IDF 余弦相似度 fallback"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        all_texts = [query] + docs
        matrix = vectorizer.fit_transform(all_texts)
        query_vec = matrix[0]
        doc_vecs = matrix[1:]
        scores = []
        for i in range(doc_vecs.shape[0]):
            dv = doc_vecs[i]
            dot = (query_vec * dv.T).toarray()[0][0]
            n1 = np.sqrt((query_vec * query_vec.T).toarray()[0][0])
            n2 = np.sqrt((dv * dv.T).toarray()[0][0])
            scores.append(float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0)
        return scores
    except Exception:
        return [0.0] * len(docs)


async def _write_match_progress(done: int, total: int, current_title: str = "", finished: bool = False):
    """将匹配进度写入 config 表，供前端轮询"""
    import json as _json
    try:
        from backend.services.config_service import set_config
        await set_config("match_progress", _json.dumps({
            "done": done, "total": total,
            "current": current_title[:40] if current_title else "",
            "finished": finished,
        }))
    except Exception:
        pass


async def match_pending_news(batch_size: int = 10) -> int:
    """匹配未处理的新闻，返回处理条数"""
    async with get_db() as db:
        async with db.execute(
            """SELECT n.id, n.title, n.summary, n.industries, n.keywords, n.sentiment
               FROM news n
               LEFT JOIN match_results mr ON n.id = mr.news_id
               WHERE n.summary IS NOT NULL
                 AND n.summary != '[内容审核限制]'
                 AND mr.id IS NULL
               ORDER BY n.created_at DESC LIMIT ?""",
            (batch_size,),
        ) as cur:
            rows = await cur.fetchall()

    if not rows:
        return 0

    total = len(rows)
    top_k = await get_config("match_top_k", 5)
    # 分析匹配模型：只做 LLM 精排
    client = await get_llm_client()
    # Embedding 专用模型：独立配置，支持 OpenAI/Qwen 免费额度
    from backend.services.llm_client import get_embed_client
    embed_client = await get_embed_client()
    matched = 0

    # 初始化进度
    await _write_match_progress(0, total, "", False)

    for i, row in enumerate(rows):
        # 更新当前进度
        await _write_match_progress(i, total, row["title"], False)
        try:
            result = await _match_one_news(
                news_id=row["id"],
                title=row["title"],
                summary=row["summary"] or "",
                industries=json.loads(row["industries"] or "[]"),
                keywords=json.loads(row["keywords"] or "[]"),
                sentiment=row["sentiment"],
                top_k=top_k,
                client=client,
                embed_client=embed_client,
            )
            if result:
                async with get_db() as db:
                    await db.execute("DELETE FROM match_results WHERE news_id=?", (row["id"],))
                    await db.execute(
                        "INSERT INTO match_results(news_id, matched_stocks) VALUES(?,?)",
                        (row["id"], json.dumps(result, ensure_ascii=False)),
                    )
                    await db.commit()
                matched += 1
        except Exception as e:
            logger.error(f"匹配新闻 {row['id']} 失败: {e}")

    # 完成
    await _write_match_progress(total, total, "", True)
    logger.info(f"匹配完成: {matched}/{total}")
    return matched


async def _match_one_news(
    news_id: int, title: str, summary: str,
    industries: list[str], keywords: list[str],
    sentiment: str, top_k: int, client, embed_client=None
) -> Optional[list[dict]]:

    # ── 阶段一：行业粗筛 ──────────────────────────────────────────────────────
    candidates = await _industry_filter(industries, keywords)
    if not candidates:
        logger.debug(f"新闻 {news_id}: 无候选股票")
        return []

    # ── 阶段二：向量语义匹配 ──────────────────────────────────────────────────
    query_text = f"{title} {summary} {' '.join(keywords)}"
    # doc 文本：名称 + 描述，描述为空时重复名称增加权重
    doc_texts = []
    for c in candidates:
        desc = c.get('business_desc') or ''
        name = c.get('name', '')
        domains = c.get('domains') or ''
        text = f"{name} {name} {desc} {domains}".strip()
        doc_texts.append(text if text else name)

    semantic_scores = [0.0] * len(candidates)
    # 优先用 Embedding 专用模型，fallback 到分析匹配模型（仅 openai/qwen 支持）
    _embed = embed_client or client
    if _embed:
        embeddings = await _embed.embed([query_text] + doc_texts)
        if len(embeddings) == len(candidates) + 1:
            query_emb = embeddings[0]
            for i, doc_emb in enumerate(embeddings[1:]):
                semantic_scores[i] = _cosine_similarity(query_emb, doc_emb)
        else:
            semantic_scores = _tfidf_similarity(query_text, doc_texts)
    else:
        logger.debug("无可用 Embedding 模型，使用 TF-IDF")
        semantic_scores = _tfidf_similarity(query_text, doc_texts)

    # 附加语义分数，取 top_k*2 进精排
    for i, c in enumerate(candidates):
        c["semantic_score"] = semantic_scores[i]
    candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
    top_candidates = candidates[:top_k * 2]

    # ── 阶段三：大模型精排 ────────────────────────────────────────────────────
    if client and top_candidates:
        llm_results = await _llm_rerank(client, title, summary, top_candidates, top_k, sentiment)
        if llm_results:
            return llm_results

    # fallback: 语义分数低于阈值时不强行匹配（宁可无结果也不要乱匹配）
    MIN_SEMANTIC_SCORE = 0.15
    qualified = [c for c in top_candidates if c.get("semantic_score", 0) >= MIN_SEMANTIC_SCORE]
    if not qualified:
        logger.debug(f"新闻 {news_id}: 候选语义分均低于 {MIN_SEMANTIC_SCORE}，跳过匹配")
        return []

    results = []
    for c in qualified[:top_k]:
        name = c["name"]
        desc = c.get("business_desc") or ""
        industry = c.get("industry") or ""
        # 找与新闻关键词重叠的业务描述片段
        matched_kws = [kw for kw in keywords if kw in desc or kw in name]
        if matched_kws and desc:
            # 截取包含关键词的描述片段
            snippet = desc[:40].rstrip("，。、")
            reason = f"{name}主营「{snippet}」，与新闻{'/'.join(matched_kws[:2])}方向直接相关"
        elif desc:
            reason = f"{name}（{industry}）：{desc[:35].rstrip('，。、')}，与新闻行业方向匹配"
        else:
            reason = f"{name}（{industry}）与新闻行业方向匹配"
        results.append({
            "ts_code": c["ts_code"],
            "name": name,
            "score": round(max(c["semantic_score"], 0.1), 4),
            "reason": reason,
            "sentiment_impact": sentiment,
            "semantic_score": round(c["semantic_score"], 4),
            "industry_score": 0.5,
        })
    return results


# 通用词黑名单：这些词出现在几乎所有公司描述里，用于搜索会产生大量噪音
_GENERIC_TERMS = {
    "上市公司", "市场", "市场活力", "高质量发展", "创新", "发展", "体系",
    "全链条", "支持", "改革", "政策", "监管", "规范", "制度", "管理",
    "服务", "业务", "企业", "公司", "经营", "运营", "投资", "金融",
    "经济", "产业", "行业", "市场化", "国际化", "数字化", "智能化",
    "资本", "资产", "收益", "利润", "增长", "规模", "战略", "布局",
}

async def _industry_filter(industries: list[str], keywords: list[str]) -> list[dict]:
    """候选集筛选：
    优先级1 — 股票名称精确包含关键词（最相关）
    优先级2 — stock_profile 关键词/描述匹配（有 profile 时）
    兜底   — 返回随机 150 只股票供语义排序
    """
    # 过滤掉通用词，只保留有区分度的词
    raw_terms = list(set(industries + keywords))
    search_terms = [t for t in raw_terms if t not in _GENERIC_TERMS and len(t) >= 2]

    results = []

    # 扩展同义词：确保关键行业词都能覆盖
    EXPAND_MAP = {
        "半导体": ["半导体", "芯片", "集成电路", "晶圆", "代工", "封测"],
        "AI算力": ["AI", "算力", "人工智能", "GPU", "训练", "推理"],
        "数据中心": ["数据中心", "IDC", "服务器", "存储"],
        "云计算": ["云计算", "云服务", "云平台"],
        "AI应用": ["人工智能", "大模型", "AI应用"],
        "通信": ["通信", "5G", "运营商", "电信"],
        "新能源": ["新能源", "光伏", "储能", "风电"],
        "汽车": ["汽车", "新能源汽车", "智能驾驶"],
    }
    expanded = list(search_terms)
    for term in list(search_terms):
        for key, synonyms in EXPAND_MAP.items():
            if any(s in term for s in synonyms):
                expanded.extend(synonyms)
    search_terms = list(dict.fromkeys(expanded))  # 去重保序

    # ── 阶段1：股票名称匹配（不依赖 profile）──
    if search_terms:
        name_conds = " OR ".join(["s.name LIKE ?" for _ in search_terms])
        params_name = [f"%{t}%" for t in search_terms]
        async with get_db() as db:
            async with db.execute(
                f"""SELECT s.ts_code, s.name, COALESCE(s.industry,'') as industry,
                           COALESCE(s.market,'') as market,
                           COALESCE(sp.business_desc,'') as business_desc,
                           COALESCE(sp.domains,'[]') as domains,
                           COALESCE(sp.keywords,'[]') as kw_text
                    FROM stocks s LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                    WHERE {name_conds} LIMIT 200""",
                params_name
            ) as cur:
                rows = await cur.fetchall()
        results.extend([dict(r) for r in rows])

    # ── 阶段2：profile 描述 + 行业匹配 ──
    if search_terms:
        profile_conds = " OR ".join([
            "sp.business_desc LIKE ? OR s.industry LIKE ?" for _ in search_terms
        ])
        params_prof = []
        for t in search_terms:
            params_prof.extend([f"%{t}%", f"%{t}%"])
        async with get_db() as db:
            async with db.execute(
                f"""SELECT s.ts_code, s.name, COALESCE(s.industry,'') as industry,
                           COALESCE(s.market,'') as market,
                           COALESCE(sp.business_desc,'') as business_desc,
                           COALESCE(sp.domains,'[]') as domains,
                           COALESCE(sp.keywords,'[]') as kw_text
                    FROM stocks s JOIN stock_profile sp ON s.ts_code=sp.ts_code
                    WHERE {profile_conds} LIMIT 200""",
                params_prof
            ) as cur:
                rows = await cur.fetchall()
        # 去重合并
        existing = {r['ts_code'] for r in results}
        results.extend([dict(r) for r in rows if r['ts_code'] not in existing])

    # ── 兜底：候选不足 20 只时，补充随机股票供语义排序 ──
    if len(results) < 20:
        async with get_db() as db:
            async with db.execute(
                """SELECT s.ts_code, s.name, COALESCE(s.industry,'') as industry,
                          COALESCE(s.market,'') as market,
                          COALESCE(sp.business_desc,'') as business_desc,
                          COALESCE(sp.domains,'[]') as domains,
                          COALESCE(sp.keywords,'[]') as kw_text
                   FROM stocks s LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                   ORDER BY RANDOM() LIMIT 150"""
            ) as cur:
                rows = await cur.fetchall()
        existing = {r['ts_code'] for r in results}
        results.extend([dict(r) for r in rows if r['ts_code'] not in existing])

    return results[:300]


async def _llm_rerank(client, title: str, summary: str, candidates: list[dict], top_k: int, sentiment: str) -> list[dict]:
    """大模型精排"""
    stock_list = "\n".join([
        f"{i+1}. {c['ts_code']} {c['name']}（{c.get('industry','')}）- {c.get('business_desc','')[:80]}"
        for i, c in enumerate(candidates)
    ])
    # 根据企业类型偏好调整 prompt
    from backend.services.config_service import get_config as _gc
    company_type = await _gc("match_company_type", "leader")

    type_instructions = {
        "leader": f"请优先选择同行业中**市值最大、知名度最高的龙头企业**（如行业第一、第二名），共 {top_k} 只。",
        "direct": f"请选择与新闻内容**直接相关**、业务最契合的企业，不限龙头，精准匹配，共 {top_k} 只。",
        "chain":  f"请同时考虑直接相关企业及其**上下游产业链**企业，覆盖产业链传导，共 {top_k} 只。",
        "broad":  f"请覆盖整个相关行业，选出受影响最广泛的企业，共 {top_k} 只。",
    }
    type_hint = type_instructions.get(company_type, type_instructions["leader"])

    prompt = f"""以下是一条 A 股新闻和候选股票列表。

【新闻标题】{title}
【新闻摘要】{summary}
【情感倾向】{sentiment}

【候选股票】
{stock_list}

{type_hint}

请返回严格 JSON 数组（不加 markdown 代码块），每条包含：
[
  {{
    "ts_code": "股票代码",
    "name": "股票名称",
    "score": 0.95,
    "reason": "一句话匹配理由（不超过50字，说明为何是龙头/直接相关/产业链）",
    "sentiment_impact": "positive 或 negative 或 neutral"
  }}
]
按相关性降序排列。
若候选股票中没有与新闻真正相关的，请返回空数组 []，不要强行匹配。"""

    try:
        resp = await client.chat([
            {"role": "system", "content": "你是 A 股投研助手，只返回 JSON 数组。"},
            {"role": "user", "content": prompt},
        ], json_mode=False)
        text = resp.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        results = json.loads(text)

        # 附加语义分数（从候选中查找）
        code_to_semantic = {c["ts_code"]: c.get("semantic_score", 0.0) for c in candidates}
        for r in results:
            r["semantic_score"] = code_to_semantic.get(r["ts_code"], 0.0)
            r["industry_score"] = 0.6
            # 综合评分
            r["score"] = round(
                r["semantic_score"] * 0.4 + float(r.get("score", 0.5)) * 0.5 + 0.5 * 0.1, 4
            )
        return results[:top_k]
    except Exception as e:
        logger.warning(f"大模型精排失败: {e}")
        return []
