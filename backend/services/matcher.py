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


async def _write_match_progress(done: int, total: int, current_title: str = "",
                                finished: bool = False, error: str | None = None):
    """将匹配进度写入 config 表，供前端轮询"""
    import json as _json
    try:
        from backend.services.config_service import set_config
        msg = ""
        if finished:
            msg = f"匹配异常: {error}" if error else (
                f"✅ 匹配完成，共处理 {done} 条" if done > 0 else "✅ 最新新闻均已匹配完毕"
            )
        await set_config("match_progress", _json.dumps({
            "done": done, "total": total,
            "current": current_title[:40] if current_title else "",
            "finished": finished,
            "error": error,
            "message": msg,
        }))
    except Exception as e:
        logger.warning(f"写入匹配进度失败: {e}")


async def match_pending_news(batch_size: int = 10) -> int:
    """匹配未处理的新闻，返回处理条数"""
    async with get_db() as db:
        async with db.execute(
            """SELECT n.id, n.title, n.summary, n.industries, n.keywords, n.sentiment,
                      n.beneficiary_chain, n.news_level, n.time_horizon
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
    client = await get_llm_client()
    if not client:
        logger.warning("立即匹配：未配置分析匹配模型，LLM精排跳过，仅使用TF-IDF/语义匹配")
    from backend.services.llm_client import get_embed_client
    embed_client = await get_embed_client()
    matched = 0

    await _write_match_progress(0, total, "", False)

    for i, row in enumerate(rows):
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
                beneficiary_chain=json.loads(row["beneficiary_chain"] or "[]"),
                news_level=row["news_level"] or "medium",
                time_horizon=row["time_horizon"] or "short",
            )
            if result is not None:
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

    await _write_match_progress(total, total, "", True)
    logger.info(f"匹配完成: {matched}/{total}")
    return matched


async def _match_one_news(
    news_id: int, title: str, summary: str,
    industries: list[str], keywords: list[str],
    sentiment: str, top_k: int, client, embed_client=None,
    beneficiary_chain: list = None,
    news_level: str = "medium",
    time_horizon: str = "short",
) -> Optional[list[dict]]:

    # ── 阶段一：标签粗筛 ──────────────────────────────────────────────────────
    # 合并 keywords + beneficiary_chain 扩大候选范围
    all_search_terms = list(keywords)
    if beneficiary_chain:
        all_search_terms = list(dict.fromkeys(all_search_terms + list(beneficiary_chain)))
    candidates = await _industry_filter(industries, all_search_terms)
    if not candidates:
        logger.debug(f"新闻 {news_id}: 无候选股票")
        return []

    # ── 阶段二：向量语义匹配 ──────────────────────────────────────────────────
    query_text = f"{title} {summary} {' '.join(keywords)}"
    doc_texts = []
    for c in candidates:
        desc     = c.get('business_desc') or ''
        name     = c.get('name', '')
        industry = c.get('industry') or ''

        def _jl(field, limit=8):
            try:
                lst = json.loads(c.get(field) or '[]')
                return ' '.join(lst[:limit]) if isinstance(lst, list) else ''
            except Exception:
                return ''

        domains    = _jl('domains', 4)
        all_tags   = _jl('all_tags', 8)
        products   = _jl('products', 6)
        techs      = _jl('techs', 4)
        themes     = _jl('themes', 4)
        chain_pos  = _jl('chain_pos', 2)

        text = f"{name} {name} {industry} {desc} {products} {techs} {themes} {chain_pos} {domains} {all_tags}".strip()
        doc_texts.append(text if text else name)

    semantic_scores = [0.0] * len(candidates)
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

    # 综合评分：tag_score 权重 40%，semantic_score 权重 60%
    for i, c in enumerate(candidates):
        c["semantic_score"] = semantic_scores[i]
        tag_s = c.get("tag_score", 0.0)
        c["combined_score"] = tag_s * 0.4 + semantic_scores[i] * 0.6
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    top_candidates = candidates[:top_k * 6]  # 扩大精排候选池，避免好股票被挤出

    # ── 阶段三：大模型精排 ────────────────────────────────────────────────────
    if client and top_candidates:
        llm_results = await _llm_rerank(
            client, title, summary, keywords, top_candidates, top_k, sentiment,
            beneficiary_chain=beneficiary_chain or [],
            news_level=news_level,
            time_horizon=time_horizon,
        )
        if llm_results:
            return llm_results

    # fallback: 保留 tag_score>=2.0 或语义分足够的股票
    MIN_SEMANTIC_SCORE = 0.05
    qualified = [c for c in top_candidates if c.get("tag_score", 0) >= 2.0
                 or c.get("semantic_score", 0) >= MIN_SEMANTIC_SCORE]
    if not qualified:
        logger.debug(f"新闻 {news_id}: 候选语义分均低于 {MIN_SEMANTIC_SCORE}，跳过匹配")
        return []
    qualified.sort(key=lambda x: (x.get("tag_score", 0), x.get("semantic_score", 0)), reverse=True)

    results = []
    for c in qualified[:top_k]:
        name = c["name"]
        desc = c.get("business_desc") or ""
        industry = c.get("industry") or ""
        matched_kws = [kw for kw in keywords if kw in desc or kw in name]
        if matched_kws and desc:
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


# 通用词黑名单：出现在几乎所有公司描述里，用于搜索会产生大量噪音
_GENERIC_TERMS = {
    "上市公司", "市场", "市场活力", "高质量发展", "创新", "发展", "体系",
    "全链条", "支持", "改革", "政策", "监管", "规范", "制度", "管理",
    "服务", "业务", "企业", "公司", "经营", "运营", "投资", "金融",
    "经济", "产业", "行业", "市场化", "国际化", "数字化", "智能化",
    "资本", "资产", "收益", "利润", "增长", "规模", "战略", "布局",
    "农业", "食品", "食品行业", "工业", "制造业", "消费", "零售",
    "医疗", "医药", "教育", "地产", "房地产", "能源", "交通", "物流",
    "科技", "互联网", "电商", "传媒", "文化", "旅游", "餐饮",
}

async def _industry_filter(industries: list[str], keywords: list[str]) -> list[dict]:
    """候选集筛选（优先级从高到低）：
    优先级0 — 股票标签精确匹配（最精准，直接命中产品/技术标签）
    优先级1 — 股票名称精确包含关键词
    优先级2 — stock_profile 关键词/描述匹配
    兜底   — 仅在有标签命中的前提下补足候选，不再随机拉取无关股票
    """
    raw_terms = list(set(industries + keywords))
    search_terms = [t for t in raw_terms if t not in _GENERIC_TERMS and len(t) >= 2]

    results = []
    tag_hit_codes = set()

    # ── 优先级0：股票标签精确匹配（stock_tags + stock_board_tags 双路）──────
    if search_terms:
        tag_conds   = " OR ".join(["st.all_tags LIKE ?" for _ in search_terms])
        board_conds = " OR ".join(["sbt.all_board_tags LIKE ?" for _ in search_terms])
        tag_params  = [f'%{t}%' for t in search_terms]

        # 0a. 主营标签匹配
        async with get_db() as db:
            async with db.execute(
                f"""SELECT s.ts_code, s.name, COALESCE(s.industry,'') as industry,
                           COALESCE(s.market,'') as market,
                           COALESCE(sp.business_desc,'') as business_desc,
                           COALESCE(sp.domains,'[]') as domains,
                           COALESCE(sp.keywords,'[]') as kw_text,
                           COALESCE(st.all_tags,'[]') as all_tags,
                           COALESCE(st.products,'[]') as products,
                           COALESCE(st.sectors,'[]') as sectors,
                           COALESCE(st.techs,'[]') as techs,
                           COALESCE(st.themes,'[]') as themes,
                           COALESCE(st.chain_pos,'[]') as chain_pos
                    FROM stocks s
                    LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                    LEFT JOIN stock_tags st ON s.ts_code=st.ts_code
                    WHERE {tag_conds} LIMIT 200""",
                tag_params
            ) as cur:
                rows = await cur.fetchall()
        for r in rows:
            d = dict(r)
            d["tag_score"] = 2.0
            results.append(d)
            tag_hit_codes.add(r["ts_code"])

        # 0b. 板块标签匹配
        async with get_db() as db:
            async with db.execute(
                f"""SELECT s.ts_code, s.name, COALESCE(s.industry,'') as industry,
                           COALESCE(s.market,'') as market,
                           COALESCE(sp.business_desc,'') as business_desc,
                           COALESCE(sp.domains,'[]') as domains,
                           COALESCE(sp.keywords,'[]') as kw_text
                    FROM stocks s
                    LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                    JOIN stock_board_tags sbt ON s.ts_code=sbt.ts_code
                    WHERE {board_conds} LIMIT 200""",
                tag_params
            ) as cur:
                board_rows = await cur.fetchall()

        existing = {r["ts_code"] for r in results}
        for r in board_rows:
            if r["ts_code"] not in existing:
                d = dict(r)
                d["tag_score"] = 1.8
                d["all_tags"] = "[]"
                results.append(d)
                tag_hit_codes.add(r["ts_code"])
            else:
                for res in results:
                    if res["ts_code"] == r["ts_code"]:
                        res["tag_score"] = min(res["tag_score"] + 0.5, 3.0)
                        break

        if results:
            logger.debug(f"标签匹配命中 {len(results)} 只（主营+板块）: {search_terms}")

    # 扩展同义词
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
    search_terms = list(dict.fromkeys(expanded))

    # ── 阶段1：股票名称匹配 ──
    existing_codes = {r['ts_code'] for r in results}
    if search_terms:
        name_conds  = " OR ".join(["s.name LIKE ?" for _ in search_terms])
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
        for r in rows:
            if r["ts_code"] not in existing_codes:
                d = dict(r)
                d["tag_score"] = 0.5
                results.append(d)
                existing_codes.add(r["ts_code"])

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
        for r in rows:
            if r['ts_code'] not in existing_codes:
                d = dict(r)
                d["tag_score"] = 0.3
                results.append(d)
                existing_codes.add(r['ts_code'])

    # ── 修复点③：移除随机兜底 ─────────────────────────────────────────────────
    # 原代码：候选不足 20 只时，塞入 150 只随机股票（tag_score=0）
    # 这是"匹配到不相关股票"的直接元凶：随机股票进入 Embedding 排序后，
    # 语义分略高就能混进精排候选，LLM 看到了错误的候选池，输出了不相关结果。
    #
    # 修改：不再随机补充。候选不足时记录日志，宁可返回少量高质量结果，
    # 也不用随机股票稀释候选池的精准度。
    if len(results) == 0:
        logger.debug(f"标签/描述匹配无结果，跳过该新闻匹配，search_terms={search_terms}")

    # 按 tag_score 排序
    results.sort(key=lambda x: x.get("tag_score", 0), reverse=True)
    return results[:300]


async def _llm_rerank(
    client, title: str, summary: str, keywords: list[str],
    candidates: list[dict], top_k: int, sentiment: str,
    beneficiary_chain: list = None,
    news_level: str = "medium",
    time_horizon: str = "short",
) -> list[dict]:
    """大模型精排"""
    # 修复点④：精排时向 LLM 传入 keywords，给足信息量
    # 原来只传了 title 和 60 字 summary，LLM 缺乏精准判断的上下文
    stock_list = "\n".join([
        f"{i+1}. {c['ts_code']} {c['name']}（{c.get('industry','')}）- {c.get('business_desc','')[:120]}"
        for i, c in enumerate(candidates)
    ])
    from backend.services.config_service import get_config as _gc
    company_type = await _gc("match_company_type", "leader")

    type_instructions = {
        "leader": f"请优先选择同行业中**市值最大、知名度最高的龙头企业**（如行业第一、第二名），共 {top_k} 只。",
        "direct": f"请选择与新闻内容**直接相关**、业务最契合的企业，不限龙头，精准匹配，共 {top_k} 只。",
        "chain":  f"请同时考虑直接相关企业及其**上下游产业链**企业，覆盖产业链传导，共 {top_k} 只。",
        "broad":  f"请覆盖整个相关行业，选出受影响最广泛的企业，共 {top_k} 只。",
    }
    type_hint = type_instructions.get(company_type, type_instructions["leader"])

    # 精排 prompt：三梯队受益逻辑（年化30%高手方法论）
    beneficiary_str = "、".join(beneficiary_chain) if beneficiary_chain else "（未提取）"
    time_map = {"short": "短线1-4周", "medium": "中线1-6月", "long": "长线1-3年"}
    time_str = time_map.get(time_horizon, "短线")
    level_map = {"高": "高价值新闻（政策/大额投资/重大事件）", "中": "中价值新闻（工商变更/中小投资）", "低": "低价值新闻"}
    level_str = level_map.get(news_level, "中价值新闻")

    prompt = f"""你是一位年化30%的A股高手，用产业链受益方思维筛选股票。

【新闻标题】{title}
【新闻摘要】{summary}
【新闻级别】{level_str}
【产业链受益词】{beneficiary_str}
【行业关键词】{', '.join(keywords)}
【情感倾向】{sentiment}
【时间维度】{time_str}

【候选股票】
{stock_list}

{type_hint}

**三梯队受益分析框架：**

第一梯队（直接受益，逻辑最强）：
- 新闻主体花的钱直接流向的供应商/服务商
- 项目所需的设备商、IT系统商、基础设施提供商
- 产业链受益词能直接命中主营业务的股票

第二梯队（产业链配套，逻辑次之）：
- 上游原材料供应商
- 关键零部件/技术提供商
- 下游核心客户

第三梯队（概念受益，逻辑最弱）：
- 政策/地域概念相关
- 间接受益、需要多步推理

**严格排除（最高优先级）：**
- 与新闻主体直接竞争的同行公司，无论理由多充分，一律排除
- 判断方法：新闻主体做什么业务，做同样业务的公司就是竞争方
- 蜂鸟即配（即时配送平台）→ 顺丰同城、达达、美团配送等同类配送平台全部排除
- 即使是行业龙头，只要是竞争方就排除，不要用"技术溢出""估值重塑"等理由强行入选
- 传统快递公司（顺丰、韵达、圆通）与即时配送平台是不同赛道，也应排除或降至第三梯队

请按受益梯队强度排序，返回严格JSON数组（不加markdown）：
[
  {{
    "ts_code": "股票代码",
    "name": "股票名称",
    "score": 0.95,
    "benefit_tier": "第一梯队/第二梯队/第三梯队",
    "reason": "受益逻辑（说明属于哪个梯队、为何受益，不超过50字）",
    "sentiment_impact": "positive/negative/neutral"
  }}
]
若候选中无真正受益股（均为竞争方或无关），返回[]，不要强行匹配。"""

    try:
        resp = await client.chat([
            {"role": "system", "content": "你是A股投研助手，只返回JSON数组。"},
            {"role": "user", "content": prompt},
        ], json_mode=False, timeout=180)
        text = resp.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        raw = json.loads(text)

        code_to_candidate = {c["ts_code"]: c for c in candidates}
        code_to_semantic  = {c["ts_code"]: c.get("semantic_score", 0.0) for c in candidates}
        results = []
        for item in raw:
            if isinstance(item, str):
                ts_code = item.strip()
                # 修复点⑤：LLM 返回字符串格式时，必须验证该 ts_code 确实在候选池里
                # 原来没有这个校验，LLM 可能幻觉出不在候选池里的股票代码
                if ts_code not in code_to_candidate:
                    logger.debug(f"LLM 返回了不在候选池中的代码 {ts_code}，已过滤")
                    continue
                cand = code_to_candidate[ts_code]
                results.append({
                    "ts_code": ts_code,
                    "name": cand.get("name", ts_code),
                    "score": 0.8,
                    "reason": f"{cand.get('name', ts_code)}（{cand.get('industry','')}）：{(cand.get('business_desc') or '')[:40]}，与新闻相关",
                    "sentiment_impact": sentiment,
                    "semantic_score": code_to_semantic.get(ts_code, 0.0),
                    "industry_score": 0.6,
                })
            elif isinstance(item, dict):
                ts_code = item.get("ts_code", "")
                if ts_code not in code_to_candidate:
                    logger.debug(f"LLM 返回了不在候选池中的代码 {ts_code}，已过滤")
                    continue
                item["semantic_score"] = code_to_semantic.get(ts_code, 0.0)
                item["industry_score"] = 0.6
                # 第一梯队得分加成，让受益逻辑强的股票排更前
                tier = item.get("benefit_tier", "第三梯队")
                tier_bonus = 0.15 if "第一" in tier else 0.08 if "第二" in tier else 0.0
                item["score"] = round(
                    item["semantic_score"] * 0.4 + float(item.get("score", 0.5)) * 0.5 + tier_bonus, 4
                )
                results.append(item)
        return results[:top_k]
    except Exception as e:
        logger.warning(f"大模型精排失败（所有模型均已尝试）: {e}，降级为候选结果")
        return []
