"""FastAPI 应用入口"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ── 持久化日志（写文件，10MB 切割，保留 7 天）────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    encoding="utf-8",
)

from backend.database import init_db
from backend.services.scheduler import start_scheduler, stop_scheduler, is_running
from backend.services.config_service import get_config
from backend.services.scheduler import update_collect_interval
from backend.routers import news, stocks, config_router, data_router
from backend.models import APIResponse, DashboardStats


async def _resume_direct_match():
    """服务重启后自动恢复未完成的直接推理任务"""
    import asyncio as _asyncio
    await _asyncio.sleep(2)  # 等待服务完全就绪
    try:
        from backend.database import get_db as _get_db
        async with _get_db() as db:
            async with db.execute(
                "SELECT COUNT(*) as cnt FROM news n LEFT JOIN match_results mr ON n.id=mr.news_id WHERE mr.id IS NULL"
            ) as cur:
                pending = (await cur.fetchone())["cnt"]
        if pending > 0:
            from backend.services.config_service import set_config as _sc
            import json as _json
            await _sc("match_progress", _json.dumps({
                "done": 0, "total": pending, "current": "", "finished": False,
                "error": None, "message": f"恢复推理，共 {pending} 条待处理...",
            }, ensure_ascii=False))
            # 触发推理（复用 direct-match-all 逻辑）
            import httpx as _httpx
            async with _httpx.AsyncClient() as client:
                await client.post("http://localhost:7777/api/direct-match-all",
                                  json={}, timeout=10)
    except Exception as e:
        from loguru import logger as _log
        _log.warning(f"自动恢复推理失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    import asyncio as _asyncio
    logger.info("A股新闻匹配系统启动中...")
    await init_db()
    from backend.auth import init_users
    await init_users()
    from backend.services.tag_service import init_tags_table
    await init_tags_table()
    from backend.services.board_tag_service import init_board_tags_table
    await init_board_tags_table()
    # 保存 event loop 供 scheduler 线程使用
    from backend.services import scheduler as _sched_mod
    _sched_mod._MAIN_LOOP = _asyncio.get_running_loop()
    start_scheduler()
    # 恢复各路源独立采集间隔
    try:
        from backend.services.scheduler import update_source_intervals
        newsapi_iv  = await get_config("newsapi_interval", 7200)
        rss_iv      = await get_config("rss_interval", 1800)
        llm_iv      = await get_config("llm_search_interval", 3600)
        trending_iv = await get_config("trending_interval", 1800)
        await update_source_intervals(
            newsapi=int(newsapi_iv or 7200),
            rss=int(rss_iv or 1800),
            llm_search=int(llm_iv or 3600),
            trending=int(trending_iv or 1800),
        )
    except Exception as e:
        logger.warning(f"恢复采集间隔失败: {e}")
    # 恢复定时匹配设置
    match_times = await get_config("match_schedule_times", [])
    if match_times:
        from backend.services.scheduler import update_match_schedule
        update_match_schedule(match_times)
        logger.info(f"定时匹配已恢复: {match_times}")
    # 启动时检查是否有未完成的推理任务，自动恢复
    try:
        import json as _json
        from backend.services.config_service import get_config as _gc
        prog_str = await _gc("match_progress", "{}")
        if isinstance(prog_str, str):
            prog = _json.loads(prog_str)
        else:
            prog = prog_str or {}
        if not prog.get("finished", True) and prog.get("total", 0) > 0:
            logger.info(f"检测到未完成推理任务（{prog.get('done',0)}/{prog.get('total',0)}），自动恢复...")
            _asyncio.create_task(_resume_direct_match())
    except Exception as _e:
        logger.warning(f"恢复推理任务检查失败: {_e}")
    logger.info("系统就绪")
    yield
    # 关闭
    stop_scheduler()
    logger.info("系统已关闭")


# _background_worker 已移除：匹配仅由「立即匹配」或「定时匹配」触发


app = FastAPI(
    title="A股新闻-龙头股匹配系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# on_startup 后台 worker 已移除


# ── 认证路由 ──────────────────────────────────────────────────────────────────
from fastapi import Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from backend.auth import (
    get_user, verify_password, create_token, get_current_user,
    list_users, create_user, update_password, delete_user
)

class LoginReq(BaseModel):
    username: str
    password: str

class CreateUserReq(BaseModel):
    username: str
    password: str

class UpdatePwReq(BaseModel):
    username: str
    new_password: str

@app.post("/api/auth/login")
async def login(req: LoginReq):
    from fastapi import HTTPException as _HE
    user = await get_user(req.username)
    if not user or not verify_password(req.password, user["hashed_pw"]):
        raise _HE(status_code=401, detail="用户名或密码错误")
    token = create_token(req.username)
    return {"token": token, "username": req.username, "role": user["role"]}

@app.get("/api/auth/me")
async def me(current=Depends(get_current_user)):
    return {"username": current["username"], "role": current["role"]}

@app.get("/api/auth/users")
async def get_users(current=Depends(get_current_user)):
    return await list_users()

@app.post("/api/auth/users")
async def add_user(req: CreateUserReq, current=Depends(get_current_user)):
    try:
        result = await create_user(req.username, req.password)
        return APIResponse(message=f"用户 {req.username} 已创建", data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/auth/password")
async def change_password(req: UpdatePwReq, current=Depends(get_current_user)):
    await update_password(req.username, req.new_password)
    return APIResponse(message=f"用户 {req.username} 密码已更新")

@app.delete("/api/auth/users/{username}")
async def remove_user(username: str, current=Depends(get_current_user)):
    try:
        await delete_user(username)
        return APIResponse(message=f"用户 {username} 已删除")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# 注册路由
app.include_router(news.router)
app.include_router(stocks.router)
app.include_router(config_router.router)
app.include_router(data_router.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "scheduler": is_running()}


@app.get("/api/stats", response_model=APIResponse)
async def dashboard_stats():
    from backend.database import get_db
    from backend.services.config_service import get_config as gc
    async with get_db() as db:
        async with db.execute("SELECT COUNT(*) as cnt FROM news") as cur:
            total_news = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT COUNT(DISTINCT news_id) as cnt FROM match_results") as cur:
            total_matches = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT COUNT(*) as cnt FROM news WHERE sentiment='positive'") as cur:
            pos = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT COUNT(*) as cnt FROM news WHERE sentiment='negative'") as cur:
            neg = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT COUNT(*) as cnt FROM news WHERE sentiment='neutral'") as cur:
            neu = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT COUNT(*) as cnt FROM stocks") as cur:
            total_stocks = (await cur.fetchone())["cnt"]
        async with db.execute("SELECT MAX(created_at) as last FROM news") as cur:
            last_row = await cur.fetchone()
            last_collect = last_row["last"] if last_row else None

    # token 消耗统计
    token_usage = await gc("token_usage_total", {"input": 0, "output": 0, "calls": 0})
    if not isinstance(token_usage, dict):
        token_usage = {"input": 0, "output": 0, "calls": 0}

    provider = await gc("llm_provider", "")
    model    = await gc("llm_model", "")
    current_model = f"{provider}/{model}" if provider and model else "未配置"

    stats = DashboardStats(
        total_news=total_news,
        total_matches=total_matches,
        positive_count=pos,
        negative_count=neg,
        neutral_count=neu,
        total_stocks=total_stocks,
        last_collect_at=last_collect,
        current_model=current_model,
        scheduler_running=is_running(),
        token_usage=token_usage,
    )
    return APIResponse(data=stats.model_dump())


@app.get("/api/trending")
async def get_trending(limit: int = 20):
    """返回三个热搜平台的最新数据"""
    import json as _json
    from backend.database import get_db
    async with get_db() as db:
        result = {}
        for source, label in [("百度热搜","baidu"), ("微博热搜","weibo"), ("抖音热搜","douyin")]:
            async with db.execute(
                """SELECT id, title, content, url, published_at, created_at
                   FROM news WHERE source=? AND raw_source='trending'
                   ORDER BY COALESCE(published_at, created_at) DESC LIMIT ?""",
                (source, limit)
            ) as cur:
                rows = await cur.fetchall()
            result[label] = [{
                "id": r["id"], "title": r["title"],
                "content": r["content"], "url": r["url"],
                "created_at": r["created_at"],
            } for r in rows]
    return result


@app.get("/api/matched-ids")
async def get_matched_ids():
    """只返回已匹配的 news_id 集合，供前端标记绿点用（轻量接口）"""
    from backend.database import get_db as _get_db
    async with _get_db() as db:
        async with db.execute("SELECT news_id FROM match_results") as cur:
            rows = await cur.fetchall()
    return {"ids": [r["news_id"] for r in rows]}


@app.get("/api/news")
async def list_news(
    limit: int = 50, offset: int = 0,
    sentiment: str = "all",
):
    """所有新闻列表（分页）"""
    import json as _json
    from backend.database import get_db
    async with get_db() as db:
        sent_filter = "" if sentiment == "all" else f"AND sentiment = '{sentiment}'"
        async with db.execute(
            f"SELECT COUNT(*) as cnt FROM news WHERE raw_source != 'trending' {sent_filter}"
        ) as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute(
            f"""SELECT id, url, title, source, published_at, summary,
                       sentiment, industries, keywords, raw_source, created_at
                FROM news WHERE raw_source != 'trending' {sent_filter}
                ORDER BY COALESCE(published_at, created_at) DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
    items = [{
        "id": r["id"], "url": r["url"], "title": r["title"],
        "source": r["source"], "published_at": r["published_at"],
        "summary": r["summary"], "sentiment": r["sentiment"] or "neutral",
        "industries": _json.loads(r["industries"] or "[]"),
        "keywords": _json.loads(r["keywords"] or "[]"),
        "raw_source": r["raw_source"], "created_at": r["created_at"],
    } for r in rows]
    return {"total": total, "offset": offset, "limit": limit,
            "has_more": (offset + limit) < total, "items": items}


@app.get("/api/results")
async def list_results(limit: int = 50, offset: int = 0, sentiment: str = "all", news_id: int = None):
    """匹配结果列表，含新闻详情，按时间倒序"""
    import json as _json
    from backend.database import get_db
    async with get_db() as db:
        where = "" if sentiment == "all" else f"AND n.sentiment='{sentiment}'"
        news_filter = f"AND mr.news_id = {int(news_id)}" if news_id else ""
        async with db.execute(
            f"SELECT COUNT(*) as cnt FROM match_results mr JOIN news n ON mr.news_id=n.id WHERE 1=1 {where} {news_filter}"
        ) as cur:
            total_results = (await cur.fetchone())["cnt"]
        async with db.execute(f"""
            SELECT mr.id, mr.news_id, mr.matched_stocks, mr.created_at,
                   n.title, n.summary, n.sentiment, n.source, n.published_at,
                   n.industries, n.keywords, n.url
            FROM match_results mr
            JOIN news n ON mr.news_id = n.id
            WHERE 1=1 {where} {news_filter}
            ORDER BY COALESCE(n.published_at, n.created_at) DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)) as cur:
            rows = await cur.fetchall()
    items = []
    for r in rows:
        try:
            stocks = _json.loads(r["matched_stocks"] or "[]")
            # 从匹配结果里取第一只股票的 sentiment_impact 作为整体情感
            match_sentiment = stocks[0].get("sentiment_impact", r["sentiment"]) if stocks else r["sentiment"]
            items.append({
                "id": r["id"],
                "news_id": r["news_id"],
                "title": r["title"],
                "summary": r["summary"],
                "sentiment": r["sentiment"],
                "match_sentiment": match_sentiment,
                "source": r["source"],
                "published_at": r["published_at"],
                "industries": _json.loads(r["industries"] or "[]"),
                "keywords": _json.loads(r["keywords"] or "[]"),
                "url": r["url"],
                "matched_stocks": stocks,
                "created_at": r["created_at"],
            })
        except Exception:
            pass
    return {"total": total_results, "offset": offset, "limit": limit, "has_more": (offset + limit) < total_results, "items": items}


@app.get("/api/config/match-schedule")
async def get_match_schedule_api():
    from backend.services.scheduler import get_match_schedule
    from backend.services.config_service import get_config as _gc
    saved = await _gc("match_schedule_times", [])
    return APIResponse(data={"times": saved})


@app.post("/api/config/match-schedule")
async def set_match_schedule_api(body: dict):
    """设置定时匹配时间点，body: {"times": ["08:00","20:00"]}"""
    times = body.get("times", [])
    # 验证格式：HH:MM，小时 0-23，分钟 0-59
    import re
    def _valid_time(t):
        m = re.fullmatch(r"([0-9]{1,2}):([0-9]{2})", str(t).strip())
        if not m: return False
        return 0 <= int(m.group(1)) <= 23 and 0 <= int(m.group(2)) <= 59
    valid = [t.strip() for t in times if _valid_time(t)]
    from backend.services.config_service import set_config as _sc
    from backend.services.scheduler import update_match_schedule
    await _sc("match_schedule_times", valid)
    update_match_schedule(valid)
    return APIResponse(message=f"定时匹配已更新: {valid}")


@app.post("/api/news/{news_id}/match-now")
async def match_news_now(news_id: int):
    """对单条新闻立即执行分类（如需）+ 匹配，同步等待结果返回"""
    import json as _json
    from backend.database import get_db
    from backend.services.llm_client import get_llm_client
    from backend.services.matcher import _match_one_news
    from backend.services.config_service import get_config

    async with get_db() as db:
        async with db.execute(
            "SELECT id, title, content, summary, industries, keywords, sentiment, beneficiary_chain, news_level, time_horizon FROM news WHERE id=?",
            (news_id,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        return {"success": False, "message": "新闻不存在"}

    client = await get_llm_client()
    from backend.services.llm_client import get_embed_client
    embed_client = await get_embed_client()

    # Step 1: 分类已移除，直接执行匹配
    # Step 2: 执行匹配
    top_k = await get_config("match_top_k", 5)
    try:
        result = await _match_one_news(
            news_id=news_id,
            title=row["title"],
            summary=row["summary"] or "",
            industries=_json.loads(row["industries"] or "[]"),
            keywords=_json.loads(row["keywords"] or "[]"),
            sentiment=row["sentiment"] or "neutral",
            top_k=top_k,
            client=client,
            embed_client=embed_client,
            beneficiary_chain=_json.loads(row["beneficiary_chain"] or "[]"),
            news_level=row["news_level"] or "medium",
            time_horizon=row["time_horizon"] or "short",
        )
        if result:
            async with get_db() as db:
                await db.execute("DELETE FROM match_results WHERE news_id=?", (news_id,))
                await db.execute(
                    "INSERT INTO match_results(news_id, matched_stocks) VALUES(?,?)",
                    (news_id, _json.dumps(result, ensure_ascii=False)),
                )
                await db.commit()
        return {
            "success": True,
            "news_id": news_id,
            "news_title": row["title"],
            "matched_stocks": result or [],
        }
    except Exception as e:
        logger.error(f"即时匹配失败: {e}")
        return {"success": False, "message": f"匹配失败: {e}"}


@app.post("/api/news/{news_id}/direct-match")
async def direct_match_news(news_id: int, body: dict = None):
    """直接推理模式：新闻全文+方法论 → LLM主动推理受益股票 → 库里验证存在性
    body 可选字段：methodology（自定义方法论文本）
    """
    import json as _json
    from backend.database import get_db
    from backend.services.llm_client import get_llm_client
    from backend.services.config_service import get_config
    from fastapi import Body

    body = body or {}
    custom_methodology = body.get("methodology", "")  # 前端传入选中的方法论

    async with get_db() as db:
        async with db.execute(
            "SELECT id, title, content, summary, sentiment FROM news WHERE id=?",
            (news_id,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        return {"success": False, "message": "新闻不存在"}

    client = await get_llm_client()
    if not client:
        return {"success": False, "message": "未配置大模型"}

    top_k = await get_config("match_top_k", 5)
    company_type = await get_config("match_company_type", "leader")

    type_hints = {
        "leader": f"优先选择行业龙头（市值最大、知名度最高），共{top_k}只",
        "direct": f"选择与新闻直接相关、业务最契合的企业，共{top_k}只",
        "chain":  f"同时考虑直接受益方及产业链上下游，共{top_k}只",
        "broad":  f"覆盖整个受益行业，共{top_k}只",
    }
    type_hint = type_hints.get(company_type, type_hints["leader"])

    # 使用前端选中的方法论，没有则用内置默认
    DEFAULT_METHOD = """你是一位A股投资专家，每年年化收益超过30%，请按照以下方法论选股：

第一步：判断新闻级别（高/中/低）
第二步：提炼核心关键词（行业+技术+地区+动作+量化）
第三步：资金流向分析——钱最终花到哪里？
  - 运力/工具层：配送扩张→电动两轮车；工厂自动化→工业机器人
  - 基础设施层：互联网扩张→IDC数据中心；数字化→ERP/SaaS
  - 消耗品层：配送量增加→快递包装/锂电池
  - 上游材料：汽车扩产→锂电池/铝合金；地产松绑→水泥/钢铁
第四步：产业链映射——直接受益 > 上下游配套 > 概念炒作
第五步：严格排除新闻主体的同行竞争对手
第五步补充：只选A股（沪深京），严禁推荐港股、美股及境外上市公司
第六步：判断新闻性质（客观事实，非主观情绪）
  - positive（利好）：有明确受益方，资金有流向，如政策支持、订单落地、扩产投资
  - negative（利空）：有明确受损方，如竞争加剧、成本上升、政策收紧、需求萎缩
  - neutral（中性）：事实陈述，无明确资金方向，或影响尚不明确
第七步：区分时间维度（短线/中线/长线）"""
    active_methodology = custom_methodology.strip() if custom_methodology.strip() else DEFAULT_METHOD

    # 全文传入，保证信息完整性，不截断
    news_content = row["content"] or row["title"]
    prompt = (
        "你是一位A股投研专家。请严格按照以下方法论逐步分析新闻，输出分析结果。\n\n"
        "【新闻内容】\n" + news_content + "\n\n"
        "【匹配方法论】\n" + active_methodology + "\n\n"
        f"【选股偏好】{type_hint}\n\n"
        '请返回严格JSON对象（不加markdown）：\n'
        '{\n'
        '  "news_sentiment": "positive/negative/neutral",\n'
        '  "news_level": "高/中/低",\n'
        '  "news_summary": "一句话摘要（不超过60字）",\n'
        '  "stocks": [\n'
        '    {\n'
        '      "name": "公司名称（A股上市公司全称）",\n'
        '      "ts_code": "股票代码（如000001.SZ，不确定可留空）",\n'
        '      "benefit_tier": "第一梯队/第二梯队/第三梯队",\n'
        '      "reason": "受益逻辑（说明资金流向和直接关联，不超过50字）",\n'
        '      "time_horizon": "short/medium/long",\n'
        '      "sentiment_impact": "positive/negative/neutral"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "只返回JSON，不加任何说明。"
    )

    try:
        resp = await client.chat([
            {"role": "system", "content": "你是A股投研专家，只返回JSON对象。只推荐A股（沪深京交易所上市股票，代码以.SH/.SZ/.BJ结尾），严禁推荐港股（.HK）、美股及任何境外上市公司。"},
            {"role": "user", "content": prompt},
        ], timeout=60, no_failover=True)
        text = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        llm_output = _json.loads(text)
        # 兼容直接返回数组的旧格式
        if isinstance(llm_output, list):
            llm_results = llm_output
            news_sentiment = "neutral"
            news_level = "medium"
            news_summary = ""
        else:
            llm_results = llm_output.get("stocks", [])
            news_sentiment = llm_output.get("news_sentiment", "neutral")
            news_level = llm_output.get("news_level", "medium")
            news_summary = llm_output.get("news_summary", "")
        if not isinstance(llm_results, list):
            raise ValueError("非数组格式")
    except Exception as e:
        logger.error(f"直接推理失败: {e}")
        return {"success": False, "message": f"LLM推理失败: {e}"}

    # 直接使用 LLM 推理结果，不做股票库验证
    verified = []
    for item in llm_results:
        verified.append({
            "ts_code":          item.get("ts_code", ""),
            "name":             item.get("name", ""),
            "industry":         item.get("industry", ""),
            "score":            0.9 if item.get("benefit_tier","") == "第一梯队"
                                else 0.75 if "第二" in item.get("benefit_tier","")
                                else 0.6,
            "benefit_tier":     item.get("benefit_tier", ""),
            "reason":           item.get("reason", ""),
            "time_horizon":     item.get("time_horizon", "short"),
            "sentiment_impact": item.get("sentiment_impact", "positive"),
            "match_mode":       "direct",
        })

    if not verified:
        return {"success": False, "message": "LLM未推理出受益股票"}

    # 写入匹配结果 + 同步更新 news 表的情感/摘要/级别
    async with get_db() as db:
        await db.execute("DELETE FROM match_results WHERE news_id=?", (news_id,))
        await db.execute(
            "INSERT INTO match_results(news_id, matched_stocks) VALUES(?,?)",
            (news_id, _json.dumps(verified, ensure_ascii=False)),
        )
        # 把情感判断和摘要写回 news 表，让前端展示正确
        if news_sentiment or news_summary:
            await db.execute(
                """UPDATE news SET sentiment=?, summary=COALESCE(NULLIF(summary,''), ?),
                   news_level=? WHERE id=?""",
                (news_sentiment, news_summary, news_level, news_id)
            )
        await db.commit()

    return {
        "success":       True,
        "news_id":       news_id,
        "news_title":    row["title"],
        "news_sentiment": news_sentiment,
        "news_summary":  news_summary,
        "news_level":    news_level,
        "match_mode":    "direct",
        "total":         len(verified),
        "matched_stocks": verified,
    }


# /api/classify 端点已移除：分类由直接推理统一完成

@app.get("/api/classify/progress")
async def classify_progress_snapshot():
    """分类进度快照（轮询用）"""
    from fastapi.responses import JSONResponse
    data = get_classify_progress()
    return JSONResponse(
        content={"success": True, "message": "ok", "data": data},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}
    )


@app.get("/api/classify/progress-sse")
async def classify_progress_sse():
    """分类进度 SSE 流"""
    from fastapi.responses import StreamingResponse
    import json as _json
    async def event_stream():
        while True:
            progress = get_classify_progress()
            yield f"data: {_json.dumps(progress, ensure_ascii=False)}\n\n"
            if progress.get("done"):
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/embed/test")
async def test_embed_connection(req: dict):
    """测试 Embedding 模型连接（调 /embeddings 接口，不是 chat）"""
    import httpx as _httpx
    provider  = req.get("provider", "")
    api_key   = req.get("api_key", "")
    model     = req.get("model", "")
    base_url  = req.get("base_url", "").rstrip("/")

    # 确定实际 base_url
    PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "qwen":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    if not base_url:
        base_url = PROVIDER_URLS.get(provider, "")
    if not base_url:
        return {"success": False, "error": "请填写 Base URL"}

    try:
        async with _httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Failover-Enabled": "true",  # Gitee AI 等平台需要
                },
                json={"model": model, "input": ["测试连接"]},
            )
            if resp.status_code == 200:
                data = resp.json()
                dim = len(data["data"][0]["embedding"]) if data.get("data") else "?"
                return {"success": True, "response": f"✅ 向量维度: {dim}"}
            else:
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/llm/test")
async def test_llm_connection_api(req: dict):
    """测试大模型连接 / Embedding 连接（embed_mode=true 时走 /embeddings 接口）"""
    import httpx as _httpx
    provider = req.get("provider", "")
    api_key  = req.get("api_key", "")
    model    = req.get("model", "")
    base_url = (req.get("base_url") or "").rstrip("/")
    embed_mode = req.get("embed_mode", False)

    if not provider or not api_key or not model:
        return {"success": False, "error": "请填写提供商、模型和 API Key"}

    # Embedding 测试：调 /embeddings 端点
    if embed_mode:
        PROVIDER_URLS = {
            "openai": "https://api.openai.com/v1",
            "qwen":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "gitee":  "https://ai.gitee.com/v1",
        }
        if not base_url:
            base_url = PROVIDER_URLS.get(provider, "")
        if not base_url:
            return {"success": False, "error": "请填写 Base URL"}
        try:
            async with _httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-Failover-Enabled": "true",
                    },
                    json={"model": model, "input": ["测试连接"]},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    dim = len(data["data"][0]["embedding"]) if data.get("data") else "?"
                    return {"success": True, "response": f"✅ 向量维度: {dim}"}
                else:
                    return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # 普通 LLM 测试
    from backend.services.llm_client import test_llm_connection
    result = await test_llm_connection(provider, api_key, model, base_url or None)
    return result


@app.post("/api/config/test-key")
async def test_config_key(req: dict):
    """兼容旧版测试接口"""
    from backend.services.llm_client import test_llm_connection
    result = await test_llm_connection(
        req.get("provider", ""), req.get("api_key", ""),
        req.get("model", ""), req.get("base_url") or None
    )
    return APIResponse(success=result["success"], data=result)


@app.get("/api/match/progress")
async def get_match_progress():
    """获取当前匹配进度"""
    import json as _json
    from backend.services.config_service import get_config as _gc
    raw = await _gc("match_progress", None)
    if not raw:
        return {"done": 0, "total": 0, "current": "", "finished": True}
    try:
        return _json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"done": 0, "total": 0, "current": "", "finished": True}



@app.post("/api/direct-match-all", response_model=APIResponse)
async def trigger_direct_match_all():
    """批量直接推理：对所有未匹配的新闻用直接推理方式匹配，已匹配的跳过"""
    from backend.database import get_db as _get_db
    from backend.services.config_service import get_config as _gc, set_config as _sc
    from backend.services.llm_client import get_llm_client as _get_llm
    import asyncio as _asyncio
    import json as _json

    # 统计未匹配数量
    async with _get_db() as db:
        async with db.execute(
            """SELECT COUNT(*) as cnt FROM news n
               LEFT JOIN match_results mr ON n.id=mr.news_id
               WHERE mr.id IS NULL"""
        ) as cur:
            pending = (await cur.fetchone())["cnt"]

    if pending == 0:
        await _sc("match_progress", _json.dumps({
            "done": 0, "total": 0, "current": "", "finished": True,
            "error": None, "message": "✅ 所有新闻均已匹配完毕",
        }, ensure_ascii=False))
        return APIResponse(message="所有新闻均已匹配，无需重复处理 ✅")

    async def _run():
        import asyncio as _asyncio
        client = await _get_llm()
        if not client:
            await _sc("match_progress", _json.dumps({
                "done": 0, "total": pending, "current": "", "finished": True,
                "error": "未配置大模型", "message": "❌ 未配置大模型",
            }, ensure_ascii=False))
            return

        top_k = await _gc("match_top_k", 5)
        company_type = await _gc("match_company_type", "leader")
        custom_methodology = await _gc("selected_methodology_content", "")

        type_hints = {
            "leader": f"优先选择行业龙头（市值最大、知名度最高），共{top_k}只",
            "direct": f"选择与新闻直接相关、业务最契合的企业，共{top_k}只",
            "chain":  f"同时考虑直接受益方及产业链上下游，共{top_k}只",
            "broad":  f"覆盖整个受益行业，共{top_k}只",
        }
        type_hint = type_hints.get(company_type, type_hints["leader"])

        DEFAULT_METHOD = """你是一位A股投资专家，每年年化收益超过30%，请按照以下方法论选股：

第一步：判断新闻级别（高/中/低）
第二步：提炼核心关键词（行业+技术+地区+动作+量化）
第三步：资金流向分析——钱最终花到哪里？
  - 运力/工具层：配送扩张→电动两轮车；工厂自动化→工业机器人
  - 基础设施层：互联网扩张→IDC数据中心；数字化→ERP/SaaS
  - 消耗品层：配送量增加→快递包装/锂电池
  - 上游材料：汽车扩产→锂电池/铝合金；地产松绑→水泥/钢铁
第四步：产业链映射——直接受益 > 上下游配套 > 概念炒作
第五步：严格排除新闻主体的同行竞争对手
第五步补充：只选A股（沪深京），严禁推荐港股、美股及境外上市公司
第六步：判断新闻性质（客观事实，非主观情绪）
  - positive（利好）：有明确受益方，资金有流向，如政策支持、订单落地、扩产投资
  - negative（利空）：有明确受损方，如竞争加剧、成本上升、政策收紧、需求萎缩
  - neutral（中性）：事实陈述，无明确资金方向，或影响尚不明确
第七步：区分时间维度（短线/中线/长线）"""
        active_methodology = custom_methodology.strip() if custom_methodology.strip() else DEFAULT_METHOD

        # 拉取所有未匹配新闻
        async with _get_db() as db:
            async with db.execute(
                """SELECT n.id, n.title, n.content FROM news n
                   LEFT JOIN match_results mr ON n.id=mr.news_id
                   WHERE mr.id IS NULL ORDER BY n.id DESC"""
            ) as cur:
                rows = await cur.fetchall()

        total = len(rows)
        done = 0
        failed_ids = []  # 失败的新闻ID，最后重试
        await _sc("match_progress", _json.dumps({
            "done": 0, "total": total, "current": "", "finished": False,
            "error": None, "message": f"准备直接推理 {total} 条...",
        }, ensure_ascii=False))

        for row in rows:
            news_id = row["id"]
            title   = row["title"]
            news_content = row["content"] or title

            await _sc("match_progress", _json.dumps({
                "done": done, "total": total,
                "current": title[:40], "finished": False,
                "error": None, "message": f"推理中 {done}/{total}: {title[:30]}",
            }, ensure_ascii=False))

            prompt = (
                "你是一位A股投研专家。请严格按照以下方法论逐步分析新闻，输出分析结果。\n\n"
                "【新闻内容】\n" + news_content + "\n\n"
                "【匹配方法论】\n" + active_methodology + "\n\n"
                f"【选股偏好】{type_hint}\n\n"
                '请返回严格JSON对象（不加markdown）：\n'
                '{\n'
                '  "news_sentiment": "positive/negative/neutral",\n'
                '  "news_level": "高/中/低",\n'
                '  "news_summary": "一句话摘要（不超过60字）",\n'
                '  "stocks": [\n'
                '    {\n'
                '      "name": "公司名称（A股上市公司全称）",\n'
                '      "ts_code": "股票代码（如000001.SZ，不确定可留空）",\n'
                '      "benefit_tier": "第一梯队/第二梯队/第三梯队",\n'
                '      "reason": "受益逻辑（说明资金流向和直接关联，不超过50字）",\n'
                '      "time_horizon": "short/medium/long",\n'
                '      "sentiment_impact": "positive/negative/neutral"\n'
                '    }\n'
                '  ]\n'
                '}\n'
                "只返回JSON，不加任何说明。"
            )

            try:
                resp = await _asyncio.wait_for(
                    client.chat([
                        {"role": "system", "content": "你是A股投研专家，只返回JSON对象。只推荐A股（沪深京交易所上市股票，代码以.SH/.SZ/.BJ结尾），严禁推荐港股（.HK）、美股及任何境外上市公司。"},
                        {"role": "user",   "content": prompt},
                    ], timeout=60, no_failover=True),
                    timeout=75  # 双重保护，最多等75秒
                )
                text = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                llm_output = _json.loads(text)
                if isinstance(llm_output, list):
                    llm_results = llm_output
                    news_sentiment, news_level, news_summary = "neutral", "中", ""
                else:
                    llm_results  = llm_output.get("stocks", [])
                    news_sentiment = llm_output.get("news_sentiment", "neutral")
                    news_level     = llm_output.get("news_level", "中")
                    news_summary   = llm_output.get("news_summary", "")

                verified = []
                for item in llm_results:
                    verified.append({
                        "ts_code":          item.get("ts_code", ""),
                        "name":             item.get("name", ""),
                        "industry":         item.get("industry", ""),
                        "score":            0.9 if item.get("benefit_tier","") == "第一梯队"
                                            else 0.75 if "第二" in item.get("benefit_tier","")
                                            else 0.6,
                        "benefit_tier":     item.get("benefit_tier", ""),
                        "reason":           item.get("reason", ""),
                        "time_horizon":     item.get("time_horizon", "short"),
                        "sentiment_impact": item.get("sentiment_impact", "positive"),
                        "match_mode":       "direct",
                    })

                async with _get_db() as db:
                    await db.execute("DELETE FROM match_results WHERE news_id=?", (news_id,))
                    await db.execute(
                        "INSERT INTO match_results(news_id, matched_stocks) VALUES(?,?)",
                        (news_id, _json.dumps(verified, ensure_ascii=False)),
                    )
                    if verified and (news_sentiment or news_summary):
                        # 只有有受益股时才写 sentiment/summary
                        await db.execute(
                            """UPDATE news SET sentiment=?,
                               summary=COALESCE(NULLIF(summary,''), ?),
                               news_level=? WHERE id=?""",
                            (news_sentiment, news_summary, news_level, news_id)
                        )
                    await db.commit()
            except Exception as e:
                logger.warning(f"直接推理批量: news_id={news_id} 失败，加入重试队列: {e}")
                failed_ids.append(news_id)

            done += 1

        # ── 重试失败的新闻（排到最后，只重试一次）────────────────────────────
        if failed_ids:
            retry_total = len(failed_ids)
            logger.info(f"开始重试 {retry_total} 条失败新闻")
            await _sc("match_progress", _json.dumps({
                "done": done, "total": total + retry_total,
                "current": "", "finished": False,
                "error": None, "message": f"重试 {retry_total} 条失败新闻...",
            }, ensure_ascii=False))

            # 重新拉取失败条目的内容
            async with _get_db() as db:
                placeholders = ",".join(["?"]*len(failed_ids))
                async with db.execute(
                    f"SELECT id, title, content FROM news WHERE id IN ({placeholders})",
                    failed_ids
                ) as cur:
                    retry_rows = await cur.fetchall()

            for retry_row in retry_rows:
                news_id = retry_row["id"]
                news_content = retry_row["content"] or retry_row["title"]
                await _sc("match_progress", _json.dumps({
                    "done": done, "total": total + retry_total,
                    "current": retry_row["title"][:40], "finished": False,
                    "error": None, "message": f"重试 {done}/{total+retry_total}: {retry_row['title'][:30]}",
                }, ensure_ascii=False))
                prompt = (
                    "你是一位A股投研专家。请严格按照以下方法论逐步分析新闻，输出分析结果。\\n\\n"
                    "【新闻内容】\\n" + news_content + "\\n\\n"
                    "【匹配方法论】\\n" + active_methodology + "\\n\\n"
                    f"【选股偏好】{type_hint}\\n\\n"
                    '请返回严格JSON对象（不加markdown）：\n'
                    '{\n'
                    '  "news_sentiment": "positive/negative/neutral",\n'
                    '  "news_level": "高/中/低",\n'
                    '  "news_summary": "一句话摘要（不超过60字）",\n'
                    '  "stocks": [{"name":"公司名","ts_code":"代码","benefit_tier":"第一/二/三梯队","reason":"受益逻辑","time_horizon":"short/medium/long","sentiment_impact":"positive/negative/neutral"}]\n'
                    '}\n'
                    "只返回JSON，不加任何说明。"
                )
                try:
                    resp = await _asyncio.wait_for(
                        client.chat([
                            {"role": "system", "content": "你是A股投研专家，只返回JSON对象。只推荐A股（沪深京交易所上市股票，代码以.SH/.SZ/.BJ结尾），严禁推荐港股（.HK）、美股及任何境外上市公司。"},
                            {"role": "user",   "content": prompt},
                        ], timeout=60, no_failover=True),
                        timeout=75
                    )
                    text = resp.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    llm_output = _json.loads(text)
                    llm_results  = llm_output.get("stocks", []) if isinstance(llm_output, dict) else llm_output
                    news_sentiment = llm_output.get("news_sentiment", "neutral") if isinstance(llm_output, dict) else "neutral"
                    news_level     = llm_output.get("news_level", "中") if isinstance(llm_output, dict) else "中"
                    news_summary   = llm_output.get("news_summary", "") if isinstance(llm_output, dict) else ""
                    verified = []
                    for item in llm_results:
                        verified.append({
                            "ts_code": item.get("ts_code", ""), "name": item.get("name", ""),
                            "industry": item.get("industry", ""),
                            "score": 0.9 if "第一" in item.get("benefit_tier","") else 0.75 if "第二" in item.get("benefit_tier","") else 0.6,
                            "benefit_tier": item.get("benefit_tier", ""), "reason": item.get("reason", ""),
                            "time_horizon": item.get("time_horizon", "short"),
                            "sentiment_impact": item.get("sentiment_impact", "positive"),
                            "match_mode": "direct",
                        })
                    async with _get_db() as db:
                        await db.execute("DELETE FROM match_results WHERE news_id=?", (news_id,))
                        await db.execute("INSERT INTO match_results(news_id, matched_stocks) VALUES(?,?)",
                            (news_id, _json.dumps(verified, ensure_ascii=False)))
                        if verified and (news_sentiment or news_summary):
                            await db.execute("UPDATE news SET sentiment=?, summary=COALESCE(NULLIF(summary,''),?), news_level=? WHERE id=?",
                                (news_sentiment, news_summary, news_level, news_id))
                        await db.commit()
                except Exception as e2:
                    logger.warning(f"重试失败 news_id={news_id}: {e2}，跳过")
                done += 1

        await _sc("match_progress", _json.dumps({
            "done": done, "total": total + len(failed_ids) if failed_ids else total,
            "current": "", "finished": True,
            "error": None, "message": f"✅ 直接推理完成，共处理 {done} 条",
        }, ensure_ascii=False))

    _asyncio.create_task(_run())
    return APIResponse(message=f"直接推理已启动，共 {pending} 条待处理")


@app.post("/api/match", response_model=APIResponse)
async def trigger_match():
    """手动触发：只对「已分类但未匹配」的新闻执行匹配，已匹配的跳过"""
    import asyncio as _asyncio
    from backend.database import get_db as _get_db

    # 先统计待处理数量，告知用户
    async with _get_db() as db:
        async with db.execute(
            """SELECT COUNT(*) as cnt FROM news n
               LEFT JOIN match_results mr ON n.id=mr.news_id
               WHERE n.summary IS NOT NULL AND n.summary!='[内容审核限制]' AND mr.id IS NULL"""
        ) as cur:
            pending = (await cur.fetchone())["cnt"]

    if pending == 0:
        from backend.services.config_service import set_config as _set_config
        import json as _json
        await _set_config("match_progress", _json.dumps({
            "done": 0, "total": 0, "current": "", "finished": True,
            "error": None, "message": "✅ 最新新闻均已匹配完毕",
        }, ensure_ascii=False))
        return APIResponse(message="所有已分类新闻均已匹配，无需重复处理 ✅")

    async def _run():
        nonlocal pending
        from backend.services.matcher import match_pending_news, _write_match_progress
        from backend.database import get_db as _gdb
        total_processed = 0
        retry_count = 0
        try:
            while True:
                async with _gdb() as db:
                    async with db.execute(
                        """SELECT COUNT(*) as cnt FROM news n
                           LEFT JOIN match_results mr ON n.id=mr.news_id
                           WHERE n.summary IS NOT NULL
                             AND n.summary!='[内容审核限制]'
                             AND mr.id IS NULL"""
                    ) as cur:
                        remaining = (await cur.fetchone())["cnt"]

                if remaining == 0:
                    await _write_match_progress(pending, pending, "", True)
                    break

                # 分类任务可能持续新增已分类新闻，动态更新 pending 上限
                pending = max(pending, remaining)
                done_count = max(0, pending - remaining)
                from backend.services.llm_client import get_last_failover_msg as _glfm2
                _fo_msg2 = _glfm2()
                _match_msg = f"{_fo_msg2}  剩余 {remaining} 条待匹配" if _fo_msg2 else f"剩余 {remaining} 条待匹配"
                await _write_match_progress(done_count, pending, _match_msg, False)
                processed = await match_pending_news(batch_size=min(20, remaining))
                total_processed += processed

                async with _gdb() as db2:
                    async with db2.execute(
                        """SELECT COUNT(*) as cnt FROM news n
                           LEFT JOIN match_results mr ON n.id=mr.news_id
                           WHERE n.summary IS NOT NULL
                             AND n.summary!='[内容审核限制]'
                             AND mr.id IS NULL"""
                    ) as cur2:
                        new_remaining = (await cur2.fetchone())["cnt"]

                if new_remaining < remaining:
                    retry_count = 0
                    continue

                retry_count += 1
                wait = min(30 * retry_count, 300)
                await _write_match_progress(done_count, pending,
                                            f"本轮暂无进展，后台将在 {wait}s 后继续重试（剩余 {new_remaining} 条）",
                                            False)
                await _asyncio.sleep(wait)
            logger.info(f"手动匹配全部完成: {total_processed} 条")
        except Exception as e:
            logger.error(f"手动匹配异常: {e}")
            await _write_match_progress(total_processed, pending, "", True, str(e))
    _asyncio.create_task(_run())
    return APIResponse(message=f"匹配任务已触发，待处理 {pending} 条（已匹配的自动跳过）")


@app.get("/api/news/window")
async def news_window(
    unit: str = "day", value: int = 1,
    limit: int = 50, offset: int = 0,
    sentiment: str = "all",
):
    """
    按时间窗口返回新闻（分页）
    unit: day | hour
    value: 窗口大小
    limit: 每页条数（默认50）
    offset: 偏移量（分页用）
    """
    import json as _json
    from backend.database import get_db

    if unit == "day":
        since_expr = f"datetime('now', '+8 hours', '-{value} days')"
        label = f"最近 {value} 天" if value > 1 else "今天"
    else:
        since_expr = f"datetime('now', '+8 hours', '-{value} hours')"
        label = f"最近 {value} 小时"

    sent_filter = "" if sentiment == "all" else f"AND sentiment = '{sentiment}'"

    async with get_db() as db:
        # 总数（排除热搜，热搜已在仪表盘单独展示）
        async with db.execute(
            f"""SELECT COUNT(*) as cnt FROM news
                WHERE raw_source != 'trending'
                AND datetime(created_at, '+8 hours') >= {since_expr}
                {sent_filter}""",
        ) as cur:
            total = (await cur.fetchone())["cnt"]
        # 分页数据
        async with db.execute(
            f"""SELECT * FROM news
                WHERE raw_source != 'trending'
                AND datetime(created_at, '+8 hours') >= {since_expr}
                {sent_filter}
                ORDER BY COALESCE(published_at, created_at) DESC
                LIMIT ? OFFSET ?""",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
        # 历史条数
        async with db.execute(
            f"""SELECT COUNT(*) as cnt FROM news
                WHERE raw_source != 'trending'
                AND datetime(created_at, '+8 hours') < {since_expr}""",
        ) as cur:
            history_count = (await cur.fetchone())["cnt"]

        # 窗口内匹配结果数
        async with db.execute(
            f"""SELECT COUNT(*) as cnt FROM match_results mr
                JOIN news n ON mr.news_id = n.id
                WHERE n.raw_source != 'trending'
                AND datetime(n.created_at, '+8 hours') >= {since_expr}""",
        ) as cur:
            match_count = (await cur.fetchone())["cnt"]

        # 窗口内利好/利空数
        async with db.execute(
            f"""SELECT sentiment, COUNT(*) as cnt FROM news
                WHERE raw_source != 'trending'
                AND datetime(created_at, '+8 hours') >= {since_expr}
                AND sentiment IN ('positive','negative')
                GROUP BY sentiment""",
        ) as cur:
            sent_rows = await cur.fetchall()
        pos_count = next((r["cnt"] for r in sent_rows if r["sentiment"] == "positive"), 0)
        neg_count = next((r["cnt"] for r in sent_rows if r["sentiment"] == "negative"), 0)

    items = []
    for row in rows:
        items.append({
            "id": row["id"], "url": row["url"], "title": row["title"],
            "source": row["source"], "published_at": row["published_at"],
            "summary": row["summary"], "sentiment": row["sentiment"] or "neutral",
            "industries": _json.loads(row["industries"] or "[]"),
            "keywords": _json.loads(row["keywords"] or "[]"),
            "raw_source": row["raw_source"], "created_at": row["created_at"],
        })
    return {
        "label": label, "unit": unit, "value": value,
        "total": total, "offset": offset, "limit": limit,
        "has_more": (offset + limit) < total,
        "count": total, "history_count": history_count,
        "match_count": match_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "items": items,
    }


@app.get("/api/history/news")
async def history_news(
    unit: str = "day", value: int = 1,
    limit: int = 500, offset: int = 0,
    sentiment: str = "all"
):
    """历史新闻（当前窗口之外的）"""
    import json as _json
    from backend.database import get_db

    if unit == "day":
        since_expr = f"datetime('now', '+8 hours', '-{value} days')"
    else:
        since_expr = f"datetime('now', '+8 hours', '-{value} hours')"

    sent_where = f"AND sentiment='{sentiment}'" if sentiment != "all" else ""

    async with get_db() as db:
        async with db.execute(
            f"""SELECT COUNT(*) as cnt FROM news
                WHERE datetime(created_at, '+8 hours') < {since_expr} {sent_where}"""
        ) as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute(
            f"""SELECT * FROM news
                WHERE datetime(created_at, '+8 hours') < {since_expr} {sent_where}
                ORDER BY COALESCE(published_at, created_at) DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        ) as cur:
            rows = await cur.fetchall()

    items = []
    for row in rows:
        items.append({
            "id": row["id"], "title": row["title"], "source": row["source"],
            "published_at": row["published_at"], "summary": row["summary"],
            "sentiment": row["sentiment"] or "neutral",
            "industries": _json.loads(row["industries"] or "[]"),
            "keywords": _json.loads(row["keywords"] or "[]"),
            "created_at": row["created_at"],
        })
    return {"total": total, "items": items}


@app.get("/api/history/results")
async def history_results(
    unit: str = "day", value: int = 1,
    limit: int = 500, offset: int = 0
):
    """历史匹配结果（当前窗口之外的）"""
    import json as _json
    from backend.database import get_db

    if unit == "day":
        since_expr = f"datetime('now', '+8 hours', '-{value} days')"
    else:
        since_expr = f"datetime('now', '+8 hours', '-{value} hours')"

    async with get_db() as db:
        async with db.execute(
            f"""SELECT COUNT(*) as cnt FROM match_results mr
                JOIN news n ON mr.news_id=n.id
                WHERE datetime(n.created_at, '+8 hours') < {since_expr}"""
        ) as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute(
            f"""SELECT mr.id, mr.news_id, mr.matched_stocks, mr.created_at,
                       n.title, n.summary, n.sentiment
                FROM match_results mr JOIN news n ON mr.news_id=n.id
                WHERE datetime(n.created_at, '+8 hours') < {since_expr}
                ORDER BY COALESCE(n.published_at, n.created_at) DESC LIMIT ? OFFSET ?""",
            (limit, offset)
        ) as cur:
            rows = await cur.fetchall()

    items = []
    for row in rows:
        stocks = _json.loads(row["matched_stocks"] or "[]")
        si = stocks[0].get("sentiment_impact", row["sentiment"]) if stocks else row["sentiment"]
        items.append({
            "id": row["id"], "news_id": row["news_id"],
            "title": row["title"], "summary": row["summary"],
            "sentiment": row["sentiment"], "match_sentiment": si,
            "matched_stocks": stocks, "created_at": row["created_at"],
        })
    return {"total": total, "items": items}



@app.post("/api/collect/newsapi", response_model=APIResponse)
async def trigger_collect_newsapi():
    """仅触发 NewsAPI.ai 采集"""
    import asyncio
    from backend.services.news_collector import get_collect_progress
    prog = get_collect_progress()
    if prog.get("running"):
        return APIResponse(message=f"采集任务正在运行中... {prog.get('message','')}")

    async def _run():
        from backend.services.news_collector import run_collection
        try:
            await run_collection(sources=["newsapi"])
        except Exception as e:
            from loguru import logger
            logger.error(f"NewsAPI采集异常: {e}")
    asyncio.create_task(_run())
    return APIResponse(message="NewsAPI 采集已触发")


@app.post("/api/collect", response_model=APIResponse)
async def trigger_collect():
    import asyncio
    from backend.services.news_collector import get_collect_progress
    prog = get_collect_progress()
    if prog.get("running"):
        return APIResponse(message=f"采集任务正在运行中... {prog.get('message','')}")

    async def _full_pipeline():
        from backend.services.news_collector import run_collection
        try:
            result = await run_collection()
        except Exception as e:
            from loguru import logger
            logger.error(f"流水线异常: {e}")
    asyncio.create_task(_full_pipeline())
    return APIResponse(message="采集任务已触发")


@app.get("/api/collect/progress")
async def collect_progress_snapshot():
    """采集进度快照"""
    from backend.services.news_collector import get_collect_progress
    from fastapi.responses import JSONResponse
    data = get_collect_progress()
    return JSONResponse(
        content={"success": True, "message": "ok", "data": data},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}
    )


# 静态文件 & SPA fallback（必须放最后）
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # API 路径不走 SPA fallback
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        index = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
        return {"error": "frontend not found"}
