"""股票相关路由"""
import asyncio
import json
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from backend.database import get_db
from backend.models import StockListResponse, StockItem, APIResponse
from backend.services import stock_service

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("")
async def list_stocks(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, le=200),
    keyword: str = Query(""),
):
    offset = (page - 1) * page_size
    where = "WHERE s.name LIKE ? OR s.ts_code LIKE ?" if keyword else ""
    params = [f"%{keyword}%", f"%{keyword}%"] if keyword else []

    async with get_db() as db:
        async with db.execute(f"SELECT COUNT(*) as cnt FROM stocks s {where}", params) as cur:
            total = (await cur.fetchone())["cnt"]
        async with db.execute(
            f"""SELECT s.ts_code, s.name, s.market, s.industry,
                       COALESCE(sp.business_desc, '') as business_desc
                FROM stocks s LEFT JOIN stock_profile sp ON s.ts_code=sp.ts_code
                {where} ORDER BY s.name LIMIT ? OFFSET ?""",
            params + [page_size, offset],
        ) as cur:
            rows = await cur.fetchall()

    items = [dict(r) for r in rows]
    return {"total": total, "page": page, "page_size": page_size, "items": items}



@router.get("/stats")
async def stock_stats():
    stats = await stock_service.get_stock_stats()
    return APIResponse(data=stats)



@router.post("/update", response_model=APIResponse)
async def trigger_update():
    if stock_service._stock_list_running:
        return APIResponse(message="股票列表更新已在运行中，请等待完成")
    if stock_service._profiles_running:
        return APIResponse(message="主营业务补全已在运行中，请等待完成后再触发全量更新")
    asyncio.create_task(_run_full_update())
    return APIResponse(message="股票更新任务已触发")



@router.post("/refill-profiles", response_model=APIResponse)
async def trigger_refill_profiles():
    """强制重新补全主营业务（清除兜底值，重新从 cninfo/LLM 拉取）"""
    if stock_service._profiles_running:
        return APIResponse(message="补全任务已在运行中，请等待完成后再试")
    # 清除所有兜底值（llm_filled=0 且 <30字）
    async with get_db() as db:
        await db.execute(
            "DELETE FROM stock_profile WHERE llm_filled=0 AND length(business_desc) < 30"
        )
        await db.commit()
    asyncio.create_task(stock_service.update_stock_profiles(limit=9999))
    return APIResponse(message="主营业务重新补全任务已触发")


async def _run_full_update():
    await stock_service.update_stock_list()
    await stock_service.update_stock_profiles(limit=9999)
    # 简介补全完成后自动生成主营业务项目标签，进度写入主进度流
    from backend.services.tag_service import generate_all_tags, get_tag_progress
    stock_service._set_progress("tags", 0, 1, "🏷️ 准备生成主营业务项目标签...", done=False)
    await generate_all_tags(force=False)
    # 同步最终标签进度到主进度流
    tp = get_tag_progress()
    stock_service._set_progress(
        "tags", tp.get("current", 0), tp.get("total", 1),
        tp.get("message", "标签生成完成"), done=True
    )



@router.get("/update-progress-snapshot")
async def update_progress_snapshot():
    """进度快照（普通 JSON，供 SSE 断开后 fallback 轮询）"""
    return APIResponse(data=stock_service.get_progress())



@router.post("/generate-tags", response_model=APIResponse)
async def trigger_generate_tags(force: bool = False):
    """触发批量生成股票标签"""
    from backend.services.tag_service import generate_all_tags
    asyncio.create_task(generate_all_tags(force=force))
    return APIResponse(message="标签生成任务已触发")



@router.get("/tag-progress-snapshot")
async def tag_progress_snapshot():
    """标签生成进度快照"""
    from backend.services.tag_service import get_tag_progress
    return APIResponse(data=get_tag_progress())



@router.get("/tag-progress")
async def tag_progress_sse():
    """标签生成进度 SSE 流"""
    from backend.services.tag_service import get_tag_progress
    async def event_stream():
        while True:
            progress = get_tag_progress()
            yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
            if progress.get("done"):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@router.post("/generate-board-tags", response_model=APIResponse)
async def trigger_generate_board_tags():
    """触发批量生成板块标签（概念/行业板块 → 成分股映射）"""
    from backend.services.board_tag_service import generate_board_tags
    asyncio.create_task(generate_board_tags())
    return APIResponse(message="板块标签生成任务已触发")



@router.get("/board-tag-progress")
async def board_tag_progress_sse():
    """板块标签生成进度 SSE"""
    from backend.services.board_tag_service import get_board_progress
    async def event_stream():
        while True:
            progress = get_board_progress()
            yield f"data: {json.dumps(progress, ensure_ascii=False)}\n\n"
            if progress.get("done"):
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@router.get("/board-tag-progress-snapshot")
async def board_tag_progress_snapshot():
    from backend.services.board_tag_service import get_board_progress
    return APIResponse(data=get_board_progress())



@router.get("/update-progress")
async def update_progress():
    """SSE 进度流"""
    async def event_stream():
        import asyncio
        idle_count = 0
        while True:
            progress = stock_service.get_progress()
            data = json.dumps(progress, ensure_ascii=False)
            yield f"data: {data}\n\n"
            # 只有 profiles 阶段完成（或 idle 超时）才真正结束
            if progress.get("done") and progress.get("stage") in ("profiles", "idle"):
                break
            # 如果 done=True 但 stage 是 stocks，继续等 profiles 开始（最多 5s）
            if progress.get("done") and progress.get("stage") == "stocks":
                idle_count += 1
                if idle_count > 5:
                    break
            else:
                idle_count = 0
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{ts_code}/mainbz")
async def get_stock_mainbz(ts_code: str):
    """获取单只股票主营产品构成+占比（AKShare 东方财富）"""
    import asyncio as _asyncio
    try:
        import akshare as ak
        parts = ts_code.split(".")
        em_code = parts[1] + parts[0] if len(parts) == 2 else ts_code
        df = await _asyncio.wait_for(
            _asyncio.to_thread(ak.stock_zygc_em, symbol=em_code),
            timeout=12
        )
        if df is None or df.empty:
            return APIResponse(data={"items": [], "source": "无数据"})

        df_s = df.sort_values("报告日期", ascending=False)
        latest_date = str(df_s["报告日期"].iloc[0])[:10]

        # 按行业分类优先，无则按产品分类
        result = {}
        for cat in ["按行业分类", "按产品分类", "按地区分类"]:
            items = df_s[
                (df_s["报告日期"].astype(str).str[:10] == latest_date) &
                (df_s["分类类型"] == cat)
            ].sort_values("收入比例", ascending=False)
            if not items.empty:
                result[cat] = [
                    {
                        "name":  str(row.get("主营构成", "")),
                        "ratio": round(float(row.get("收入比例", 0)) * 100, 1),
                        "income": float(row.get("主营收入", 0) or 0),
                    }
                    for _, row in items.iterrows()
                ]

        return APIResponse(data={
            "report_date": latest_date,
            "categories": result,
            "source": "东方财富(AKShare)"
        })
    except Exception as e:
        return APIResponse(data={"items": [], "source": f"获取失败: {str(e)[:50]}"})



@router.get("/{ts_code}/tags")
async def get_stock_tags_api(ts_code: str):
    """获取单只股票标签"""
    from backend.services.tag_service import get_stock_tags
    tags = await get_stock_tags(ts_code)
    return APIResponse(data=tags)



@router.put("/{ts_code}/tags")
async def update_stock_tags_api(ts_code: str, body: dict):
    """手动更新单只股票标签"""
    import json as _json
    from backend.services.tag_service import init_tags_table
    await init_tags_table()
    all_tags = list(dict.fromkeys(
        body.get("products", []) + body.get("techs", []) +
        body.get("sectors", []) + body.get("themes", [])
    ))
    async with get_db() as db:
        await db.execute("""
            INSERT OR REPLACE INTO stock_tags
            (ts_code, products, techs, sectors, chain_pos, themes, all_tags, updated_at)
            VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
        """, (
            ts_code,
            _json.dumps(body.get("products", []),  ensure_ascii=False),
            _json.dumps(body.get("techs", []),     ensure_ascii=False),
            _json.dumps(body.get("sectors", []),   ensure_ascii=False),
            _json.dumps(body.get("chain_pos", []), ensure_ascii=False),
            _json.dumps(body.get("themes", []),    ensure_ascii=False),
            _json.dumps(all_tags,                  ensure_ascii=False),
        ))
        await db.commit()
    return APIResponse(message="标签已更新")


