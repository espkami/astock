"""APScheduler 定时任务"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

_scheduler: AsyncIOScheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
_MAIN_LOOP = None  # 由 main.py lifespan 注入
_collect_job_id = "news_collect"
_clean_job_id = "data_clean"
_stock_job_id = "stock_update"


async def _collect_task():
    from backend.services.news_collector import run_collection
    logger.info("定时采集开始")
    result = await run_collection()
    logger.info(f"定时采集完成: {result}")


async def _clean_task():
    from backend.services.config_service import get_config
    from backend.database import get_db
    days = await get_config("data_retention_days", 30)
    async with get_db() as db:
        await db.execute("DELETE FROM news WHERE created_at < datetime('now', ? || ' days')", (f"-{days}",))
        await db.execute("DELETE FROM match_results WHERE created_at < datetime('now', ? || ' days')", (f"-{days}",))
        await db.commit()
    logger.info(f"定时清理完成: 保留 {days} 天")


async def _stock_update_task():
    from backend.services.stock_service import update_stock_list
    logger.info("定时股票更新开始")
    await update_stock_list()


def _process_task():
    """后台分类任务（仅分类，不匹配）"""
    import asyncio
    async def _run():
        from backend.services.news_processor import process_pending_news
        try:
            n = await process_pending_news(batch_size=10)
            if n > 0:
                logger.info(f"后台分类: {n} 条")
        except Exception as e:
            logger.error(f"后台处理异常: {e}")
    try:
        from backend.services import scheduler as _self
        loop = _self._MAIN_LOOP
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_run(), loop)
        else:
            logger.warning("后台处理：event loop 未就绪，跳过本次")
    except Exception as e:
        logger.error(f"后台处理调度失败: {e}")


def start_scheduler():
    if not _scheduler.running:
        # 每天 02:00 清理旧数据
        _scheduler.add_job(_clean_task, CronTrigger(hour=2, minute=0), id=_clean_job_id, replace_existing=True)
        _scheduler.start()
        logger.info("调度器已启动")


def stop_scheduler():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("调度器已停止")


async def update_collect_interval(seconds: int):
    """动态更新采集间隔（兼容旧接口，设置所有源）"""
    await update_source_intervals(newsapi=seconds, rss=seconds, llm_search=seconds)


async def update_source_intervals(
    newsapi: int = None,
    rss: int = None,
    llm_search: int = None,
    trending: int = None,
):
    """按路源独立更新采集间隔"""
    if not _scheduler.running:
        return

    source_map = {
        "newsapi_collect":  (newsapi,    ["newsapi"]),
        "rss_collect":      (rss,        ["rss"]),
        "llm_collect":      (llm_search, ["llm"]),
        "trending_collect": (trending,   ["trending"]),
    }
    for job_id, (interval, sources) in source_map.items():
        if interval is None:
            continue
        async def _make_task(srcs):
            async def _t():
                from backend.services.news_collector import run_collection
                await run_collection(sources=srcs)
            return _t
        task_fn_name = f"_collect_{job_id}"

        def _make_sync(srcs):
            def _sync():
                import asyncio
                async def _run():
                    from backend.services.news_collector import run_collection
                    await run_collection(sources=srcs)
                from backend.services import scheduler as _self
                loop = _self._MAIN_LOOP
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(_run(), loop)
            return _sync

        _scheduler.add_job(
            _make_sync(sources),
            IntervalTrigger(seconds=max(60, interval)),
            id=job_id,
            replace_existing=True,
        )
        logger.info(f"{job_id} 间隔更新为 {interval}s")


def is_running() -> bool:
    return _scheduler.running


_match_job_id = "scheduled_match"


def update_match_schedule(times: list[str]):
    """
    更新定时匹配时间点，times 格式如 ["08:00","12:00","20:00"]
    空列表则移除定时匹配任务
    """
    # 先移除旧的（job id 是 scheduled_match_0, _1 ... 需逐一移除）
    for job in _scheduler.get_jobs():
        if job.id.startswith(_match_job_id + "_") or job.id == _match_job_id:
            try:
                _scheduler.remove_job(job.id)
            except Exception:
                pass

    if not times or not _scheduler.running:
        return

    # 多个时间点 → 多个 cron job，共用同一 id 不行，用 idx
    for idx, t in enumerate(times):
        try:
            hour, minute = map(int, t.split(":"))
            _scheduler.add_job(
                _match_task,
                CronTrigger(hour=hour, minute=minute, timezone="Asia/Shanghai"),
                id=f"{_match_job_id}_{idx}",
                replace_existing=True,
            )
            logger.info(f"定时匹配已设置: {t}")
        except Exception as e:
            logger.error(f"设置定时匹配失败 {t}: {e}")


def _match_task():
    """定时匹配任务（同步包装）"""
    import asyncio

    async def _run():
        from backend.services.news_processor import process_pending_news
        from backend.services.matcher import match_pending_news
        try:
            logger.info("定时匹配开始")
            n = await process_pending_news(batch_size=50)
            if n > 0:
                logger.info(f"定时分类: {n} 条")
                await asyncio.sleep(2)
            m = await match_pending_news(batch_size=50)
            logger.info(f"定时匹配完成: {m} 条")
        except Exception as e:
            logger.error(f"定时匹配异常: {e}")

    from backend.services import scheduler as _self
    loop = _self._MAIN_LOOP
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(_run(), loop)
    else:
        logger.warning("定时匹配：event loop 未就绪")


def get_match_schedule() -> list[str]:
    """获取当前已设置的定时匹配时间点"""
    times = []
    for job in _scheduler.get_jobs():
        if job.id.startswith(_match_job_id + "_"):
            trigger = job.trigger
            # CronTrigger → 提取 hour/minute
            try:
                h = str(trigger.fields[5]).zfill(2)   # hour field
                m = str(trigger.fields[6]).zfill(2)   # minute field
                times.append(f"{h}:{m}")
            except Exception:
                pass
    return sorted(times)
