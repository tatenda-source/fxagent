from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from pipeline.orchestrator import Orchestrator
from config import UPDATE_INTERVAL_MINUTES


_scheduler = None
_orchestrator = None


def start_scheduler():
    """Start the background scheduler for periodic pipeline runs."""
    global _scheduler, _orchestrator

    if _scheduler is not None:
        logger.info("Scheduler already running")
        return

    _orchestrator = Orchestrator()
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _run_pipeline,
        "interval",
        minutes=UPDATE_INTERVAL_MINUTES,
        id="forex_pipeline",
    )
    _scheduler.start()
    logger.info(f"Scheduler started — running every {UPDATE_INTERVAL_MINUTES} minutes")


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
        _scheduler = None
        logger.info("Scheduler stopped")


def _run_pipeline():
    try:
        _orchestrator.run_full_pipeline()
    except Exception as e:
        logger.error(f"Scheduled pipeline run failed: {e}")


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running
