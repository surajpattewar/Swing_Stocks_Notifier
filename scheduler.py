"""
Optional: keep this process running 24/7 (e.g. on a small VPS) and it will
fire the screener automatically every weekday after market close.
If you're using cron or GitHub Actions instead, you don't need this file —
just call `python main.py` directly on a schedule.
"""
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from main import main as run_once

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def job():
    logger.info("Running scheduled screener job...")
    try:
        run_once()
    except Exception as e:
        logger.exception("Screener run failed: %s", e)


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    # Runs Mon-Fri at 3:45 PM IST (after NSE close at 3:30 PM)
    scheduler.add_job(job, CronTrigger(day_of_week="mon-fri", hour=15, minute=45))
    logger.info("Scheduler started. Waiting for next run (Mon-Fri 15:45 IST)...")
    job()  # run once immediately on startup too, remove if undesired
    scheduler.start()
