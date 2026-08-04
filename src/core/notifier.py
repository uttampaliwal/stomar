"""Critical event notifications (P5.1) + daily summaries (B7).

Sends notifications on critical events (pipeline failure, kill switch,
drift alerts, drawdown breach) and the daily paper-trading summary.
Outputs to the structured log always, and to Telegram and/or email when
configured via environment variables:

    STOMAR_TELEGRAM_BOT_TOKEN=...    bot token from @BotFather
    STOMAR_TELEGRAM_CHAT_ID=...      chat id to deliver to (numeric)

    STOMAR_SMTP_HOST=...             e.g. smtp.gmail.com
    STOMAR_SMTP_PORT=587
    STOMAR_SMTP_USER=...
    STOMAR_SMTP_PASSWORD=...         (app password, never your real one)
    STOMAR_SMTP_TO=...               recipient address(es), comma separated

Delivery is best-effort: failures to send are logged, never raised.
"""

import logging
import os
import smtplib
import time
import traceback
from email.message import EmailMessage
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

APP_NAME = "StoMar"

# Freshness threshold (days) beyond which a ticker is flagged in the summary
STALE_DATA_DAYS = 5
# Telegram messages cap at 4096 characters — stay well under
_TELEGRAM_MAX_CHARS = 4000


def _smtp_config() -> dict:
    return {
        "host": os.environ.get("STOMAR_SMTP_HOST", ""),
        "port": int(os.environ.get("STOMAR_SMTP_PORT", "587")),
        "user": os.environ.get("STOMAR_SMTP_USER", ""),
        "password": os.environ.get("STOMAR_SMTP_PASSWORD", ""),
        "to": [a.strip() for a in os.environ.get("STOMAR_SMTP_TO", "").split(",") if a.strip()],
    }


def _telegram_config() -> tuple[str, str]:
    return (
        os.environ.get("STOMAR_TELEGRAM_BOT_TOKEN", ""),
        os.environ.get("STOMAR_TELEGRAM_CHAT_ID", ""),
    )


def _send_email(subject: str, body: str) -> bool:
    cfg = _smtp_config()
    if not (cfg["host"] and cfg["user"] and cfg["password"] and cfg["to"]):
        logger.debug("SMTP not configured — skipping email notification")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = f"[{APP_NAME}] {subject}"
        msg["From"] = cfg["user"]
        msg["To"] = ", ".join(cfg["to"])
        msg.set_content(body)
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=15) as server:
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
        logger.info("Email notification sent: %s", subject)
        return True
    except Exception as e:
        logger.warning("Email notification failed (%s): %s", subject, e)
        return False


def _send_telegram(body: str) -> bool:
    token, chat_id = _telegram_config()
    if not (token and chat_id):
        logger.debug("Telegram not configured — skipping Telegram notification")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": body[:_TELEGRAM_MAX_CHARS],
            "disable_web_page_preview": True,
        }
        for attempt in (1, 2):
            try:
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 200:
                    logger.info("Telegram notification sent")
                    return True
                logger.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
                return False
            except requests.RequestException:
                if attempt == 1:
                    time.sleep(2)
                    continue
                raise
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)
        return False
    return False


def send_message(subject: str, body: str) -> bool:
    """Best-effort delivery to every configured channel (Telegram, email)."""
    sent = _send_telegram(f"[{APP_NAME}] {subject}\n\n{body}")
    sent_email = _send_email(subject, body)
    return sent or sent_email


def notify(event: str, severity: str = "info", details: str = "",
           exc: BaseException = None) -> None:
    """Send a notification for a critical event.

    Args:
        event: Short event name, e.g. "pipeline_failure".
        severity: "info", "warning", or "critical".
        details: Human-readable description.
        exc: Optional exception to include a traceback for.
    """
    ts = datetime.now().isoformat()
    body = f"{APP_NAME} {severity.upper()} | {ts}\n\nEvent: {event}\nDetails: {details}"
    if exc is not None:
        body += "\n\nTraceback:\n" + "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    log_fn = logger.critical if severity == "critical" else logger.warning
    log_fn("%s | event=%s | details=%s", severity.upper(), event, details)

    if severity == "critical":
        send_message(event, body)


def send_daily_summary(health: dict, daily: dict | None = None,
                       paper: dict | None = None) -> bool:
    """Build and send the daily summary digest (B7).

    Args:
        health: The dict produced by AutoPipeline._run_health_sweep().
        daily: Optional result dict of the daily orchestrator step.
        paper: Optional result dict of the paper-trading step.
    """
    date_str = str(health.get("timestamp", datetime.now().isoformat()))[:10]
    lines = [
        f"{APP_NAME} daily summary | {date_str}",
        "",
        f"Status: {health.get('status', '?')}",
    ]
    if daily is not None:
        if daily.get("skipped"):
            lines.append("Daily loop: already ran today (skipped)")
        else:
            lines.append(
                f"Decisions: {daily.get('decisions', 0)} | "
                f"Errors: {daily.get('errors', 0)}"
            )
    if paper is not None:
        if paper.get("skipped"):
            lines.append("Paper trades: skipped")
        else:
            lines.append(f"Paper trades: {paper.get('trades', 0)}")
    lines.append(f"Paper days: {health.get('paper_days', 0)}")

    if health.get("kill_switch_active"):
        lines.append("⚠️ KILL SWITCH ACTIVE — trading halted")

    stale = [
        t for t, d in health.get("data_freshness_days", {}).items()
        if d is not None and d > STALE_DATA_DAYS
    ]
    if stale:
        lines.append(f"Stale data (> {STALE_DATA_DAYS}d): {', '.join(sorted(stale)[:10])}")
    missing = [
        t for t, age in health.get("model_age_days", {}).items() if age is None
    ]
    if missing:
        lines.append(f"Models missing: {', '.join(sorted(missing)[:10])}")

    return send_message("Daily summary", "\n".join(lines))
