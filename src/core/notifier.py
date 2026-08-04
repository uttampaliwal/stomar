"""Critical event notifications (P5.1).

Sends notifications on critical events (pipeline failure, kill switch,
drift alerts, drawdown breach). Outputs to the structured log always,
and to email when SMTP is configured via environment variables:

    STOMAR_SMTP_HOST=...          e.g. smtp.gmail.com
    STOMAR_SMTP_PORT=587
    STOMAR_SMTP_USER=...
    STOMAR_SMTP_PASSWORD=...      (app password, never your real one)
    STOMAR_SMTP_TO=...            recipient address(es), comma separated

Email is best-effort: failures to send are logged, never raised.
"""

import logging
import os
import smtplib
import traceback
from email.message import EmailMessage
from datetime import datetime

logger = logging.getLogger(__name__)

APP_NAME = "StoMar"


def _smtp_config() -> dict:
    return {
        "host": os.environ.get("STOMAR_SMTP_HOST", ""),
        "port": int(os.environ.get("STOMAR_SMTP_PORT", "587")),
        "user": os.environ.get("STOMAR_SMTP_USER", ""),
        "password": os.environ.get("STOMAR_SMTP_PASSWORD", ""),
        "to": [a.strip() for a in os.environ.get("STOMAR_SMTP_TO", "").split(",") if a.strip()],
    }


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
        _send_email(event, body)
