"""FCM push notifications (req 11).

firebase-admin is imported lazily so this module always imports when the package
is absent (req 15.8). Delivery retries up to 3 times with exponential backoff
(1s, 2s, 4s); on permanent failure it logs an ERROR and returns False (req 11.5).
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


class FCMNotifier:
    def __init__(self, credentials_path: str) -> None:
        try:
            import firebase_admin
            from firebase_admin import credentials
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "firebase-admin is not installed but the fcm feature is enabled"
            ) from exc
        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)

    async def send(self, device_token: str, title: str, body: str,
                   data: Optional[dict] = None) -> bool:
        from firebase_admin import messaging
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in (data or {}).items()},
            token=device_token,
        )
        for attempt in range(MAX_ATTEMPTS):
            try:
                await asyncio.to_thread(messaging.send, message)
                return True
            except Exception as exc:
                backoff = 2 ** attempt
                logger.warning(
                    f"[FCM] Send failed attempt {attempt + 1}/{MAX_ATTEMPTS}: {exc}. "
                    f"Retrying in {backoff}s" if attempt < MAX_ATTEMPTS - 1 else
                    f"[FCM] Send failed attempt {attempt + 1}/{MAX_ATTEMPTS}: {exc}"
                )
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(backoff)
        logger.error(f"[FCM] Permanent delivery failure for token={device_token[:8]}... (status=FAILED)")
        return False

    async def send_task_completed(self, device_token: str, task_id: int, task_title: str) -> None:
        await self.send(
            device_token=device_token,
            title="Task Completed",
            body=f'"{task_title}" has been completed.',
            data={"task_id": str(task_id), "event_type": "task_completed"},
        )

    async def send_plan_created(self, device_token: str, subtask_count: int) -> None:
        await self.send(
            device_token=device_token,
            title="Agent Working",
            body=f"Working on your request ({subtask_count} steps).",
            data={"subtask_count": str(subtask_count), "event_type": "plan_created"},
        )


# Singleton — assigned in main.py lifespan when the fcm flag is enabled.
fcm_notifier: Optional[FCMNotifier] = None
