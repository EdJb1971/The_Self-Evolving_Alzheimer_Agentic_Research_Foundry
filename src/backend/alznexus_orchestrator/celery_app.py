import os
from celery import Celery

CELERY_BROKER_URL = os.getenv("ORCHESTRATOR_CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("ORCHESTRATOR_CELERY_RESULT_BACKEND")

if not CELERY_BROKER_URL:
    raise ValueError("ORCHESTRATOR_CELERY_BROKER_URL environment variable not set.")
if not CELERY_RESULT_BACKEND:
    raise ValueError("ORCHESTRATOR_CELERY_RESULT_BACKEND environment variable not set.")

celery_app = Celery(
    "alznexus_orchestrator",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["src.backend.alznexus_orchestrator.tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)
