import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from celery import Celery

CELERY_BROKER_URL = os.getenv("LLM_CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("LLM_CELERY_RESULT_BACKEND")

if not CELERY_BROKER_URL:
    raise ValueError("LLM_CELERY_BROKER_URL environment variable not set.")
if not CELERY_RESULT_BACKEND:
    raise ValueError("LLM_CELERY_RESULT_BACKEND environment variable not set.")

celery_app = Celery(
    "alznexus_llm_service",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["src.backend.alznexus_llm_service.tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)
