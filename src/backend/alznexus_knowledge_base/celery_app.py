from celery import Celery
import os

# Celery configuration
CELERY_BROKER_URL = os.getenv("KNOWLEDGE_CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("KNOWLEDGE_CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "alznexus_knowledge_base",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["alznexus_knowledge_base.tasks"]
)

# Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "alznexus_knowledge_base.tasks.ingest_knowledge_task": {"queue": "knowledge_ingestion"},
        "alznexus_knowledge_base.tasks.update_embeddings_task": {"queue": "knowledge_processing"},
        "alznexus_knowledge_base.tasks.validate_knowledge_task": {"queue": "knowledge_validation"},
        "alznexus_knowledge_base.tasks.extract_insights_task": {"queue": "knowledge_insights"},
    },
    task_default_queue="knowledge_default",
    task_default_exchange="knowledge_exchange",
    task_default_routing_key="knowledge_default",
)

if __name__ == "__main__":
    celery_app.start()