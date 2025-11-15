from celery import Celery
import os

# Celery configuration
celery_app = Celery(
    "alznexus_statistical_engine",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["alznexus_statistical_engine.tasks"]
)

# Celery settings
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,

    # Result settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        'retry_policy': {'timeout': 5.0}
    },

    # Task routing
    task_routes={
        'alznexus_statistical_engine.tasks.perform_correlation_analysis': {'queue': 'statistical'},
        'alznexus_statistical_engine.tasks.perform_hypothesis_test': {'queue': 'statistical'},
        'alznexus_statistical_engine.tasks.perform_cross_validation': {'queue': 'statistical'},
        'alznexus_statistical_engine.tasks.generate_data_quality_report': {'queue': 'statistical'},
        'alznexus_statistical_engine.tasks.perform_power_analysis': {'queue': 'statistical'},
    },

    # Task time limits
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
)