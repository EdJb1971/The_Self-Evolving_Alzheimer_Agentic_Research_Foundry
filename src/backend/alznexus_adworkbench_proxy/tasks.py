import time
import json
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud

@celery_app.task(bind=True, name="simulate_federated_query")
def simulate_federated_query(self, query_id: int):
    db: Session = SessionLocal()
    try:
        crud.update_adworkbench_query_status(db, query_id, "PROCESSING")
        # Simulate a long-running federated query operation
        time.sleep(10) # Simulate 10 seconds of work

        # Mock result data
        mock_result = {
            "status": "success",
            "data": [
                {"patient_id": "AD-001", "biomarker_a": 1.2, "biomarker_b": 0.8},
                {"patient_id": "AD-002", "biomarker_a": 1.5, "biomarker_b": 0.9}
            ],
            "message": "Federated query completed successfully."
        }
        result_json = json.dumps(mock_result)

        crud.update_adworkbench_query_status(db, query_id, "COMPLETED", result_json)
        return {"query_id": query_id, "status": "COMPLETED", "result": mock_result}
    except Exception as e:
        crud.update_adworkbench_query_status(db, query_id, "FAILED", str(e))
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()
