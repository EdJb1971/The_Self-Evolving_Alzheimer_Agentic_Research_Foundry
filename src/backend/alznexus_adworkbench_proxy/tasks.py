import time
import json
import requests
import os
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud

ADWORKBENCH_BASE_URL = os.getenv("ADWORKBENCH_BASE_URL", "https://adworkbench.example.com/api")
ADWORKBENCH_API_KEY = os.getenv("ADWORKBENCH_API_KEY")

@celery_app.task(bind=True, name="execute_federated_query")
def execute_federated_query(self, query_id: int):
    db: Session = SessionLocal()
    try:
        crud.update_adworkbench_query_status(db, query_id, "PROCESSING")
        
        # Retrieve the query details from DB
        query_obj = crud.get_adworkbench_query(db, query_id)
        if not query_obj:
            raise ValueError("Query not found")
        
        # Prepare the request to AD Workbench
        headers = {
            "Authorization": f"Bearer {ADWORKBENCH_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query_obj.query_text,
            "parameters": query_obj.parameters or {}
        }
        
        # Submit the federated query
        submit_url = f"{ADWORKBENCH_BASE_URL}/federated-query"
        response = requests.post(submit_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("job_id")
        
        if not job_id:
            raise ValueError("No job_id returned from AD Workbench")
        
        # Poll for completion
        poll_url = f"{ADWORKBENCH_BASE_URL}/job/{job_id}"
        max_polls = 60  # 10 minutes max
        poll_count = 0
        
        while poll_count < max_polls:
            time.sleep(10)  # Poll every 10 seconds
            poll_response = requests.get(poll_url, headers=headers, timeout=30)
            poll_response.raise_for_status()
            status_data = poll_response.json()
            
            status = status_data.get("status")
            if status == "COMPLETED":
                result = status_data.get("result", {})
                result_json = json.dumps(result)
                crud.update_adworkbench_query_status(db, query_id, "COMPLETED", result_json)
                return {"query_id": query_id, "status": "COMPLETED", "result": result}
            elif status == "FAILED":
                error_msg = status_data.get("error", "Unknown error")
                crud.update_adworkbench_query_status(db, query_id, "FAILED", error_msg)
                raise Exception(f"AD Workbench query failed: {error_msg}")
            elif status == "PROCESSING":
                poll_count += 1
                continue
            else:
                raise Exception(f"Unknown status: {status}")
        
        # Timeout
        crud.update_adworkbench_query_status(db, query_id, "FAILED", "Query timed out")
        raise Exception("Query timed out after 10 minutes")
        
    except requests.RequestException as e:
        crud.update_adworkbench_query_status(db, query_id, "FAILED", f"Request error: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    except Exception as e:
        crud.update_adworkbench_query_status(db, query_id, "FAILED", str(e))
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()
