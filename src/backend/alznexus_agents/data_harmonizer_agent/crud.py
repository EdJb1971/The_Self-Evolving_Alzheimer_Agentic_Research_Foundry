from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def create_agent_task(db: Session, task: schemas.AgentTaskCreate):
    db_task = models.AgentTask(
        agent_id=task.agent_id,
        orchestrator_task_id=task.orchestrator_task_id,
        task_description=task.task_description,
        metadata_json=task.metadata_json,
        status="PENDING"
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

def get_agent_task(db: Session, task_id: int):
    return db.query(models.AgentTask).filter(models.AgentTask.id == task_id).first()

def update_agent_task_status(db: Session, task_id: int, status: str, result_data: dict = None):
    db_task = db.query(models.AgentTask).filter(models.AgentTask.id == task_id).first()
    if db_task:
        db_task.status = status
        if result_data:
            db_task.result_data = result_data
        db.commit()
        db.refresh(db_task)
    return db_task

def get_or_create_agent_state(db: Session, agent_id: str):
    db_state = db.query(models.AgentState).filter(models.AgentState.agent_id == agent_id).first()
    if not db_state:
        db_state = models.AgentState(agent_id=agent_id)
        db.add(db_state)
        db.commit()
        db.refresh(db_state)
    return db_state

def update_agent_state(db: Session, agent_id: str, current_goal: str = None, current_task_id: int = None, last_reflection_at: datetime = None, metadata_json: dict = None):
    db_state = db.query(models.AgentState).filter(models.AgentState.agent_id == agent_id).first()
    if db_state:
        if current_goal is not None: db_state.current_goal = current_goal
        if current_task_id is not None: db_state.current_task_id = current_task_id
        if last_reflection_at is not None: db_state.last_reflection_at = last_reflection_at
        if metadata_json is not None: db_state.metadata_json = metadata_json
        db.commit()
        db.refresh(db_state)
    return db_state
