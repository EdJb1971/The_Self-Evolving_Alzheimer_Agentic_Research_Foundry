from sqlalchemy.orm import Session
from . import models, schemas

def create_research_goal(db: Session, goal: schemas.ResearchGoalCreate):
    db_goal = models.ResearchGoal(goal_text=goal.goal_text, status="ACTIVE")
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)
    return db_goal

def get_research_goal(db: Session, goal_id: int):
    return db.query(models.ResearchGoal).filter(models.ResearchGoal.id == goal_id).first()

def get_active_research_goal(db: Session):
    """Retrieves the most recently created active research goal."""
    return db.query(models.ResearchGoal).filter(models.ResearchGoal.status == "ACTIVE").order_by(models.ResearchGoal.created_at.desc()).first()

def update_research_goal_status(db: Session, goal_id: int, status: str):
    db_goal = db.query(models.ResearchGoal).filter(models.ResearchGoal.id == goal_id).first()
    if db_goal:
        db_goal.status = status
        db.commit()
        db.refresh(db_goal)
    return db_goal

def create_orchestrator_task(db: Session, task: schemas.OrchestratorTaskCreate):
    db_task = models.OrchestratorTask(
        goal_id=task.goal_id,
        task_type=task.task_type,
        description=task.description,
        assigned_agent_id=task.assigned_agent_id,
        metadata_json=task.metadata_json,
        status="PENDING"
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

def get_orchestrator_task(db: Session, task_id: int):
    return db.query(models.OrchestratorTask).filter(models.OrchestratorTask.id == task_id).first()

def update_orchestrator_task_status(db: Session, task_id: int, status: str, result_data: dict = None):
    db_task = db.query(models.OrchestratorTask).filter(models.OrchestratorTask.id == task_id).first()
    if db_task:
        db_task.status = status
        if result_data:
            db_task.metadata_json = result_data # Store result in metadata_json for simplicity
        db.commit()
        db.refresh(db_task)
    return db_task
