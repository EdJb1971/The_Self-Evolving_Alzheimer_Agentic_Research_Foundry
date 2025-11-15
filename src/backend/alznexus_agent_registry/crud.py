from sqlalchemy.orm import Session
from . import models, schemas

def register_agent(db: Session, agent_data: schemas.AgentRegister):
    db_agent = db.query(models.RegisteredAgent).filter(models.RegisteredAgent.agent_id == agent_data.agent_id).first()
    if db_agent:
        # Update existing agent
        db_agent.capabilities = agent_data.capabilities
        db_agent.api_endpoint = agent_data.api_endpoint
        db.commit()
        db.refresh(db_agent)
        return db_agent, False # False indicates update
    else:
        # Create new agent
        db_agent = models.RegisteredAgent(
            agent_id=agent_data.agent_id,
            capabilities=agent_data.capabilities,
            api_endpoint=agent_data.api_endpoint
        )
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        return db_agent, True # True indicates creation

def get_agent_by_id(db: Session, agent_id: str):
    return db.query(models.RegisteredAgent).filter(models.RegisteredAgent.agent_id == agent_id).first()

def get_all_agents(db: Session):
    return db.query(models.RegisteredAgent).all()
