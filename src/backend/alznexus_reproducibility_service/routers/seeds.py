from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

try:
    # Try relative imports first (when run as package)
    from ..database import get_db
    from ..models import RandomSeed, SeedRotationPolicy
    from ..schemas import (
        SeedRequest, SeedResponse,
        SeedRotationPolicyCreate, SeedRotationPolicyResponse
    )
    from ..crud import (
        create_random_seed, get_seed_by_value, get_seeds_by_agent, get_active_seeds,
        create_seed_rotation_policy, get_seed_rotation_policy, get_active_policies
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from database import get_db
    from models import RandomSeed, SeedRotationPolicy
    from schemas import (
        SeedRequest, SeedResponse,
        SeedRotationPolicyCreate, SeedRotationPolicyResponse
    )
    from crud import (
        create_random_seed, get_seed_by_value, get_seeds_by_agent, get_active_seeds,
        create_seed_rotation_policy, get_seed_rotation_policy, get_active_policies
    )

router = APIRouter()

@router.post("/seeds", response_model=SeedResponse)
async def request_random_seed(
    request: SeedRequest,
    db: Session = Depends(get_db)
) -> SeedResponse:
    """
    Request a random seed for reproducible analysis.
    Automatically manages seed rotation based on policies.
    """
    try:
        # Check if there's an active rotation policy for this agent/analysis type
        policy = db.query(SeedRotationPolicy).filter(
            SeedRotationPolicy.agent_type == request.agent_id.split('_')[0],  # Extract agent type
            SeedRotationPolicy.analysis_type == request.analysis_type,
            SeedRotationPolicy.is_active == True
        ).first()

        if policy:
            # Check if rotation is needed
            current_time = datetime.utcnow()
            needs_rotation = (
                (current_time - policy.last_rotation) > timedelta(hours=policy.rotation_interval_hours) or
                policy.current_seed_id is None
            )

            if needs_rotation:
                # Create new seed for this policy
                seed_request = SeedRequest(
                    purpose=f"{request.analysis_type}_rotation",
                    agent_id=request.agent_id,
                    task_id=request.task_id,
                    analysis_type=request.analysis_type,
                    parameters={"rotation_policy": policy.policy_name}
                )
                new_seed = create_random_seed(db, seed_request)
                policy.current_seed_id = new_seed.id
                policy.last_rotation = current_time
                db.commit()

                seed = new_seed
            else:
                # Use existing seed from policy
                seed = db.query(RandomSeed).filter(RandomSeed.id == policy.current_seed_id).first()
                if not seed:
                    # Fallback: create new seed
                    seed = create_random_seed(db, request)
                    policy.current_seed_id = seed.id
                    db.commit()
        else:
            # No policy: create new seed
            seed = create_random_seed(db, request)

        return SeedResponse(
            seed_id=seed.id,
            seed_value=seed.seed_value,
            purpose=seed.purpose,
            agent_id=seed.agent_id,
            task_id=seed.task_id,
            analysis_type=seed.analysis_type,
            parameters=seed.parameters,
            created_at=seed.created_at,
            expires_at=seed.expires_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to allocate random seed: {str(e)}")

@router.get("/seeds/{seed_value}", response_model=SeedResponse)
async def get_seed_info(
    seed_value: int,
    db: Session = Depends(get_db)
) -> SeedResponse:
    """
    Get information about a specific seed by its value
    """
    try:
        seed = get_seed_by_value(db, seed_value)
        if not seed:
            raise HTTPException(status_code=404, detail="Seed not found")

        return SeedResponse(
            seed_id=seed.id,
            seed_value=seed.seed_value,
            purpose=seed.purpose,
            agent_id=seed.agent_id,
            task_id=seed.task_id,
            analysis_type=seed.analysis_type,
            parameters=seed.parameters,
            created_at=seed.created_at,
            expires_at=seed.expires_at
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve seed information: {str(e)}")

@router.get("/seeds/agent/{agent_id}", response_model=List[SeedResponse])
async def get_agent_seeds(
    agent_id: str,
    db: Session = Depends(get_db)
) -> List[SeedResponse]:
    """
    Get all seeds used by a specific agent
    """
    try:
        seeds = get_seeds_by_agent(db, agent_id)
        return [
            SeedResponse(
                seed_id=seed.id,
                seed_value=seed.seed_value,
                purpose=seed.purpose,
                agent_id=seed.agent_id,
                task_id=seed.task_id,
                analysis_type=seed.analysis_type,
                parameters=seed.parameters,
                created_at=seed.created_at,
                expires_at=seed.expires_at
            )
            for seed in seeds
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent seeds: {str(e)}")

@router.get("/seeds/active", response_model=List[SeedResponse])
async def get_active_seeds_list(
    db: Session = Depends(get_db)
) -> List[SeedResponse]:
    """
    Get all currently active (non-expired) seeds
    """
    try:
        seeds = get_active_seeds(db)
        return [
            SeedResponse(
                seed_id=seed.id,
                seed_value=seed.seed_value,
                purpose=seed.purpose,
                agent_id=seed.agent_id,
                task_id=seed.task_id,
                analysis_type=seed.analysis_type,
                parameters=seed.parameters,
                created_at=seed.created_at,
                expires_at=seed.expires_at
            )
            for seed in seeds
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve active seeds: {str(e)}")

@router.post("/policies", response_model=SeedRotationPolicyResponse)
async def create_rotation_policy(
    policy: SeedRotationPolicyCreate,
    db: Session = Depends(get_db)
) -> SeedRotationPolicyResponse:
    """
    Create a new seed rotation policy
    """
    try:
        # Check if policy name already exists
        existing = get_seed_rotation_policy(db, policy.policy_name)
        if existing:
            raise HTTPException(status_code=400, detail="Policy name already exists")

        db_policy = create_seed_rotation_policy(db, policy)

        return SeedRotationPolicyResponse(
            id=db_policy.id,
            policy_name=db_policy.policy_name,
            agent_type=db_policy.agent_type,
            analysis_type=db_policy.analysis_type,
            rotation_interval_hours=db_policy.rotation_interval_hours,
            max_uses_per_seed=db_policy.max_uses_per_seed,
            seed_range_start=db_policy.seed_range_start,
            seed_range_end=db_policy.seed_range_end,
            last_rotation=db_policy.last_rotation,
            current_seed_id=db_policy.current_seed_id,
            is_active=db_policy.is_active,
            created_at=db_policy.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create rotation policy: {str(e)}")

@router.get("/policies", response_model=List[SeedRotationPolicyResponse])
async def get_rotation_policies(
    db: Session = Depends(get_db)
) -> List[SeedRotationPolicyResponse]:
    """
    Get all active seed rotation policies
    """
    try:
        policies = get_active_policies(db)
        return [
            SeedRotationPolicyResponse(
                id=policy.id,
                policy_name=policy.policy_name,
                agent_type=policy.agent_type,
                analysis_type=policy.analysis_type,
                rotation_interval_hours=policy.rotation_interval_hours,
                max_uses_per_seed=policy.max_uses_per_seed,
                seed_range_start=policy.seed_range_start,
                seed_range_end=policy.seed_range_end,
                last_rotation=policy.last_rotation,
                current_seed_id=policy.current_seed_id,
                is_active=policy.is_active,
                created_at=policy.created_at
            )
            for policy in policies
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve rotation policies: {str(e)}")

@router.get("/policies/{policy_name}", response_model=SeedRotationPolicyResponse)
async def get_rotation_policy(
    policy_name: str,
    db: Session = Depends(get_db)
) -> SeedRotationPolicyResponse:
    """
    Get a specific seed rotation policy by name
    """
    try:
        policy = get_seed_rotation_policy(db, policy_name)
        if not policy:
            raise HTTPException(status_code=404, detail="Rotation policy not found")

        return SeedRotationPolicyResponse(
            id=policy.id,
            policy_name=policy.policy_name,
            agent_type=policy.agent_type,
            analysis_type=policy.analysis_type,
            rotation_interval_hours=policy.rotation_interval_hours,
            max_uses_per_seed=policy.max_uses_per_seed,
            seed_range_start=policy.seed_range_start,
            seed_range_end=policy.seed_range_end,
            last_rotation=policy.last_rotation,
            current_seed_id=policy.current_seed_id,
            is_active=policy.is_active,
            created_at=policy.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve rotation policy: {str(e)}")

@router.put("/policies/{policy_name}/deactivate")
async def deactivate_rotation_policy(
    policy_name: str,
    db: Session = Depends(get_db)
):
    """
    Deactivate a seed rotation policy
    """
    try:
        policy = get_seed_rotation_policy(db, policy_name)
        if not policy:
            raise HTTPException(status_code=404, detail="Rotation policy not found")

        policy.is_active = False
        db.commit()

        return {"message": f"Policy '{policy_name}' deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to deactivate rotation policy: {str(e)}")

@router.post("/seeds/{seed_value}/extend")
async def extend_seed_lifetime(
    seed_value: int,
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """
    Extend the lifetime of a specific seed
    """
    try:
        seed = get_seed_by_value(db, seed_value)
        if not seed:
            raise HTTPException(status_code=404, detail="Seed not found")

        if seed.expires_at:
            seed.expires_at = seed.expires_at + timedelta(hours=hours)
        else:
            seed.expires_at = datetime.utcnow() + timedelta(hours=hours)

        db.commit()

        return {
            "message": f"Seed {seed_value} lifetime extended by {hours} hours",
            "new_expiry": seed.expires_at
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to extend seed lifetime: {str(e)}")