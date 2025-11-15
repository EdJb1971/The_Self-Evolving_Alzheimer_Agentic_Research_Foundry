from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import RiskAssessment
from ..schemas import RiskAssessmentRequest, RiskAssessmentResponse
from ..tasks import perform_risk_assessment_task
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/assess", response_model=RiskAssessmentResponse)
async def perform_risk_assessment(
    request: RiskAssessmentRequest,
    db: Session = Depends(get_db)
) -> RiskAssessmentResponse:
    """
    Perform clinical risk assessment for research findings.

    Evaluates clinical significance, false positive rates, and decision confidence.
    """
    try:
        # Perform risk assessment
        result = await perform_risk_assessment_task(
            assessment_type=request.assessment_type,
            research_question=request.research_question,
            input_parameters=request.input_parameters,
            clinical_thresholds=request.clinical_thresholds
        )

        # Store assessment results
        assessment = RiskAssessment(
            assessment_type=request.assessment_type,
            research_question=request.research_question,
            input_parameters=request.input_parameters,
            risk_metrics=result["risk_metrics"],
            confidence_intervals=result["confidence_intervals"],
            recommendations=result["recommendations"],
            clinical_thresholds_used=request.clinical_thresholds
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)

        return RiskAssessmentResponse(
            assessment_id=assessment.id,
            risk_metrics=result["risk_metrics"],
            confidence_intervals=result["confidence_intervals"],
            recommendations=result["recommendations"],
            clinical_significance=result["clinical_significance"],
            decision_confidence=result["decision_confidence"],
            metadata={
                "assessment_type": request.assessment_type,
                "computation_time": result.get("computation_time"),
                "method": "clinical_risk_assessment"
            }
        )

    except Exception as e:
        logger.error(f"Risk assessment failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@router.get("/assessments")
async def list_risk_assessments(
    assessment_type: str = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List risk assessments with optional filtering"""
    query = db.query(RiskAssessment)
    if assessment_type:
        query = query.filter(RiskAssessment.assessment_type == assessment_type)

    assessments = query.offset(skip).limit(limit).all()
    return [
        {
            "id": assessment.id,
            "assessment_type": assessment.assessment_type,
            "research_question": assessment.research_question[:100] + "..." if len(assessment.research_question) > 100 else assessment.research_question,
            "clinical_significance": assessment.clinical_significance,
            "decision_confidence": assessment.decision_confidence,
            "created_at": assessment.created_at
        }
        for assessment in assessments
    ]

@router.get("/thresholds/{disease_area}")
async def get_clinical_thresholds(disease_area: str):
    """
    Get predefined clinical significance thresholds for different disease areas.

    These thresholds help determine if research findings have clinical relevance.
    """
    thresholds_library = {
        "alzheimer": {
            "biomarker_effect_size": {
                "small": 0.2,
                "medium": 0.5,
                "large": 0.8
            },
            "cognitive_decline_threshold": 0.3,  # points per year
            "amyloid_reduction_target": 0.25,  # 25% reduction
            "tau_reduction_target": 0.20,  # 20% reduction
            "acceptable_false_positive_rate": 0.10,
            "minimum_statistical_power": 0.80
        },
        "cancer": {
            "survival_improvement": 0.15,  # 15% improvement
            "tumor_response_rate": 0.30,  # 30% response rate
            "progression_free_survival": 3.0,  # months
            "acceptable_toxicity_rate": 0.20
        },
        "cardiovascular": {
            "risk_reduction": 0.20,  # 20% risk reduction
            "blood_pressure_target": 130,  # systolic mmHg
            "cholesterol_target": 100,  # LDL mg/dL
            "event_rate_threshold": 0.05  # 5% annual event rate
        }
    }

    if disease_area not in thresholds_library:
        available_areas = list(thresholds_library.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Clinical thresholds not found for disease area '{disease_area}'. Available: {available_areas}"
        )

    return {
        "disease_area": disease_area,
        "clinical_thresholds": thresholds_library[disease_area],
        "description": f"Clinical significance thresholds for {disease_area} research"
    }