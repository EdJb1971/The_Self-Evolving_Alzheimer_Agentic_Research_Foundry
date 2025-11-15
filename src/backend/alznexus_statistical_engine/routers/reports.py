from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime, timedelta

from ..database import get_db
from ..models import StatisticalAnalysis, DataQualityReport, ValidationMetric
from ..schemas import (
    StatisticalAnalysisResponse,
    DataQualityReportResponse,
    AnalysisSummaryRequest, AnalysisSummaryResponse,
    ReportGenerationRequest, ReportGenerationResponse
)
from ..crud import (
    get_statistical_analyses,
    get_data_quality_reports,
    get_statistical_analysis,
    get_data_quality_report
)

router = APIRouter()

@router.get("/analyses", response_model=List[StatisticalAnalysisResponse])
async def get_analyses(
    skip: int = 0,
    limit: int = 100,
    analysis_type: Optional[str] = None,
    db: Session = Depends(get_db)
) -> List[StatisticalAnalysisResponse]:
    """
    Retrieve list of statistical analyses with optional filtering
    """
    try:
        analyses = get_statistical_analyses(db, skip=skip, limit=limit, analysis_type=analysis_type)
        return [
            StatisticalAnalysisResponse(
                id=analysis.id,
                analysis_type=analysis.analysis_type,
                method=analysis.method,
                parameters=json.loads(analysis.parameters) if analysis.parameters else {},
                results=json.loads(analysis.results) if analysis.results else {},
                confidence_level=analysis.confidence_level,
                created_at=analysis.created_at
            )
            for analysis in analyses
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analyses: {str(e)}")

@router.get("/analyses/{analysis_id}", response_model=StatisticalAnalysisResponse)
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
) -> StatisticalAnalysisResponse:
    """
    Retrieve a specific statistical analysis by ID
    """
    try:
        analysis = get_statistical_analysis(db, analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return StatisticalAnalysisResponse(
            id=analysis.id,
            analysis_type=analysis.analysis_type,
            method=analysis.method,
            parameters=json.loads(analysis.parameters) if analysis.parameters else {},
            results=json.loads(analysis.results) if analysis.results else {},
            confidence_level=analysis.confidence_level,
            created_at=analysis.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")

@router.get("/quality-reports", response_model=List[DataQualityReportResponse])
async def get_quality_reports(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[DataQualityReportResponse]:
    """
    Retrieve list of data quality reports
    """
    try:
        reports = get_data_quality_reports(db, skip=skip, limit=limit)
        return [
            DataQualityReportResponse(
                id=report.id,
                dataset_name=report.dataset_name,
                report_data=json.loads(report.report_data) if report.report_data else {},
                quality_score=report.quality_score,
                issues_found=report.issues_found,
                created_at=report.created_at
            )
            for report in reports
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve quality reports: {str(e)}")

@router.get("/quality-reports/{report_id}", response_model=DataQualityReportResponse)
async def get_quality_report(
    report_id: int,
    db: Session = Depends(get_db)
) -> DataQualityReportResponse:
    """
    Retrieve a specific data quality report by ID
    """
    try:
        report = get_data_quality_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Quality report not found")

        return DataQualityReportResponse(
            id=report.id,
            dataset_name=report.dataset_name,
            report_data=json.loads(report.report_data) if report.report_data else {},
            quality_score=report.quality_score,
            issues_found=report.issues_found,
            created_at=report.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve quality report: {str(e)}")

@router.post("/summary", response_model=AnalysisSummaryResponse)
async def get_analysis_summary(
    request: AnalysisSummaryRequest,
    db: Session = Depends(get_db)
) -> AnalysisSummaryResponse:
    """
    Generate a summary of statistical analyses within a date range
    """
    try:
        # Calculate date range
        end_date = request.end_date or datetime.utcnow()
        start_date = request.start_date or (end_date - timedelta(days=30))

        # Query analyses within date range
        analyses = db.query(StatisticalAnalysis).filter(
            StatisticalAnalysis.created_at >= start_date,
            StatisticalAnalysis.created_at <= end_date
        ).all()

        # Calculate summary statistics
        total_analyses = len(analyses)
        analysis_types = {}
        methods = {}

        for analysis in analyses:
            analysis_types[analysis.analysis_type] = analysis_types.get(analysis.analysis_type, 0) + 1
            methods[analysis.method] = methods.get(analysis.method, 0) + 1

        # Get recent quality reports
        quality_reports = db.query(DataQualityReport).filter(
            DataQualityReport.created_at >= start_date,
            DataQualityReport.created_at <= end_date
        ).all()

        avg_quality_score = np.mean([r.quality_score for r in quality_reports]) if quality_reports else 0.0

        summary = {
            "total_analyses": total_analyses,
            "analysis_types": analysis_types,
            "methods_used": methods,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "quality_reports_count": len(quality_reports),
            "average_quality_score": float(avg_quality_score)
        }

        return AnalysisSummaryResponse(
            summary=summary,
            start_date=start_date,
            end_date=end_date
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis summary: {str(e)}")

@router.post("/generate-report", response_model=ReportGenerationResponse)
async def generate_comprehensive_report(
    request: ReportGenerationRequest,
    db: Session = Depends(get_db)
) -> ReportGenerationResponse:
    """
    Generate a comprehensive statistical report for a project or dataset
    """
    try:
        # Gather all relevant analyses and reports
        analyses = db.query(StatisticalAnalysis).filter(
            StatisticalAnalysis.created_at >= request.start_date,
            StatisticalAnalysis.created_at <= request.end_date
        ).all()

        quality_reports = db.query(DataQualityReport).filter(
            DataQualityReport.created_at >= request.start_date,
            DataQualityReport.created_at <= request.end_date
        ).all()

        validation_metrics = db.query(ValidationMetric).filter(
            ValidationMetric.created_at >= request.start_date,
            ValidationMetric.created_at <= request.end_date
        ).all()

        # Compile comprehensive report
        report = {
            "report_title": request.report_title or "Statistical Analysis Report",
            "date_range": {
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat()
            },
            "summary": {
                "total_analyses": len(analyses),
                "total_quality_reports": len(quality_reports),
                "total_validation_metrics": len(validation_metrics)
            },
            "analyses_breakdown": {},
            "quality_assessment": {},
            "validation_summary": {},
            "recommendations": []
        }

        # Analysis breakdown
        for analysis in analyses:
            analysis_type = analysis.analysis_type
            if analysis_type not in report["analyses_breakdown"]:
                report["analyses_breakdown"][analysis_type] = []
            report["analyses_breakdown"][analysis_type].append({
                "id": analysis.id,
                "method": analysis.method,
                "created_at": analysis.created_at.isoformat()
            })

        # Quality assessment
        if quality_reports:
            scores = [r.quality_score for r in quality_reports]
            report["quality_assessment"] = {
                "average_score": float(np.mean(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "reports_count": len(quality_reports)
            }

        # Validation summary
        validation_types = {}
        for metric in validation_metrics:
            val_type = metric.metric_type
            validation_types[val_type] = validation_types.get(val_type, [])
            validation_types[val_type].append(metric.metric_value)

        report["validation_summary"] = {
            type_name: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
            for type_name, values in validation_types.items()
        }

        # Generate recommendations
        if report["quality_assessment"].get("average_score", 0) < 0.7:
            report["recommendations"].append("Data quality scores are below acceptable threshold. Consider data cleaning and preprocessing.")

        if len(analyses) < 10:
            report["recommendations"].append("Limited number of statistical analyses performed. Consider expanding the analysis scope.")

        if not validation_metrics:
            report["recommendations"].append("No validation metrics found. Implement proper statistical validation procedures.")

        return ReportGenerationResponse(
            report_id=f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            report_data=report,
            generated_at=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate comprehensive report: {str(e)}")

@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a statistical analysis record
    """
    try:
        analysis = get_statistical_analysis(db, analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        db.delete(analysis)
        db.commit()

        return {"message": f"Analysis {analysis_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

@router.delete("/quality-reports/{report_id}")
async def delete_quality_report(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a data quality report
    """
    try:
        report = get_data_quality_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Quality report not found")

        db.delete(report)
        db.commit()

        return {"message": f"Quality report {report_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete quality report: {str(e)}")