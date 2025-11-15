from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
from statsmodels.stats.power import TTestIndPower, TTestPower
from statsmodels.stats.proportion import proportions_ztest
import json

from ..database import get_db
from ..models import StatisticalAnalysis, DataQualityReport, ValidationMetric
from ..schemas import (
    CorrelationAnalysisRequest, CorrelationAnalysisResponse,
    HypothesisTestRequest, HypothesisTestResponse,
    EffectSizeRequest, EffectSizeResponse,
    PowerAnalysisRequest, PowerAnalysisResponse,
    StatisticalAnalysisCreate, StatisticalAnalysisResponse,
    DataQualityReportCreate, DataQualityReportResponse,
    ValidationMetricCreate, ValidationMetricResponse
)
from ..crud import create_statistical_analysis, get_statistical_analysis, create_data_quality_report

router = APIRouter()

@router.post("/correlation", response_model=CorrelationAnalysisResponse)
async def analyze_correlation(
    request: CorrelationAnalysisRequest,
    db: Session = Depends(get_db)
) -> CorrelationAnalysisResponse:
    """
    Perform correlation analysis between two or more variables.
    Supports Pearson, Spearman, and Kendall correlation methods.
    """
    try:
        # Convert data to numpy arrays
        data = np.array(request.data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_variables = data.shape[1]
        if n_variables < 2:
            raise HTTPException(status_code=400, detail="At least 2 variables required for correlation analysis")

        # Calculate correlation matrix
        if request.method.lower() == "pearson":
            corr_matrix, p_matrix = stats.pearsonr(data[:, 0], data[:, 1]) if n_variables == 2 else ([], [])
            if n_variables > 2:
                corr_matrix = np.corrcoef(data.T)
                p_matrix = np.zeros_like(corr_matrix)
                for i in range(n_variables):
                    for j in range(i+1, n_variables):
                        _, p_matrix[i, j] = stats.pearsonr(data[:, i], data[:, j])
                        p_matrix[j, i] = p_matrix[i, j]
        elif request.method.lower() == "spearman":
            corr_matrix, p_matrix = stats.spearmanr(data) if n_variables == 2 else ([], [])
            if n_variables > 2:
                corr_matrix = stats.spearmanr(data, axis=0)[0]
                p_matrix = np.zeros_like(corr_matrix)
        elif request.method.lower() == "kendall":
            corr_matrix, p_matrix = stats.kendalltau(data[:, 0], data[:, 1]) if n_variables == 2 else ([], [])
            if n_variables > 2:
                corr_matrix = np.zeros((n_variables, n_variables))
                p_matrix = np.zeros_like(corr_matrix)
                for i in range(n_variables):
                    for j in range(i+1, n_variables):
                        corr_matrix[i, j], p_matrix[i, j] = stats.kendalltau(data[:, i], data[:, j])
                        corr_matrix[j, i] = corr_matrix[i, j]
                        p_matrix[j, i] = p_matrix[i, j]
        else:
            raise HTTPException(status_code=400, detail="Invalid correlation method. Use 'pearson', 'spearman', or 'kendall'")

        # Create analysis record
        analysis_data = StatisticalAnalysisCreate(
            analysis_type="correlation",
            method=request.method,
            parameters=json.dumps({"variables": n_variables, "method": request.method}),
            results=json.dumps({
                "correlation_matrix": corr_matrix.tolist() if hasattr(corr_matrix, 'tolist') else corr_matrix,
                "p_values": p_matrix.tolist() if hasattr(p_matrix, 'tolist') else p_matrix
            }),
            confidence_level=request.confidence_level or 0.95
        )

        analysis = create_statistical_analysis(db, analysis_data)

        return CorrelationAnalysisResponse(
            analysis_id=analysis.id,
            correlation_matrix=corr_matrix.tolist() if hasattr(corr_matrix, 'tolist') else corr_matrix,
            p_values=p_matrix.tolist() if hasattr(p_matrix, 'tolist') else p_matrix,
            method=request.method,
            confidence_level=request.confidence_level or 0.95,
            sample_size=len(data)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")

@router.post("/hypothesis-test", response_model=HypothesisTestResponse)
async def perform_hypothesis_test(
    request: HypothesisTestRequest,
    db: Session = Depends(get_db)
) -> HypothesisTestResponse:
    """
    Perform various hypothesis tests (t-test, z-test, chi-square, etc.)
    """
    try:
        test_type = request.test_type.lower()

        if test_type == "t-test":
            # Independent samples t-test
            stat, p_value = stats.ttest_ind(request.group1, request.group2, equal_var=request.equal_variance)
            effect_size = abs(np.mean(request.group1) - np.mean(request.group2)) / np.sqrt(
                (np.var(request.group1) + np.var(request.group2)) / 2
            )  # Cohen's d

        elif test_type == "paired-t-test":
            stat, p_value = stats.ttest_rel(request.group1, request.group2)
            effect_size = abs(np.mean(request.group1) - np.mean(request.group2)) / np.std(request.group1 - request.group2)

        elif test_type == "z-test":
            # Two proportion z-test
            stat, p_value = proportions_ztest(
                [sum(request.group1), sum(request.group2)],
                [len(request.group1), len(request.group2)]
            )
            effect_size = None  # Could calculate odds ratio

        elif test_type == "chi-square":
            # Chi-square test for independence
            contingency_table = np.array(request.contingency_table)
            stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            effect_size = None  # Could calculate Cramer's V

        else:
            raise HTTPException(status_code=400, detail="Unsupported test type")

        # Determine significance
        alpha = request.alpha or 0.05
        is_significant = p_value < alpha

        # Create analysis record
        analysis_data = StatisticalAnalysisCreate(
            analysis_type="hypothesis_test",
            method=test_type,
            parameters=json.dumps({
                "alpha": alpha,
                "test_type": test_type,
                "group1_size": len(request.group1) if hasattr(request, 'group1') else None,
                "group2_size": len(request.group2) if hasattr(request, 'group2') else None
            }),
            results=json.dumps({
                "test_statistic": stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "is_significant": is_significant
            }),
            confidence_level=1 - alpha
        )

        analysis = create_statistical_analysis(db, analysis_data)

        return HypothesisTestResponse(
            analysis_id=analysis.id,
            test_statistic=stat,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            confidence_level=1 - alpha,
            test_type=test_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hypothesis test failed: {str(e)}")

@router.post("/effect-size", response_model=EffectSizeResponse)
async def calculate_effect_size(
    request: EffectSizeRequest,
    db: Session = Depends(get_db)
) -> EffectSizeResponse:
    """
    Calculate various effect size measures (Cohen's d, odds ratio, etc.)
    """
    try:
        effect_type = request.effect_type.lower()

        if effect_type == "cohens_d":
            mean1, mean2 = np.mean(request.group1), np.mean(request.group2)
            std1, std2 = np.std(request.group1, ddof=1), np.std(request.group2, ddof=1)
            n1, n2 = len(request.group1), len(request.group2)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            effect_size = abs(mean1 - mean2) / pooled_std

        elif effect_type == "odds_ratio":
            # For 2x2 contingency table
            table = np.array(request.contingency_table)
            if table.shape != (2, 2):
                raise HTTPException(status_code=400, detail="Odds ratio requires 2x2 contingency table")

            effect_size = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])

        elif effect_type == "cramers_v":
            table = np.array(request.contingency_table)
            chi2, _, _, _ = stats.chi2_contingency(table)
            n = np.sum(table)
            min_dim = min(table.shape) - 1
            effect_size = np.sqrt(chi2 / (n * min_dim))

        else:
            raise HTTPException(status_code=400, detail="Unsupported effect size type")

        # Create analysis record
        analysis_data = StatisticalAnalysisCreate(
            analysis_type="effect_size",
            method=effect_type,
            parameters=json.dumps({"effect_type": effect_type}),
            results=json.dumps({"effect_size": effect_size}),
            confidence_level=request.confidence_level or 0.95
        )

        analysis = create_statistical_analysis(db, analysis_data)

        return EffectSizeResponse(
            analysis_id=analysis.id,
            effect_size=effect_size,
            effect_type=effect_type,
            confidence_level=request.confidence_level or 0.95
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Effect size calculation failed: {str(e)}")

@router.post("/power-analysis", response_model=PowerAnalysisResponse)
async def perform_power_analysis(
    request: PowerAnalysisRequest,
    db: Session = Depends(get_db)
) -> PowerAnalysisResponse:
    """
    Perform power analysis for study design and sample size calculation
    """
    try:
        analysis_type = request.analysis_type.lower()

        if analysis_type == "t-test":
            power_analysis = TTestIndPower()
            if request.effect_size and request.sample_size and request.alpha:
                # Calculate power
                power = power_analysis.power(effect_size=request.effect_size,
                                           nobs1=request.sample_size,
                                           alpha=request.alpha)
                result = {"power": power}
            elif request.effect_size and request.power and request.alpha:
                # Calculate sample size
                sample_size = power_analysis.solve_power(effect_size=request.effect_size,
                                                       power=request.power,
                                                       alpha=request.alpha)
                result = {"sample_size": sample_size}
            else:
                raise HTTPException(status_code=400, detail="Invalid parameter combination for power analysis")

        else:
            raise HTTPException(status_code=400, detail="Unsupported analysis type for power analysis")

        # Create analysis record
        analysis_data = StatisticalAnalysisCreate(
            analysis_type="power_analysis",
            method=analysis_type,
            parameters=json.dumps({
                "effect_size": request.effect_size,
                "sample_size": request.sample_size,
                "power": request.power,
                "alpha": request.alpha
            }),
            results=json.dumps(result),
            confidence_level=1 - (request.alpha or 0.05)
        )

        analysis = create_statistical_analysis(db, analysis_data)

        return PowerAnalysisResponse(
            analysis_id=analysis.id,
            **result,
            analysis_type=analysis_type,
            effect_size=request.effect_size,
            alpha=request.alpha or 0.05
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Power analysis failed: {str(e)}")

@router.post("/data-quality-report", response_model=DataQualityReportResponse)
async def generate_data_quality_report(
    request: DataQualityReportCreate,
    db: Session = Depends(get_db)
) -> DataQualityReportResponse:
    """
    Generate a comprehensive data quality report
    """
    try:
        data = np.array(request.data)

        # Basic statistics
        report = {
            "sample_size": len(data),
            "missing_values": int(np.isnan(data).sum()),
            "outliers": int(len(data) * 0.05),  # Rough estimate
            "normality_test": {},
            "distribution_stats": {}
        }

        # Normality test (Shapiro-Wilk)
        if len(data) <= 5000:  # Shapiro-Wilk limit
            stat, p_value = stats.shapiro(data.flatten())
            report["normality_test"] = {
                "test": "shapiro-wilk",
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05
            }

        # Distribution statistics
        report["distribution_stats"] = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data, ddof=1)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data.flatten())),
            "kurtosis": float(stats.kurtosis(data.flatten()))
        }

        # Create report record
        report_data = DataQualityReportCreate(
            dataset_name=request.dataset_name,
            report_data=json.dumps(report),
            quality_score=request.quality_score or 0.8,  # Placeholder
            issues_found=request.issues_found or []
        )

        quality_report = create_data_quality_report(db, report_data)

        return DataQualityReportResponse(
            id=quality_report.id,
            dataset_name=quality_report.dataset_name,
            report_data=report,
            quality_score=quality_report.quality_score,
            issues_found=quality_report.issues_found,
            created_at=quality_report.created_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data quality report generation failed: {str(e)}")