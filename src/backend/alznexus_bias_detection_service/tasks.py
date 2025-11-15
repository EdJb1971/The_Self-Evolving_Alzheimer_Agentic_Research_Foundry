import os
import logging
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import statsmodels.api as sm
from causalinference import CausalModel
import scipy.stats as stats
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas

AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")

logger = logging.getLogger(__name__)

def perform_comprehensive_bias_analysis(data_to_analyze: str, analysis_context: dict) -> Dict[str, Any]:
    """
    Perform comprehensive bias detection using statistical fairness metrics and causal inference.

    Args:
        data_to_analyze: The data to analyze for bias
        analysis_context: Additional context for the analysis

    Returns:
        Dictionary containing bias analysis results
    """
    try:
        # Parse data into structured format
        data_df = parse_data_for_analysis(data_to_analyze, analysis_context)

        if data_df is None or data_df.empty:
            return {
                "detected_bias": False,
                "bias_type": "none",
                "severity": "none",
                "analysis_summary": "Unable to parse data for bias analysis. Data may be in unsupported format.",
                "proposed_corrections": []
            }

        # Perform multiple bias detection analyses
        bias_results = []

        # 1. Statistical fairness analysis
        fairness_result = analyze_statistical_fairness(data_df)
        bias_results.append(fairness_result)

        # 2. Causal inference analysis
        causal_result = analyze_causal_inference(data_df)
        bias_results.append(causal_result)

        # 3. Data quality and representation analysis
        quality_result = analyze_data_quality(data_df)
        bias_results.append(quality_result)

        # 4. Demographic representation analysis
        demo_result = analyze_demographic_representation(data_df)
        bias_results.append(demo_result)

        # Aggregate results
        aggregated_result = aggregate_bias_results(bias_results)

        return aggregated_result

    except Exception as e:
        logger.error(f"Error in comprehensive bias analysis: {str(e)}", exc_info=True)
        return {
            "detected_bias": False,
            "bias_type": "analysis_error",
            "severity": "unknown",
            "analysis_summary": f"Bias analysis failed due to error: {str(e)}",
            "proposed_corrections": ["Review data format and analysis methodology"]
        }

def parse_data_for_analysis(data_to_analyze: str, analysis_context: dict) -> Optional[pd.DataFrame]:
    """
    Parse the input data into a structured DataFrame for analysis.
    """
    try:
        # Try to parse as JSON first
        if data_to_analyze.strip().startswith('{') or data_to_analyze.strip().startswith('['):
            data = json.loads(data_to_analyze)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
        else:
            # Try to parse as CSV-like data
            lines = data_to_analyze.strip().split('\n')
            if len(lines) > 1:
                # Assume first line is header
                headers = [h.strip() for h in lines[0].split(',')]
                rows = []
                for line in lines[1:]:
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(headers):
                        rows.append(dict(zip(headers, values)))
                if rows:
                    return pd.DataFrame(rows)

        # If parsing fails, create a simple text analysis DataFrame
        return pd.DataFrame({
            'text_content': [data_to_analyze],
            'analysis_type': [analysis_context.get('analysis_type', 'general')],
            'entity_type': [analysis_context.get('entity_type', 'unknown')]
        })

    except Exception as e:
        logger.warning(f"Failed to parse data for analysis: {str(e)}")
        return None

def analyze_statistical_fairness(data_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze statistical fairness using Fairlearn metrics.
    """
    result = {
        "bias_type": "statistical_fairness",
        "detected_bias": False,
        "severity": "none",
        "analysis_summary": "",
        "proposed_corrections": []
    }

    try:
        # Check if we have the necessary columns for fairness analysis
        required_cols = ['outcome', 'sensitive_attribute']
        if not all(col in data_df.columns for col in required_cols):
            result["analysis_summary"] = "Insufficient data for statistical fairness analysis. Missing outcome or sensitive attribute columns."
            return result

        # Prepare data
        y_true = data_df['outcome'].astype(int)
        sensitive_features = data_df['sensitive_attribute']

        # Calculate fairness metrics
        dp_diff = demographic_parity_difference(y_true, y_true, sensitive_features=sensitive_features)
        eo_diff = equalized_odds_difference(y_true, y_true, sensitive_features=sensitive_features)

        # Determine if bias is detected
        fairness_threshold = 0.1  # 10% difference threshold
        max_diff = max(dp_diff, eo_diff)

        if max_diff > fairness_threshold:
            result["detected_bias"] = True
            result["severity"] = "high" if max_diff > 0.2 else "medium" if max_diff > 0.15 else "low"
            result["analysis_summary"] = f"Statistical fairness violation detected. Max difference: {max_diff:.3f}"
        else:
            result["analysis_summary"] = f"No significant statistical fairness violations. Max difference: {max_diff:.3f}"
            result["proposed_corrections"] = [
                "Implement fairness-aware algorithms",
                "Use demographic parity constraints in model training",
                "Apply post-processing fairness techniques"
            ]

    except Exception as e:
        logger.warning(f"Statistical fairness analysis failed: {str(e)}")
        result["analysis_summary"] = f"Statistical fairness analysis failed: {str(e)}"

    return result

def analyze_causal_inference(data_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze potential causal biases using causal inference methods.
    """
    result = {
        "bias_type": "causal_inference",
        "detected_bias": False,
        "severity": "none",
        "analysis_summary": "",
        "proposed_corrections": []
    }

    try:
        # Check for causal analysis requirements
        causal_cols = ['treatment', 'outcome', 'confounder']
        if not all(col in data_df.columns for col in causal_cols):
            result["analysis_summary"] = "Insufficient data for causal inference analysis."
            return result

        # Prepare data for causal analysis
        treatment = data_df['treatment'].astype(int).values
        outcome = data_df['outcome'].astype(float).values
        confounder = data_df['confounder'].astype(float).values

        # Perform causal inference
        causal = CausalModel(Y=outcome, D=treatment, X=confounder)
        causal.est_via_ols()
        causal.est_via_matching()

        # Check for significant causal effects
        ate = causal.estimates['ols']['ate']
        ate_se = causal.estimates['ols']['ate_se']

        # Calculate statistical significance
        t_stat = abs(ate) / ate_se if ate_se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(t_stat))

        if p_value < 0.05:  # Significant causal effect
            result["detected_bias"] = True
            result["severity"] = "high" if abs(ate) > 0.5 else "medium" if abs(ate) > 0.2 else "low"
            result["analysis_summary"] = f"Significant causal bias detected. ATE: {ate:.3f}, p-value: {p_value:.3f}"
            result["proposed_corrections"] = [
                "Control for confounding variables in analysis",
                "Use propensity score matching",
                "Implement randomized controlled trials where possible"
            ]
        else:
            result["analysis_summary"] = f"No significant causal bias detected. ATE: {ate:.3f}, p-value: {p_value:.3f}"
            result["proposed_corrections"] = ["Continue monitoring for potential confounding effects"]

    except Exception as e:
        logger.warning(f"Causal inference analysis failed: {str(e)}")
        result["analysis_summary"] = f"Causal inference analysis failed: {str(e)}"

    return result

def analyze_data_quality(data_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data quality issues that could indicate bias.
    """
    result = {
        "bias_type": "data_quality",
        "detected_bias": False,
        "severity": "none",
        "analysis_summary": "",
        "proposed_corrections": []
    }

    issues = []
    corrections = []

    # Check for missing data
    missing_pct = data_df.isnull().mean().mean()
    if missing_pct > 0.1:  # More than 10% missing
        issues.append(".1%")
        corrections.append("Implement proper data imputation strategies")
        result["detected_bias"] = True

    # Check for class imbalance
    if 'outcome' in data_df.columns:
        class_counts = data_df['outcome'].value_counts()
        if len(class_counts) > 1:
            min_class_pct = class_counts.min() / class_counts.sum()
            if min_class_pct < 0.1:  # Less than 10% minority class
                issues.append(".1%")
                corrections.append("Use oversampling, undersampling, or synthetic data generation")
                result["detected_bias"] = True

    # Check for duplicate data
    duplicate_pct = data_df.duplicated().mean()
    if duplicate_pct > 0.05:  # More than 5% duplicates
        issues.append(".1%")
        corrections.append("Remove duplicate entries and ensure data uniqueness")
        result["detected_bias"] = True

    if issues:
        result["severity"] = "high" if len(issues) > 2 else "medium" if len(issues) > 1 else "low"
        result["analysis_summary"] = "Data quality issues detected: " + "; ".join(issues)
        result["proposed_corrections"] = corrections
    else:
        result["analysis_summary"] = "No significant data quality issues detected."

    return result

def analyze_demographic_representation(data_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze demographic representation in the dataset.
    """
    result = {
        "bias_type": "demographic_representation",
        "detected_bias": False,
        "severity": "none",
        "analysis_summary": "",
        "proposed_corrections": []
    }

    try:
        # Look for demographic columns
        demographic_cols = ['age', 'gender', 'race', 'ethnicity', 'socioeconomic_status']
        found_demo_cols = [col for col in demographic_cols if col in data_df.columns]

        if not found_demo_cols:
            result["analysis_summary"] = "No demographic variables found for representation analysis."
            return result

        issues = []
        corrections = []

        for col in found_demo_cols:
            value_counts = data_df[col].value_counts(normalize=True)
            # Check for underrepresented groups (< 5% representation)
            underrepresented = value_counts[value_counts < 0.05]
            if len(underrepresented) > 0:
                issues.append(f"Underrepresented groups in {col}: {', '.join(underrepresented.index.tolist())}")
                corrections.append(f"Increase representation of underrepresented {col} groups")
                result["detected_bias"] = True

            # Check for overrepresented groups (> 70% representation)
            overrepresented = value_counts[value_counts > 0.7]
            if len(overrepresented) > 0:
                issues.append(f"Overrepresented groups in {col}: {', '.join(overrepresented.index.tolist())}")
                corrections.append(f"Balance representation across {col} groups")
                result["detected_bias"] = True

        if issues:
            result["severity"] = "high" if len(issues) > 3 else "medium" if len(issues) > 1 else "low"
            result["analysis_summary"] = "Demographic representation issues: " + "; ".join(issues)
            result["proposed_corrections"] = corrections
        else:
            result["analysis_summary"] = "Demographic representation appears balanced."

    except Exception as e:
        logger.warning(f"Demographic representation analysis failed: {str(e)}")
        result["analysis_summary"] = f"Demographic representation analysis failed: {str(e)}"

    return result

def aggregate_bias_results(bias_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple bias analysis methods.
    """
    # Check if any analysis detected bias
    detected_biases = [r for r in bias_results if r.get("detected_bias", False)]

    if not detected_biases:
        return {
            "detected_bias": False,
            "bias_type": "none",
            "severity": "none",
            "analysis_summary": "No bias detected across all analysis methods.",
            "proposed_corrections": []
        }

    # Determine overall bias type and severity
    bias_types = [r["bias_type"] for r in detected_biases]
    severities = [r["severity"] for r in detected_biases]

    # Use the most severe bias type detected
    severity_hierarchy = {"low": 1, "medium": 2, "high": 3, "unknown": 0}
    max_severity = max(severities, key=lambda x: severity_hierarchy.get(x, 0))

    # Combine analysis summaries
    summaries = [r["analysis_summary"] for r in detected_biases if r["analysis_summary"]]
    combined_summary = "Multiple bias analyses completed. " + " ".join(summaries)

    # Combine proposed corrections
    all_corrections = []
    for r in detected_biases:
        all_corrections.extend(r.get("proposed_corrections", []))
    unique_corrections = list(set(all_corrections))

    return {
        "detected_bias": True,
        "bias_type": ", ".join(bias_types),
        "severity": max_severity,
        "analysis_summary": combined_summary,
        "proposed_corrections": unique_corrections
    }

def log_audit_event(entity_type: str, entity_id: str, event_type: str, description: str, metadata: dict = None):
    headers = {"X-API-Key": AUDIT_API_KEY, "Content-Type": "application/json"}
    payload = {
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "event_type": event_type,
        "description": description,
        "metadata": metadata if metadata is not None else {}
    }
    try:
        response = requests.post(f"{AUDIT_TRAIL_URL}/audit/log", headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Audit log successful: {event_type} for {entity_type}:{entity_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to log audit event: {str(e)}", exc_info=True)

@celery_app.task(bind=True, name="detect_bias_task")
def detect_bias_task(self, report_id: int, entity_type: str, entity_id: str, data_to_analyze: str, analysis_context: dict):
    db: Session = SessionLocal()
    try:
        # CQ-BIAS-001: Fetch and update the existing report, instead of creating a new one
        existing_report = crud.get_bias_detection_report(db, report_id)
        if not existing_report:
            raise ValueError(f"BiasDetectionReport with ID {report_id} not found.")

        # Update report status to IN_PROGRESS
        crud.update_bias_detection_report(db, report_id, schemas.BiasDetectionReportCreate(
            entity_type=entity_type, entity_id=entity_id, data_snapshot=data_to_analyze,
            detected_bias=False, analysis_summary="Analysis in progress.", metadata_json=analysis_context
        ))
        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(report_id),
            event_type="BIAS_DETECTION_STARTED",
            description=f"Bias detection started for {entity_type}:{entity_id}.",
            metadata={"report_id": report_id, "entity_type": entity_type, "entity_id": entity_id}
        )

        # Perform comprehensive bias detection using statistical methods
        bias_analysis_result = perform_comprehensive_bias_analysis(data_to_analyze, analysis_context)

        detected_bias = bias_analysis_result["detected_bias"]
        bias_type = bias_analysis_result["bias_type"]
        severity = bias_analysis_result["severity"]
        analysis_summary = bias_analysis_result["analysis_summary"]
        proposed_corrections = bias_analysis_result["proposed_corrections"]

        report_update_data = schemas.BiasDetectionReportCreate(
            entity_type=entity_type,
            entity_id=entity_id,
            data_snapshot=data_to_analyze,
            detected_bias=detected_bias,
            bias_type=bias_type,
            severity=severity,
            analysis_summary=analysis_summary,
            proposed_corrections={"actions": proposed_corrections},
            metadata_json=analysis_context
        )
        db_report = crud.update_bias_detection_report(db, report_id, report_update_data)
        if not db_report:
            raise ValueError(f"Failed to update BiasDetectionReport with ID {report_id}.")

        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(db_report.id),
            event_type="BIAS_DETECTED" if detected_bias else "NO_BIAS_DETECTED",
            description=f"Bias detection completed for {entity_type}:{entity_id}. Bias detected: {detected_bias}.",
            metadata=db_report.model_dump()
        )

        return {"report_id": db_report.id, "status": "COMPLETED", "detected_bias": detected_bias, "bias_type": bias_type, "severity": severity, "analysis_summary": analysis_summary, "proposed_corrections": proposed_corrections}
    except Exception as e:
        error_message = str(e)
        # CQ-BIAS-001: Update existing report with failure status
        crud.update_bias_detection_report(db, report_id, schemas.BiasDetectionReportCreate(
            entity_type=entity_type, entity_id=entity_id, data_snapshot=data_to_analyze,
            detected_bias=False, analysis_summary=f"Failed to perform bias detection: {error_message}",
            metadata_json=analysis_context
        ))
        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(report_id),
            event_type="BIAS_DETECTION_FAILED",
            description=f"Bias detection failed for {entity_type}:{entity_id}: {error_message}",
            metadata={"report_id": report_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
