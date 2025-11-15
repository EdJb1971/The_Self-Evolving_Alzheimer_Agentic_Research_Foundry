from .celery_app import celery_app
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def perform_correlation_analysis(self, data, method="pearson", confidence_level=0.95):
    """
    Asynchronous correlation analysis task
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10})

        data_array = np.array(data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        n_variables = data_array.shape[1]
        self.update_state(state='PROGRESS', meta={'progress': 30})

        if method.lower() == "pearson":
            if n_variables == 2:
                corr, p_value = stats.pearsonr(data_array[:, 0], data_array[:, 1])
                correlation_matrix = [[1.0, corr], [corr, 1.0]]
                p_matrix = [[0.0, p_value], [p_value, 0.0]]
            else:
                correlation_matrix = np.corrcoef(data_array.T)
                p_matrix = np.zeros_like(correlation_matrix)
                for i in range(n_variables):
                    for j in range(i+1, n_variables):
                        _, p_matrix[i, j] = stats.pearsonr(data_array[:, i], data_array[:, j])
                        p_matrix[j, i] = p_matrix[i, j]

        elif method.lower() == "spearman":
            if n_variables == 2:
                corr, p_value = stats.spearmanr(data_array[:, 0], data_array[:, 1])
                correlation_matrix = [[1.0, corr], [corr, 1.0]]
                p_matrix = [[0.0, p_value], [p_value, 0.0]]
            else:
                correlation_matrix = stats.spearmanr(data_array, axis=0)[0]
                p_matrix = np.zeros_like(correlation_matrix)

        self.update_state(state='PROGRESS', meta={'progress': 80})

        result = {
            "correlation_matrix": correlation_matrix.tolist() if hasattr(correlation_matrix, 'tolist') else correlation_matrix,
            "p_values": p_matrix.tolist() if hasattr(p_matrix, 'tolist') else p_matrix,
            "method": method,
            "sample_size": len(data_array),
            "confidence_level": confidence_level
        }

        self.update_state(state='PROGRESS', meta={'progress': 100})
        return result

    except Exception as e:
        logger.error(f"Correlation analysis task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def perform_hypothesis_test(self, group1, group2, test_type="t-test", alpha=0.05, equal_variance=True):
    """
    Asynchronous hypothesis testing task
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 20})

        group1_array = np.array(group1)
        group2_array = np.array(group2)

        if test_type.lower() == "t-test":
            stat, p_value = stats.ttest_ind(group1_array, group2_array, equal_var=equal_variance)
            effect_size = abs(np.mean(group1_array) - np.mean(group2_array)) / np.sqrt(
                (np.var(group1_array) + np.var(group2_array)) / 2
            )

        elif test_type.lower() == "paired-t-test":
            stat, p_value = stats.ttest_rel(group1_array, group2_array)
            effect_size = abs(np.mean(group1_array) - np.mean(group2_array)) / np.std(group1_array - group2_array)

        elif test_type.lower() == "mann-whitney":
            stat, p_value = stats.mannwhitneyu(group1_array, group2_array)
            effect_size = None  # Could calculate Cliff's delta

        self.update_state(state='PROGRESS', meta={'progress': 80})

        result = {
            "test_statistic": stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "is_significant": p_value < alpha,
            "test_type": test_type,
            "alpha": alpha,
            "sample_size_1": len(group1_array),
            "sample_size_2": len(group2_array)
        }

        self.update_state(state='PROGRESS', meta={'progress': 100})
        return result

    except Exception as e:
        logger.error(f"Hypothesis test task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def perform_cross_validation(self, X, y, model_type="linear_regression", k_folds=5, scoring="neg_mean_squared_error"):
    """
    Asynchronous cross-validation task
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10})

        X_array = np.array(X)
        y_array = np.array(y)

        # Import model based on type
        if model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "svm":
            from sklearn.svm import SVR
            model = SVR()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.update_state(state='PROGRESS', meta={'progress': 30})

        # Perform cross-validation
        scores = cross_val_score(model, X_array, y_array, cv=k_folds, scoring=scoring)

        self.update_state(state='PROGRESS', meta={'progress': 80})

        # Calculate additional metrics
        model.fit(X_array, y_array)
        y_pred = model.predict(X_array)

        mse = mean_squared_error(y_array, y_pred)
        r2 = r2_score(y_array, y_pred)

        result = {
            "cv_scores": scores.tolist(),
            "mean_cv_score": float(np.mean(scores)),
            "std_cv_score": float(np.std(scores)),
            "k_folds": k_folds,
            "scoring_metric": scoring,
            "model_type": model_type,
            "training_mse": mse,
            "training_r2": r2
        }

        self.update_state(state='PROGRESS', meta={'progress': 100})
        return result

    except Exception as e:
        logger.error(f"Cross-validation task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def generate_data_quality_report(self, data, dataset_name="unnamed_dataset"):
    """
    Asynchronous data quality report generation
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10})

        data_array = np.array(data)
        self.update_state(state='PROGRESS', meta={'progress': 30})

        # Basic statistics
        report = {
            "dataset_name": dataset_name,
            "sample_size": len(data_array),
            "dimensions": data_array.shape,
            "data_types": str(data_array.dtype),
            "missing_values": int(np.isnan(data_array).sum()),
            "infinite_values": int(np.isinf(data_array).sum())
        }

        self.update_state(state='PROGRESS', meta={'progress': 50})

        # Distribution statistics
        if data_array.size > 0:
            flattened_data = data_array.flatten()
            finite_data = flattened_data[np.isfinite(flattened_data)]

            if len(finite_data) > 0:
                report.update({
                    "mean": float(np.mean(finite_data)),
                    "median": float(np.median(finite_data)),
                    "std": float(np.std(finite_data, ddof=1)),
                    "min": float(np.min(finite_data)),
                    "max": float(np.max(finite_data)),
                    "skewness": float(stats.skew(finite_data)),
                    "kurtosis": float(stats.kurtosis(finite_data))
                })

                # Normality test
                if len(finite_data) <= 5000:
                    stat, p_value = stats.shapiro(finite_data)
                    report["normality_test"] = {
                        "test": "shapiro-wilk",
                        "statistic": stat,
                        "p_value": p_value,
                        "is_normal": p_value > 0.05
                    }
                else:
                    stat, p_value = stats.kstest(finite_data, 'norm')
                    report["normality_test"] = {
                        "test": "kolmogorov-smirnov",
                        "statistic": stat,
                        "p_value": p_value,
                        "is_normal": p_value > 0.05
                    }

        self.update_state(state='PROGRESS', meta={'progress': 80})

        # Quality score calculation
        quality_score = 1.0
        issues = []

        if report.get("missing_values", 0) > 0:
            quality_score -= 0.2
            issues.append(f"Found {report['missing_values']} missing values")

        if report.get("infinite_values", 0) > 0:
            quality_score -= 0.1
            issues.append(f"Found {report['infinite_values']} infinite values")

        if report.get("normality_test", {}).get("is_normal") == False:
            quality_score -= 0.1
            issues.append("Data may not be normally distributed")

        # Outlier detection (IQR method)
        if "median" in report:
            q1 = np.percentile(finite_data, 25)
            q3 = np.percentile(finite_data, 75)
            iqr = q3 - q1
            outliers = ((finite_data < (q1 - 1.5 * iqr)) | (finite_data > (q3 + 1.5 * iqr))).sum()
            report["outliers"] = int(outliers)

            if outliers > len(finite_data) * 0.05:
                quality_score -= 0.1
                issues.append(f"High number of outliers detected: {outliers}")

        report["quality_score"] = max(0.0, quality_score)
        report["issues_found"] = issues

        self.update_state(state='PROGRESS', meta={'progress': 100})
        return report

    except Exception as e:
        logger.error(f"Data quality report generation failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def perform_power_analysis(self, effect_size, sample_size=None, power=None, alpha=0.05, test_type="t-test"):
    """
    Asynchronous power analysis task
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 20})

        from statsmodels.stats.power import TTestIndPower, TTestPower

        if test_type.lower() == "t-test":
            power_analysis = TTestIndPower()

            if sample_size and not power:
                # Calculate power
                power_result = power_analysis.power(
                    effect_size=effect_size,
                    nobs1=sample_size,
                    alpha=alpha
                )
                result = {"power": power_result}

            elif power and not sample_size:
                # Calculate sample size
                sample_size_result = power_analysis.solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha
                )
                result = {"sample_size": sample_size_result}

            elif effect_size and power and not sample_size:
                # Calculate sample size
                sample_size_result = power_analysis.solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha
                )
                result = {"sample_size": sample_size_result}

            else:
                raise ValueError("Invalid parameter combination for power analysis")

        self.update_state(state='PROGRESS', meta={'progress': 80})

        result.update({
            "effect_size": effect_size,
            "alpha": alpha,
            "test_type": test_type
        })

        self.update_state(state='PROGRESS', meta={'progress': 100})
        return result

    except Exception as e:
        logger.error(f"Power analysis task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise