# AlzNexus Statistical Engine

A comprehensive statistical validation and analysis service for Alzheimer's research, providing rigorous mathematical validation for all agent-generated insights.

## Overview

The Statistical Engine is a FastAPI-based microservice that implements scientific-grade statistical analysis capabilities. It serves as the mathematical foundation for ensuring all AlzNexus research outputs meet publication-quality statistical standards.

## Features

### Core Statistical Analysis
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlation coefficients with p-values
- **Hypothesis Testing**: t-tests, z-tests, chi-square tests, ANOVA with multiple testing corrections
- **Effect Size Calculations**: Cohen's d, odds ratios, Cramér's V for practical significance
- **Power Analysis**: Sample size calculations and statistical power estimation

### Data Quality Assessment
- **Missing Data Analysis**: Detection and quantification of missing values
- **Outlier Detection**: Statistical outlier identification algorithms
- **Normality Testing**: Shapiro-Wilk and Kolmogorov-Smirnov tests
- **Distribution Analysis**: Skewness, kurtosis, and distribution statistics

### Model Validation
- **Cross-Validation**: k-fold cross-validation with comprehensive metrics
- **Performance Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Statistical Validation**: Normality, homoscedasticity, and independence tests
- **Confidence Intervals**: Bootstrap and parametric confidence intervals

### Reporting & Archival
- **Analysis Persistence**: All results stored with full metadata
- **Quality Reports**: Automated data quality assessment reports
- **Trend Analysis**: Statistical analysis summaries over time periods
- **Reproducibility**: Complete parameter and result archival

## API Endpoints

### Statistical Analysis (`/api/v1/statistical`)
- `POST /correlation` - Correlation analysis between variables
- `POST /hypothesis-test` - Various hypothesis testing methods
- `POST /effect-size` - Effect size calculations
- `POST /power-analysis` - Statistical power and sample size analysis
- `POST /data-quality-report` - Comprehensive data quality assessment

### Validation (`/api/v1/validation`)
- `POST /model-validation` - Machine learning model performance validation
- `POST /cross-validation` - k-fold cross-validation analysis
- `POST /statistical-validation` - Statistical assumption testing
- `GET /metrics` - Retrieve validation metrics history

### Reports (`/api/v1/reports`)
- `GET /analyses` - List statistical analyses with filtering
- `GET /analyses/{id}` - Get specific analysis details
- `GET /quality-reports` - List data quality reports
- `POST /summary` - Generate analysis summaries
- `POST /generate-report` - Create comprehensive statistical reports

## Architecture

### Technology Stack
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Database**: SQLAlchemy with SQLite/PostgreSQL support
- **Async Processing**: Celery with Redis for heavy computations
- **Statistical Libraries**: NumPy, SciPy, scikit-learn, statsmodels
- **API Authentication**: API key-based authentication

### Service Structure
```
alznexus_statistical_engine/
├── main.py                 # FastAPI application
├── database.py            # Database configuration
├── models.py              # SQLAlchemy models
├── schemas.py             # Pydantic schemas
├── crud.py                # Database operations
├── celery_app.py          # Celery configuration
├── tasks.py               # Asynchronous tasks
├── routers/               # API endpoints
│   ├── statistical_analysis.py
│   ├── validation.py
│   └── reports.py
└── requirements.txt       # Dependencies
```

## Integration with Existing Agents

The Statistical Engine is designed to be called by existing AlzNexus agents:

### Biomarker Hunter Agent
```python
# Validate statistical significance of biomarker correlations
response = requests.post(
    "http://localhost:8006/api/v1/statistical/correlation",
    json={"data": biomarker_data, "method": "spearman"},
    headers={"X-API-Key": api_key}
)
```

### Hypothesis Validator Agent
```python
# Perform rigorous hypothesis testing
response = requests.post(
    "http://localhost:8006/api/v1/statistical/hypothesis-test",
    json={"group1": control_data, "group2": treatment_data, "test_type": "t-test"},
    headers={"X-API-Key": api_key}
)
```

### Data Harmonizer Agent
```python
# Generate data quality reports
response = requests.post(
    "http://localhost:8006/api/v1/statistical/data-quality-report",
    json={"data": harmonized_data, "dataset_name": "processed_biomarkers"},
    headers={"X-API-Key": api_key}
)
```

## Scientific Standards

All analyses adhere to scientific best practices:

- **Statistical Significance**: p-values with appropriate alpha levels
- **Effect Sizes**: Practical significance alongside statistical significance
- **Multiple Testing**: Corrections for multiple comparisons
- **Power Analysis**: Sample size justification and power calculations
- **Data Quality**: Comprehensive validation before analysis
- **Reproducibility**: Complete parameter and result archival

## Usage Examples

### Correlation Analysis
```python
import requests

# Analyze correlation between biomarkers
biomarker_data = [
    [1.2, 2.3, 1.8, 2.1],  # Biomarker A
    [0.8, 1.9, 1.2, 1.7]   # Biomarker B
]

response = requests.post(
    "http://localhost:8006/api/v1/statistical/correlation",
    json={"data": biomarker_data, "method": "pearson"}
)
result = response.json()
print(f"Correlation: {result['correlation_matrix'][0][1]}")
print(f"P-value: {result['p_values'][0][1]}")
```

### Hypothesis Testing
```python
# Test if treatment group differs from control
control_group = [1.2, 1.5, 1.3, 1.4, 1.6]
treatment_group = [2.1, 2.3, 1.9, 2.4, 2.0]

response = requests.post(
    "http://localhost:8006/api/v1/statistical/hypothesis-test",
    json={
        "group1": control_group,
        "group2": treatment_group,
        "test_type": "t-test"
    }
)
result = response.json()
print(f"T-statistic: {result['test_statistic']}")
print(f"P-value: {result['p_value']}")
print(f"Effect size: {result['effect_size']}")
```

## Configuration

### Environment Variables
```bash
# API Configuration
STATISTICAL_API_KEY=your_api_key_here
ENV=development  # or production

# Database
DATABASE_URL=sqlite:///./statistical_engine.db
# or DATABASE_URL=postgresql://user:password@localhost/statistical_db

# Celery/Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Service
PORT=8006
```

### Running the Service
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (for Celery)
redis-server

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8006 --reload

# Or run Celery worker for async tasks
celery -A celery_app worker --loglevel=info
```

## Testing

Run the test suite to validate functionality:
```bash
python test_statistical_engine.py
```

## Scientific Validation

The Statistical Engine ensures all AlzNexus outputs meet rigorous scientific standards:

- **Statistical Rigor**: All analyses include appropriate statistical tests
- **Effect Size Reporting**: Practical significance alongside p-values
- **Power Analysis**: Sample size justification for all studies
- **Data Quality**: Comprehensive validation before analysis
- **Reproducibility**: Complete archival of methods and results
- **Multiple Testing**: Corrections for false positive inflation

This service transforms AlzNexus from a technical prototype into a scientifically trustworthy research platform capable of generating publication-quality Alzheimer's disease research insights.