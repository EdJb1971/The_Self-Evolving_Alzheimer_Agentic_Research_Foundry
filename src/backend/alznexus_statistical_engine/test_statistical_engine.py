#!/usr/bin/env python3
"""
Test script for the AlzNexus Statistical Engine
Tests basic functionality of the statistical analysis endpoints
"""

import numpy as np
import requests
import json
import time
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_correlation_analysis():
    """Test correlation analysis endpoint"""
    print("Testing correlation analysis...")

    # Generate sample data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 0.5 * x + np.random.normal(0, 0.5, 100)
    z = -0.3 * x + np.random.normal(0, 0.5, 100)

    data = [x.tolist(), y.tolist(), z.tolist()]

    payload = {
        "data": data,
        "method": "pearson",
        "confidence_level": 0.95
    }

    try:
        # For testing, we'll simulate the API call by importing and calling directly
        from routers.statistical_analysis import perform_correlation
        from unittest.mock import Mock
        from sqlalchemy.orm import Session

        # Mock the database session
        mock_db = Mock(spec=Session)

        # Mock the create_statistical_analysis function
        import routers.statistical_analysis as sa
        sa.create_statistical_analysis = Mock(return_value=Mock(id=1))

        result = perform_correlation(payload, mock_db)
        print(f"✓ Correlation analysis successful: {result['analysis_id']}")
        return True

    except Exception as e:
        print(f"✗ Correlation analysis failed: {e}")
        return False

def test_hypothesis_testing():
    """Test hypothesis testing endpoint"""
    print("Testing hypothesis testing...")

    # Generate sample data for t-test
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(12, 2, 50)  # Different mean

    payload = {
        "group1": group1.tolist(),
        "group2": group2.tolist(),
        "test_type": "t-test",
        "equal_variance": True,
        "alpha": 0.05
    }

    try:
        # Simulate the API call
        from routers.statistical_analysis import perform_hypothesis_test
        from unittest.mock import Mock
        from sqlalchemy.orm import Session

        mock_db = Mock(spec=Session)
        import routers.statistical_analysis as sa
        sa.create_statistical_analysis = Mock(return_value=Mock(id=2))

        result = perform_hypothesis_test(payload, mock_db)
        print(f"✓ Hypothesis testing successful: {result['test_type']} - p-value: {result['p_value']:.4f}")
        return True

    except Exception as e:
        print(f"✗ Hypothesis testing failed: {e}")
        return False

def test_data_quality_report():
    """Test data quality report generation"""
    print("Testing data quality report...")

    # Generate sample data with some issues
    np.random.seed(42)
    data = np.random.normal(0, 1, 100).tolist()
    data[10] = None  # Add missing value
    data.extend([10, -10, 15])  # Add outliers

    payload = {
        "data": data,
        "dataset_name": "test_dataset"
    }

    try:
        # Simulate the API call
        from routers.statistical_analysis import generate_data_quality_report
        from unittest.mock import Mock
        from sqlalchemy.orm import Session

        mock_db = Mock(spec=Session)
        import routers.statistical_analysis as sa
        sa.create_data_quality_report = Mock(return_value=Mock(id=3, quality_score=0.85, issues_found=["Missing values detected"], created_at=None))

        result = generate_data_quality_report(payload, mock_db)
        print(f"✓ Data quality report successful: score = {result['quality_score']}")
        return True

    except Exception as e:
        print(f"✗ Data quality report failed: {e}")
        return False

def test_effect_size_calculation():
    """Test effect size calculation"""
    print("Testing effect size calculation...")

    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(12, 2, 50)

    payload = {
        "group1": group1.tolist(),
        "group2": group2.tolist(),
        "effect_type": "cohens_d",
        "confidence_level": 0.95
    }

    try:
        # Simulate the API call
        from routers.statistical_analysis import calculate_effect_size
        from unittest.mock import Mock
        from sqlalchemy.orm import Session

        mock_db = Mock(spec=Session)
        import routers.statistical_analysis as sa
        sa.create_statistical_analysis = Mock(return_value=Mock(id=4))

        result = calculate_effect_size(payload, mock_db)
        print(f"✓ Effect size calculation successful: Cohen's d = {result['effect_size']:.3f}")
        return True

    except Exception as e:
        print(f"✗ Effect size calculation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("AlzNexus Statistical Engine - Test Suite")
    print("=" * 50)

    tests = [
        test_correlation_analysis,
        test_hypothesis_testing,
        test_data_quality_report,
        test_effect_size_calculation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Statistical Engine is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())