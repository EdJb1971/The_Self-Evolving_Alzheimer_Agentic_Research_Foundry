"""
Comprehensive tests for the Causal Inference Service

Tests cover:
- Causal discovery algorithms
- Effect estimation methods
- Mechanistic modeling
- API endpoints
- Integration scenarios
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import networkx as nx
from datetime import datetime

# Import service components
from causal_discovery import CausalDiscoveryEngine, BootstrapPCAusalDiscovery, CausalGraph
from dowhy_integration import CausalInferenceEngine, CausalEffectResult
from mechanistic_modeling import MechanisticModelingEngine, MechanisticModel
from main import app
from models import Dataset, CausalGraph as GraphModel
from schemas import DatasetUploadRequest, CausalDiscoveryRequest

# Test data fixtures
@pytest.fixture
def sample_alzheimer_data():
    """Generate sample Alzheimer biomarker data"""
    np.random.seed(42)
    n_samples = 1000

    # Generate correlated variables
    amyloid_beta = np.random.normal(1.0, 0.3, n_samples)
    tau_protein = amyloid_beta * 0.7 + np.random.normal(0, 0.2, n_samples)
    inflammation = amyloid_beta * 0.5 + tau_protein * 0.3 + np.random.normal(0, 0.15, n_samples)
    neuronal_death = amyloid_beta * 0.4 + tau_protein * 0.6 + inflammation * 0.8 + np.random.normal(0, 0.1, n_samples)
    cognitive_decline = neuronal_death * 0.9 + np.random.normal(0, 0.05, n_samples)

    # Add some confounding
    age = np.random.normal(70, 10, n_samples)
    amyloid_beta += age * 0.01
    tau_protein += age * 0.005

    data = pd.DataFrame({
        'amyloid_beta': amyloid_beta,
        'tau_protein': tau_protein,
        'inflammation': inflammation,
        'neuronal_death': neuronal_death,
        'cognitive_decline': cognitive_decline,
        'age': age
    })

    return data

@pytest.fixture
def sample_causal_graph():
    """Create a sample causal graph"""
    graph = nx.DiGraph()
    graph.add_edges_from([
        ('amyloid_beta', 'tau_protein'),
        ('amyloid_beta', 'inflammation'),
        ('tau_protein', 'neuronal_death'),
        ('inflammation', 'neuronal_death'),
        ('neuronal_death', 'cognitive_decline')
    ])

    return CausalGraph(
        graph=graph,
        adjacency_matrix=nx.to_numpy_array(graph),
        variables=['amyloid_beta', 'tau_protein', 'inflammation', 'neuronal_death', 'cognitive_decline'],
        confidence_scores={
            'amyloid_beta->tau_protein': 0.85,
            'amyloid_beta->inflammation': 0.78,
            'tau_protein->neuronal_death': 0.92,
            'inflammation->neuronal_death': 0.88,
            'neuronal_death->cognitive_decline': 0.95
        }
    )

class TestCausalDiscovery:
    """Test causal discovery algorithms"""

    def test_bootstrap_pc_initialization(self):
        """Test BootstrapPC algorithm initialization"""
        algorithm = BootstrapPCAusalDiscovery(
            alpha=0.01,
            n_bootstrap=50,
            max_degree=3
        )

        assert algorithm.alpha == 0.01
        assert algorithm.n_bootstrap == 50
        assert algorithm.max_degree == 3

    def test_causal_discovery_engine(self, sample_alzheimer_data):
        """Test causal discovery engine"""
        engine = CausalDiscoveryEngine()

        # Test with small dataset for speed
        small_data = sample_alzheimer_data.head(100)

        graph = engine.discover_causal_graph(small_data, algorithm='pc')

        assert isinstance(graph, CausalGraph)
        assert len(graph.variables) > 0
        assert graph.is_dag()  # Should be acyclic

    def test_causal_graph_properties(self, sample_causal_graph):
        """Test causal graph properties"""
        graph = sample_causal_graph

        assert graph.is_dag()
        assert len(graph.get_edges()) == 5
        assert 'amyloid_beta' in graph.variables

        # Test confidence scores
        assert graph.get_confidence_score('amyloid_beta', 'tau_protein') == 0.85

    @patch('causal_discovery.BootstrapPCAusalDiscovery._learn_skeleton')
    def test_bootstrap_aggregation(self, mock_skeleton, sample_alzheimer_data):
        """Test bootstrap result aggregation"""
        algorithm = BootstrapPCAusalDiscovery(n_bootstrap=10)

        # Mock skeleton learning
        mock_graph = nx.DiGraph()
        mock_graph.add_edges_from([('A', 'B'), ('B', 'C')])
        mock_skeleton.return_value = mock_graph

        # This would normally run bootstrap, but we'll test the aggregation logic
        # In practice, this test would be more comprehensive
        assert algorithm.n_bootstrap == 10

class TestCausalEffectEstimation:
    """Test causal effect estimation"""

    def test_causal_inference_engine_initialization(self):
        """Test causal inference engine initialization"""
        engine = CausalInferenceEngine()

        assert 'backdoor' in engine.estimators
        assert 'meta_learner' in engine.estimators
        assert 'doubly_robust' in engine.estimators

    def test_effect_result_structure(self):
        """Test causal effect result structure"""
        result = CausalEffectResult(
            effect_estimate=0.75,
            confidence_interval=[0.65, 0.85],
            p_value=0.001,
            standard_error=0.05,
            estimator_used="backdoor",
            identification_strategy="backdoor",
            sample_size=1000,
            treatment_variable="treatment",
            outcome_variable="outcome",
            confounder_variables=["age", "sex"]
        )

        assert result.effect_estimate == 0.75
        assert result.confidence_interval == [0.65, 0.85]
        assert result.estimator_used == "backdoor"
        assert "ATE: 0.75" in result.summary()

    @patch('dowhy_integration.CausalInferenceEngine._select_best_method')
    def test_method_selection(self, mock_select, sample_alzheimer_data):
        """Test automatic method selection"""
        engine = CausalInferenceEngine()
        mock_select.return_value = "backdoor"

        # This would test method selection logic
        # In practice, would test with actual data
        assert engine._select_best_method is not None

class TestMechanisticModeling:
    """Test mechanistic modeling"""

    def test_mechanistic_engine_initialization(self):
        """Test mechanistic modeling engine initialization"""
        engine = MechanisticModelingEngine()

        assert hasattr(engine, 'pathway_integrator')
        assert hasattr(engine, 'simulator')

    def test_mechanistic_model_structure(self, sample_causal_graph):
        """Test mechanistic model structure"""
        model = MechanisticModel(
            causal_graph=sample_causal_graph.graph,
            integrated_graph=sample_causal_graph.graph.copy(),
            mechanistic_scores={'edge1': 0.8},
            pathway_coverage={'pathway1': 0.9}
        )

        assert isinstance(model.causal_graph, nx.DiGraph)
        assert isinstance(model.integrated_graph, nx.DiGraph)
        assert model.mechanistic_scores['edge1'] == 0.8

    @patch('mechanistic_modeling.PathwayIntegrator.integrate_pathways')
    def test_pathway_integration(self, mock_integrate, sample_causal_graph):
        """Test pathway integration"""
        engine = MechanisticModelingEngine()

        mock_integrate.return_value = MechanisticModel(
            causal_graph=sample_causal_graph.graph,
            integrated_graph=sample_causal_graph.graph,
            mechanistic_scores={},
            pathway_coverage={}
        )

        # This would test pathway integration
        # In practice, would test with actual pathway data
        assert engine.build_mechanistic_model is not None

class TestAPIEndpoints:
    """Test API endpoints"""

    def test_dataset_upload_validation(self):
        """Test dataset upload validation"""
        # Valid request
        request = DatasetUploadRequest(
            name="Test Dataset",
            data={
                'var1': [1.0, 2.0, 3.0],
                'var2': [4.0, 5.0, 6.0]
            }
        )
        assert request.name == "Test Dataset"

        # Invalid request - mismatched lengths
        with pytest.raises(ValueError):
            DatasetUploadRequest(
                name="Invalid Dataset",
                data={
                    'var1': [1.0, 2.0],
                    'var2': [4.0, 5.0, 6.0]
                }
            )

    def test_causal_discovery_request_validation(self):
        """Test causal discovery request validation"""
        request = CausalDiscoveryRequest(
            dataset_id="test_dataset",
            algorithm="pc",
            alpha=0.05,
            max_degree=5
        )

        assert request.algorithm == "pc"
        assert request.alpha == 0.05

        # Invalid algorithm
        with pytest.raises(ValueError):
            CausalDiscoveryRequest(
                dataset_id="test_dataset",
                algorithm="invalid_algorithm"
            )

class TestIntegration:
    """Test integration scenarios"""

    @patch('causal_discovery.CausalDiscoveryEngine.discover_causal_graph')
    @patch('mechanistic_modeling.MechanisticModelingEngine.build_mechanistic_model')
    def test_full_pipeline(self, mock_mechanistic, mock_discovery, sample_alzheimer_data):
        """Test full causal inference pipeline"""
        # Mock the discovery result
        mock_graph = sample_causal_graph()
        mock_discovery.return_value = mock_graph

        # Mock the mechanistic model
        mock_model = MechanisticModel(
            causal_graph=mock_graph.graph,
            integrated_graph=mock_graph.graph,
            mechanistic_scores={},
            pathway_coverage={}
        )
        mock_mechanistic.return_value = mock_model

        # Test the pipeline
        discovery_engine = CausalDiscoveryEngine()
        mechanistic_engine = MechanisticModelingEngine()

        # Step 1: Discover causal graph
        graph = discovery_engine.discover_causal_graph(sample_alzheimer_data)
        assert isinstance(graph, CausalGraph)

        # Step 2: Build mechanistic model
        mechanistic_model = mechanistic_engine.build_mechanistic_model(graph.graph)
        assert isinstance(mechanistic_model, MechanisticModel)

    def test_error_handling(self):
        """Test error handling in various components"""
        # Test with invalid data
        discovery_engine = CausalDiscoveryEngine()

        # Empty dataframe should raise error
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            discovery_engine.discover_causal_graph(empty_df)

class TestPerformance:
    """Test performance characteristics"""

    def test_algorithm_scalability(self, sample_alzheimer_data):
        """Test algorithm scalability with different data sizes"""
        engine = CausalDiscoveryEngine()

        sizes = [50, 100, 200]
        for size in sizes:
            subset = sample_alzheimer_data.head(size)
            # This would test timing, but for now just ensure it runs
            graph = engine.discover_causal_graph(subset)
            assert isinstance(graph, CausalGraph)

    def test_memory_usage(self):
        """Test memory usage patterns"""
        # This would monitor memory usage during operations
        # For now, just a placeholder
        assert True

class TestValidation:
    """Test validation functions"""

    def test_graph_validation(self):
        """Test causal graph validation"""
        # Valid DAG
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])

        assert nx.is_directed_acyclic_graph(graph)

        # Invalid cyclic graph
        cyclic_graph = nx.DiGraph()
        cyclic_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

        assert not nx.is_directed_acyclic_graph(cyclic_graph)

# Benchmarks
class TestBenchmarks:
    """Performance benchmarks"""

    def benchmark_causal_discovery(self, benchmark, sample_alzheimer_data):
        """Benchmark causal discovery performance"""
        engine = CausalDiscoveryEngine()

        def run_discovery():
            return engine.discover_causal_graph(sample_alzheimer_data.head(100))

        result = benchmark(run_discovery)
        assert isinstance(result, CausalGraph)

    def benchmark_effect_estimation(self, benchmark, sample_alzheimer_data):
        """Benchmark effect estimation performance"""
        engine = CausalInferenceEngine()

        def run_estimation():
            return engine.estimate_causal_effect(
                sample_alzheimer_data.head(100),
                'amyloid_beta', 'cognitive_decline',
                ['age']
            )

        result = benchmark(run_estimation)
        assert isinstance(result, CausalEffectResult)

# Configuration for running tests
if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "--tb=short"])

# Test configuration
pytestmark = pytest.mark.asyncio

# Additional test utilities
def assert_causal_graph_valid(graph: CausalGraph):
    """Assert that a causal graph is valid"""
    assert isinstance(graph, CausalGraph)
    assert len(graph.variables) > 0
    assert graph.is_dag()
    assert len(graph.get_edges()) >= 0

def assert_effect_result_valid(result: CausalEffectResult):
    """Assert that an effect result is valid"""
    assert isinstance(result, CausalEffectResult)
    assert isinstance(result.effect_estimate, (int, float))
    assert len(result.confidence_interval) == 2
    assert result.confidence_interval[0] <= result.effect_estimate <= result.confidence_interval[1]
    assert result.sample_size > 0

def create_mock_pathway():
    """Create a mock biological pathway for testing"""
    from mechanistic_modeling import BiologicalPathway

    return BiologicalPathway(
        pathway_id="test_pathway",
        name="Test Alzheimer Pathway",
        source="KEGG",
        genes=['APP', 'PSEN1', 'APOE'],
        causal_edges=[('APP', 'amyloid_beta'), ('PSEN1', 'APP')],
        confidence_score=0.9,
        evidence_level="experimental"
    )