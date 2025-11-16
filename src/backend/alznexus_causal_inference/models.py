"""
Data models and schemas for the Causal Inference Service

Includes SQLAlchemy models for database persistence and Pydantic schemas
for API validation and serialization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, ConfigDict
import networkx as nx
import json
import pandas as pd

Base = declarative_base()

# SQLAlchemy Models

class Dataset(Base):
    """Dataset model for storing uploaded datasets"""
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    columns = Column(JSON)  # List of column names
    shape = Column(JSON)    # [n_rows, n_cols]
    additional_metadata = Column(JSON) # Additional metadata
    data_hash = Column(String)  # Hash for data integrity
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    causal_graphs = relationship("CausalGraph", back_populates="dataset")
    causal_effects = relationship("CausalEffect", back_populates="dataset")

class CausalGraph(Base):
    """Causal graph model"""
    __tablename__ = "causal_graphs"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    algorithm = Column(String, nullable=False)
    variables = Column(JSON)  # List of variable names
    edges = Column(JSON)      # List of (source, target) tuples
    adjacency_matrix = Column(JSON)  # Serialized adjacency matrix
    confidence_scores = Column(JSON)  # Edge confidence scores
    bootstrap_stability = Column(JSON)  # Bootstrap stability scores
    mechanistic_scores = Column(JSON)  # Biological plausibility scores
    is_dag = Column(Boolean, default=True)
    dataset_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="causal_graphs")
    mechanistic_models = relationship("MechanisticModel", back_populates="causal_graph")

class CausalEffect(Base):
    """Causal effect estimation result"""
    __tablename__ = "causal_effects"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    treatment_variable = Column(String, nullable=False)
    outcome_variable = Column(String, nullable=False)
    confounder_variables = Column(JSON)  # List of confounders
    estimator_used = Column(String, nullable=False)
    identification_strategy = Column(String, nullable=False)

    # Effect estimates
    effect_estimate = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    p_value = Column(Float)
    standard_error = Column(Float)

    # Additional results
    robustness_score = Column(Float)
    sample_size = Column(Integer)
    refutation_results = Column(JSON)  # Refutation test results
    heterogeneous_effects = Column(JSON)  # Heterogeneity analysis
    mediation_analysis = Column(JSON)    # Mediation analysis

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="causal_effects")

class BiologicalPathway(Base):
    """Biological pathway model"""
    __tablename__ = "biological_pathways"

    id = Column(String, primary_key=True)
    pathway_id = Column(String, nullable=False)  # KEGG/Reactome ID
    name = Column(String, nullable=False)
    source = Column(String, nullable=False)  # KEGG or Reactome
    genes = Column(JSON)      # List of genes
    proteins = Column(JSON)   # List of proteins
    metabolites = Column(JSON) # List of metabolites
    causal_edges = Column(JSON)  # Pathway causal edges
    confidence_score = Column(Float)
    evidence_level = Column(String)

    # Relationships
    mechanistic_models = relationship("MechanisticModelPathway", back_populates="pathway")

class MechanisticModel(Base):
    """Mechanistic model integrating causal graphs and pathways"""
    __tablename__ = "mechanistic_models"

    id = Column(String, primary_key=True)
    causal_graph_id = Column(String, ForeignKey("causal_graphs.id"))
    disease_context = Column(String, default="Alzheimer")
    integrated_graph = Column(JSON)  # Serialized NetworkX graph
    mechanistic_scores = Column(JSON)  # Mechanistic plausibility scores
    pathway_coverage = Column(JSON)   # Pathway coverage scores
    validation_results = Column(JSON) # Model validation results

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    causal_graph = relationship("CausalGraph", back_populates="mechanistic_models")
    pathways = relationship("MechanisticModelPathway", back_populates="mechanistic_model")
    simulations = relationship("InterventionSimulation", back_populates="mechanistic_model")

class MechanisticModelPathway(Base):
    """Many-to-many relationship between mechanistic models and pathways"""
    __tablename__ = "mechanistic_model_pathways"

    id = Column(Integer, primary_key=True)
    mechanistic_model_id = Column(String, ForeignKey("mechanistic_models.id"))
    pathway_id = Column(String, ForeignKey("biological_pathways.id"))

    mechanistic_model = relationship("MechanisticModel", back_populates="pathways")
    pathway = relationship("BiologicalPathway", back_populates="mechanistic_models")

class InterventionSimulation(Base):
    """Intervention simulation results"""
    __tablename__ = "intervention_simulations"

    id = Column(String, primary_key=True)
    mechanistic_model_id = Column(String, ForeignKey("mechanistic_models.id"))
    intervention = Column(JSON)  # Intervention parameters
    time_horizon = Column(Float)
    initial_conditions = Column(JSON)  # Initial state
    simulation_results = Column(JSON)  # Time series results

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    mechanistic_model = relationship("MechanisticModel", back_populates="simulations")

class CounterfactualAnalysis(Base):
    """Counterfactual analysis results"""
    __tablename__ = "counterfactual_analyses"

    id = Column(String, primary_key=True)
    mechanistic_model_id = Column(String, ForeignKey("mechanistic_models.id"))
    observed_data_id = Column(String)  # Reference to observed dataset
    hypothetical_intervention = Column(JSON)
    factual_trajectory = Column(JSON)     # Observed trajectory
    counterfactual_trajectory = Column(JSON)  # Counterfactual trajectory
    counterfactual_effects = Column(JSON)  # Computed effects

    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Schemas for API

class DatasetSchema(BaseModel):
    """Dataset schema"""
    id: str
    name: str
    description: Optional[str] = None
    columns: List[str]
    shape: List[int]
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class CausalGraphSchema(BaseModel):
    """Causal graph schema"""
    id: str
    dataset_id: str
    algorithm: str
    variables: List[str]
    edges: List[List[str]]  # List of [source, target] pairs
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    bootstrap_stability: Dict[str, float] = Field(default_factory=dict)
    mechanistic_scores: Dict[str, float] = Field(default_factory=dict)
    is_dag: bool = True
    dataset_size: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class CausalEffectSchema(BaseModel):
    """Causal effect schema"""
    id: str
    dataset_id: str
    treatment_variable: str
    outcome_variable: str
    confounder_variables: List[str]
    estimator_used: str
    identification_strategy: str
    effect_estimate: float
    confidence_interval: List[float]
    p_value: Optional[float] = None
    standard_error: Optional[float] = None
    robustness_score: Optional[float] = None
    sample_size: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class BiologicalPathwaySchema(BaseModel):
    """Biological pathway schema"""
    id: str
    pathway_id: str
    name: str
    source: str
    genes: List[str] = Field(default_factory=list)
    proteins: List[str] = Field(default_factory=list)
    metabolites: List[str] = Field(default_factory=list)
    causal_edges: List[List[str]] = Field(default_factory=list)
    confidence_score: float = 0.0
    evidence_level: str = "predicted"

    class Config:
        from_attributes = True

class MechanisticModelSchema(BaseModel):
    """Mechanistic model schema"""
    id: str
    causal_graph_id: str
    disease_context: str
    mechanistic_scores: Dict[str, float] = Field(default_factory=dict)
    pathway_coverage: Dict[str, float] = Field(default_factory=dict)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class InterventionSimulationSchema(BaseModel):
    """Intervention simulation schema"""
    id: str
    mechanistic_model_id: str
    intervention: Dict[str, float]
    time_horizon: float
    initial_conditions: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class CounterfactualAnalysisSchema(BaseModel):
    """Counterfactual analysis schema"""
    id: str
    mechanistic_model_id: str
    observed_data_id: str
    hypothetical_intervention: Dict[str, float]
    counterfactual_effects: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# Utility functions for serialization

def serialize_networkx_graph(graph: nx.DiGraph) -> Dict[str, Any]:
    """Serialize NetworkX graph to JSON-compatible format"""
    return {
        'nodes': list(graph.nodes(data=True)),
        'edges': list(graph.edges(data=True)),
        'node_count': len(graph.nodes()),
        'edge_count': len(graph.edges()),
        'is_directed': graph.is_directed(),
        'is_dag': nx.is_directed_acyclic_graph(graph) if graph.is_directed() else False
    }

def deserialize_networkx_graph(graph_data: Dict[str, Any]) -> nx.DiGraph:
    """Deserialize NetworkX graph from JSON format"""
    if graph_data.get('is_directed', True):
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    # Add nodes with attributes
    for node_data in graph_data.get('nodes', []):
        if isinstance(node_data, tuple) and len(node_data) == 2:
            node, attrs = node_data
            graph.add_node(node, **attrs)
        else:
            graph.add_node(node_data)

    # Add edges with attributes
    for edge_data in graph_data.get('edges', []):
        if isinstance(edge_data, tuple) and len(edge_data) >= 2:
            if len(edge_data) == 3:
                source, target, attrs = edge_data
                graph.add_edge(source, target, **attrs)
            else:
                graph.add_edge(edge_data[0], edge_data[1])

    return graph

def serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Serialize pandas DataFrame"""
    return {
        'data': df.to_dict('records'),
        'columns': list(df.columns),
        'index': df.index.tolist(),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

def deserialize_dataframe(df_data: Dict[str, Any]) -> pd.DataFrame:
    """Deserialize pandas DataFrame"""
    return pd.DataFrame(
        df_data['data'],
        columns=df_data.get('columns', []),
        index=df_data.get('index', None)
    )

# Validation helpers

def validate_causal_graph_data(data: Dict[str, Any]) -> bool:
    """Validate causal graph data structure"""
    required_keys = ['variables', 'edges']
    if not all(key in data for key in required_keys):
        return False

    # Check that all edge nodes are in variables
    variables = set(data['variables'])
    for edge in data['edges']:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            return False
        if edge[0] not in variables or edge[1] not in variables:
            return False

    return True

def validate_dataset_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns"""
    return all(col in df.columns for col in required_columns)

def validate_intervention_parameters(intervention: Dict[str, float],
                                   mechanistic_model: MechanisticModel) -> bool:
    """Validate intervention parameters against mechanistic model"""
    model_variables = set(mechanistic_model.causal_graph.variables)
    intervention_variables = set(intervention.keys())

    # All intervention variables should be in the model
    return intervention_variables.issubset(model_variables)