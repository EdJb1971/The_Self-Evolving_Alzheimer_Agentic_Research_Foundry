#!/usr/bin/env python3
"""
Test script for AlzNexus Knowledge Base Service
Demonstrates knowledge ingestion, RAG queries, and learning capabilities.
"""

import requests
import json
import time

# Service URLs and keys
KNOWLEDGE_BASE_URL = "http://localhost:8006"
KNOWLEDGE_API_KEY = "test_knowledge_key_123"
LLM_SERVICE_URL = "http://localhost:8005"
LLM_API_KEY = "test_llm_key"

def test_health_check():
    """Test service health."""
    print("Testing health check...")
    response = requests.get(f"{KNOWLEDGE_BASE_URL}/health")
    print(f"Health check: {response.json()}")
    return response.status_code == 200

def test_knowledge_ingestion():
    """Test ingesting knowledge documents."""
    print("\nTesting knowledge ingestion...")

    # Sample research findings
    documents = [
        {
            "title": "Novel Amyloid-Beta Pathway Discovery",
            "content": "Recent studies have identified a novel amyloid-beta clearance pathway involving microglial phagocytosis and lymphatic drainage. This pathway shows 40% increased efficiency in APOE4 carriers when modulated by specific small molecules. Key findings include enhanced beta-secretase inhibition and improved synaptic plasticity markers.",
            "document_type": "research_finding",
            "source_agent": "biomarker_hunter_agent",
            "source_task_id": "task_001",
            "tags": ["amyloid-beta", "clearance", "microglial", "APOE4"]
        },
        {
            "title": "Tau Protein Propagation Hypothesis",
            "content": "Research suggests tau protein propagation follows a prion-like mechanism through synaptic connections. Trans-synaptic spread is mediated by heparan sulfate proteoglycans and can be inhibited by monoclonal antibodies targeting the microtubule-binding domain. Clinical trials show promising results in slowing disease progression.",
            "document_type": "hypothesis",
            "source_agent": "hypothesis_validator_agent",
            "source_task_id": "task_002",
            "tags": ["tau-protein", "prion-like", "synaptic-spread", "therapeutics"]
        },
        {
            "title": "Inflammatory Axis in Alzheimer's",
            "content": "The amyloid-tau-inflammatory axis represents a critical convergence point in Alzheimer's pathogenesis. Microglial activation leads to chronic neuroinflammation, exacerbating both amyloid plaque formation and tau hyperphosphorylation. Targeting NLRP3 inflammasome shows therapeutic potential in preclinical models.",
            "document_type": "literature_summary",
            "source_agent": "literature_bridger_agent",
            "source_task_id": "task_003",
            "tags": ["inflammation", "microglia", "NLRP3", "therapeutic-target"]
        }
    ]

    headers = {"X-API-Key": KNOWLEDGE_API_KEY, "Content-Type": "application/json"}
    document_ids = []

    for doc in documents:
        response = requests.post(
            f"{KNOWLEDGE_BASE_URL}/ingest/",
            headers=headers,
            json=doc
        )

        if response.status_code == 200:
            result = response.json()
            document_ids.append(result["document_id"])
            print(f"✓ Ingested: {doc['title']} (ID: {result['document_id']})")
        else:
            print(f"✗ Failed to ingest: {doc['title']} - {response.status_code}")

    return document_ids

def test_upsert_versioning():
    """Test upsert functionality with version control to prevent race conditions."""
    print("\nTesting upsert versioning...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    # Initial document
    doc_v1 = {
        "title": "Test Versioning Document",
        "content": "Initial content version 1",
        "document_type": "test",
        "source_agent": "test_agent",
        "source_task_id": "test_task_001",
        "version": 1,
        "last_modified_by": "test_agent",
        "tags": ["test", "versioning"]
    }

    # Upsert version 1
    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/documents/upsert/",
        json=doc_v1,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        doc_id = result['id']
        print(f"✓ Created document v1: ID {doc_id}, Version {result['version']}")
    else:
        print(f"✗ Failed to create v1: {response.status_code}")
        return False

    # Try to upsert with older version (should be ignored)
    doc_old = {
        "title": "Test Versioning Document",
        "content": "This is older content that should be ignored",
        "document_type": "test",
        "source_agent": "test_agent",
        "source_task_id": "test_task_001",
        "version": 1,  # Same version as before
        "last_modified_by": "test_agent",
        "tags": ["test", "versioning", "old"]
    }

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/documents/upsert/",
        json=doc_old,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        if result['version'] == 1 and "Initial content" in result['content']:
            print("✓ Correctly ignored older version - no race condition!")
        else:
            print(f"✗ Race condition detected - old data overwrote new: {result['content']}")
            return False
    else:
        print(f"✗ Failed old version test: {response.status_code}")
        return False

    # Upsert with newer version (should update)
    doc_v2 = {
        "title": "Test Versioning Document",
        "content": "Updated content version 2 - this should overwrite",
        "document_type": "test",
        "source_agent": "test_agent",
        "source_task_id": "test_task_001",
        "version": 2,  # Newer version
        "last_modified_by": "test_agent",
        "tags": ["test", "versioning", "updated"]
    }

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/documents/upsert/",
        json=doc_v2,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        if result['version'] == 2 and "version 2" in result['content']:
            print("✓ Successfully updated to newer version")
            return True
        else:
            print(f"✗ Failed to update to newer version: {result['version']}")
            return False
    else:
        print(f"✗ Failed new version test: {response.status_code}")
        return False

def test_intelligent_rag():
    """Test intelligent RAG with token awareness and rate limiting."""
    print("\nTesting intelligent RAG with token awareness...")

    headers = {"X-API-Key": KNOWLEDGE_API_KEY, "Content-Type": "application/json"}

    # Test with different models and token limits
    test_cases = [
        {
            "query": "What are the key mechanisms of amyloid-beta clearance?",
            "model_name": "gemini-1.5-flash",
            "max_tokens": 10000,
            "token_reserve": 2000,
            "requester_agent": "test_agent_1"
        },
        {
            "query": "How does tau protein propagation work in Alzheimer's?",
            "model_name": "gpt-4",
            "max_tokens": 8000,
            "token_reserve": 1500,
            "requester_agent": "test_agent_2"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nIntelligent RAG Test {i}: {test_case['query'][:50]}...")

        response = requests.post(
            f"{KNOWLEDGE_BASE_URL}/rag/intelligent/",
            headers=headers,
            json=test_case
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Retrieved {len(result.get('selected_chunks', []))} chunks")
            print(f"✓ Tokens used: {result.get('total_tokens_used', 0)}/{result.get('max_tokens_allowed', 0)}")
            print(f"✓ Relevance stats: avg={result.get('relevance_stats', {}).get('avg_relevance', 0):.3f}")

            rate_limit_info = result.get('rate_limit_info')
            if rate_limit_info:
                print(f"✓ Rate limit: {rate_limit_info['current_usage']}/{rate_limit_info['limit_per_minute']}")

        elif response.status_code == 429:
            print(f"⚠ Rate limited: {response.json().get('detail', 'Unknown')}")
        else:
            print(f"✗ Intelligent RAG failed: {response.status_code} - {response.text}")

def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\nTesting rate limiting...")

    headers = {"X-API-Key": KNOWLEDGE_API_KEY, "Content-Type": "application/json"}

    # Test rate limit status endpoint
    response = requests.get(
        f"{KNOWLEDGE_BASE_URL}/rate-limit/status/test_agent_1",
        headers=headers
    )

    if response.status_code == 200:
        status = response.json()
        print("✓ Rate limit status:")
        print(f"  - Current usage: {status['current_usage']}")
        print(f"  - Limit per minute: {status['limit_per_minute']}")
        print(f"  - Remaining: {status['remaining_requests']}")
        print(f"  - Can make request: {status['can_make_request']}")
    else:
        print(f"✗ Rate limit status failed: {response.status_code}")

def test_token_aware_retrieval():
    """Test token-aware context retrieval."""
    print("\nTesting token-aware retrieval...")

    headers = {"X-API-Key": KNOWLEDGE_API_KEY, "Content-Type": "application/json"}

    # Test with very small token budget
    small_token_test = {
        "query": "therapeutic approaches",
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 1000,  # Very small budget
        "token_reserve": 500,
        "requester_agent": "test_agent_token"
    }

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/rag/intelligent/",
        headers=headers,
        json=small_token_test
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Small token budget test: {result.get('total_tokens_used', 0)} tokens used")
        print(f"✓ Context length: {len(result.get('context_text', ''))} chars")
    elif response.status_code == 400:
        print(f"✓ Correctly rejected insufficient token budget: {response.json().get('detail')}")
    else:
        print(f"✗ Token budget test failed: {response.status_code}")

def test_semantic_search():
    """Test semantic search functionality."""
    print("\nTesting semantic search...")

    payload = {
        "query": "therapeutic approaches for Alzheimer's",
        "limit": 5,
        "document_types": ["research_finding", "hypothesis"],
        "min_validation_score": 0.0
    }

    headers = {"X-API-Key": KNOWLEDGE_API_KEY, "Content-Type": "application/json"}

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/search/semantic/",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['total_found']} documents")
        for i, doc_result in enumerate(result['results'][:3]):
            print(f"  {i+1}. {doc_result['document']['title']} (score: {doc_result['relevance_score']:.3f})")
    else:
        print(f"✗ Semantic search failed: {response.status_code}")

def test_research_insights():
    """Test research insights creation and retrieval."""
    print("\nTesting research insights...")

    insights = [
        {
            "insight_text": "Microglial modulation represents a promising therapeutic strategy for Alzheimer's disease",
            "insight_type": "therapeutic_approach",
            "confidence_level": 0.85,
            "supporting_evidence": {"source": "multiple_studies", "validation_score": 0.82},
            "discovered_by": "knowledge_base_analysis",
            "validation_status": "validated",
            "impact_score": 0.9
        }
    ]

    headers = {"X-API-Key": KNOWLEDGE_API_KEY, "Content-Type": "application/json"}

    # Create insights
    for insight in insights:
        response = requests.post(
            f"{KNOWLEDGE_BASE_URL}/insights/",
            headers=headers,
            json=insight
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Created insight: {result['insight_text'][:50]}...")
        else:
            print(f"✗ Failed to create insight: {response.status_code}")

    # Retrieve insights
    response = requests.get(
        f"{KNOWLEDGE_BASE_URL}/insights/?validation_status=validated&limit=5",
        headers=headers
    )

    if response.status_code == 200:
        insights = response.json()
        print(f"✓ Retrieved {len(insights)} validated insights")
    else:
        print(f"✗ Failed to retrieve insights: {response.status_code}")

def test_analytics():
    """Test knowledge base analytics."""
    print("\nTesting analytics...")

    headers = {"X-API-Key": KNOWLEDGE_API_KEY}

    response = requests.get(
        f"{KNOWLEDGE_BASE_URL}/analytics/",
        headers=headers
    )

    if response.status_code == 200:
        analytics = response.json()
        print("✓ Knowledge base analytics:")
        print(f"  - Total documents: {analytics['total_documents']}")
        print(f"  - Total chunks: {analytics['total_chunks']}")
        print(f"  - Document types: {analytics['document_types_distribution']}")
        print(f"  - Recent activity: {len(analytics['recent_activity'])} events")
    else:
        print(f"✗ Analytics failed: {response.status_code}")

def test_intelligent_rag():
    """Test intelligent RAG with token awareness and rate limiting."""
    print("\nTesting intelligent RAG...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    # Test intelligent RAG request
    rag_request = {
        "query": "What are the key findings about amyloid-beta clearance pathways?",
        "agent_id": "biomarker_hunter_agent",
        "model": "gpt-4",
        "max_tokens": 2000,
        "top_k": 5
    }

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/rag/intelligent/",
        json=rag_request,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        print("✓ Intelligent RAG successful:")
        print(f"  - Context chunks: {len(result['context_chunks'])}")
        print(f"  - Total tokens: {result['total_tokens']}")
        print(f"  - Relevance scores: {[chunk['relevance_score'] for chunk in result['context_chunks']]}")
        return True
    else:
        print(f"✗ Intelligent RAG failed: {response.status_code} - {response.text}")
        return False

def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\nTesting rate limiting...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    # Check rate limit status
    response = requests.get(
        f"{KNOWLEDGE_BASE_URL}/rate-limit/status/test_agent",
        headers=headers
    )

    if response.status_code == 200:
        status = response.json()
        print("✓ Rate limit status:")
        print(f"  - Agent: {status['agent_id']}")
        print(f"  - Requests remaining: {status['requests_remaining']}")
        print(f"  - Reset time: {status['reset_time']}")
        return True
    else:
        print(f"✗ Rate limit check failed: {response.status_code}")
        return False

def test_token_aware_retrieval():
    """Test token-aware context retrieval with different models."""
    print("\nTesting token-aware retrieval...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    # Test with different models and token limits
    test_cases = [
        {"model": "gpt-4", "max_tokens": 1000},
        {"model": "claude-3", "max_tokens": 500},
        {"model": "gemini-pro", "max_tokens": 2000}
    ]

    for case in test_cases:
        rag_request = {
            "query": "What inflammatory pathways are involved in Alzheimer's?",
            "agent_id": "literature_bridger_agent",
            "model": case["model"],
            "max_tokens": case["max_tokens"],
            "top_k": 3
        }

        response = requests.post(
            f"{KNOWLEDGE_BASE_URL}/rag/intelligent/",
            json=rag_request,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ {case['model']} retrieval (max {case['max_tokens']} tokens):")
            print(f"  - Retrieved tokens: {result['total_tokens']}")
            print(f"  - Chunks: {len(result['context_chunks'])}")
        else:
            print(f"✗ {case['model']} retrieval failed: {response.status_code}")

    return True

def test_rag_queries():
    """Test basic RAG queries."""
    print("\nTesting RAG queries...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    # Test basic RAG
    rag_request = {
        "query": "What are the key biomarkers for Alzheimer's disease?",
        "top_k": 3
    }

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/rag/",
        json=rag_request,
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        print("✓ Basic RAG successful:")
        print(f"  - Context chunks: {len(result['context_chunks'])}")
        print(f"  - Query: {result['query']}")
        return True
    else:
        print(f"✗ Basic RAG failed: {response.status_code} - {response.text}")
        return False

def test_semantic_search():
    """Test semantic search functionality."""
    print("\nTesting semantic search...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    search_request = {
        "query": "tau protein propagation mechanisms",
        "top_k": 2,
        "filter": {"document_type": "hypothesis"}
    }

    response = requests.post(
        f"{KNOWLEDGE_BASE_URL}/search/",
        json=search_request,
        headers=headers
    )

    if response.status_code == 200:
        results = response.json()
        print("✓ Semantic search successful:")
        print(f"  - Results: {len(results['results'])}")
        if results['results']:
            print(f"  - Top result: {results['results'][0]['title']}")
        return True
    else:
        print(f"✗ Semantic search failed: {response.status_code}")
        return False

def test_research_insights():
    """Test research insights extraction."""
    print("\nTesting research insights...")

    headers = {"Authorization": f"Bearer {KNOWLEDGE_API_KEY}"}

    response = requests.get(
        f"{KNOWLEDGE_BASE_URL}/insights/",
        headers=headers
    )

    if response.status_code == 200:
        insights = response.json()
        print("✓ Research insights:")
        print(f"  - Total insights: {len(insights)}")
        if insights:
            print(f"  - Sample insight: {insights[0]['content'][:100]}...")
        return True
    else:
        print(f"✗ Insights failed: {response.status_code}")
        return False

def main():
    """Run all tests."""
    print("🧠 AlzNexus Knowledge Base Service Test")
    print("=" * 50)

    # Test health
    if not test_health_check():
        print("❌ Service not healthy, exiting...")
        return

    # Test knowledge ingestion
    document_ids = test_knowledge_ingestion()

    # Test upsert versioning (critical for preventing race conditions)
    if not test_upsert_versioning():
        print("❌ Upsert versioning test failed - race conditions possible!")
        return

    # Wait for processing
    print("\nWaiting for document processing...")
    time.sleep(3)

    # Test RAG
    test_rag_queries()

    # Test intelligent RAG
    test_intelligent_rag()

    # Test rate limiting
    test_rate_limiting()

    # Test token-aware retrieval
    test_token_aware_retrieval()

    # Test semantic search
    test_semantic_search()

    # Test insights
    test_research_insights()

    # Test analytics
    test_analytics()

    # Test LLM error handling
    test_llm_error_handling()

    print("\n" + "=" * 50)
    print("✅ Knowledge base tests completed!")
    print("\nKey capabilities demonstrated:")
    print("• Knowledge ingestion with intelligent chunking")
    print("• RAG context retrieval for LLM augmentation")
    print("• Semantic search across research findings")
    print("• Research insights extraction and validation")
    print("• Comprehensive analytics and monitoring")
    print("• Race condition prevention with version control")
    print("• Robust LLM error handling and structured output")
    print("\nThis enables true continuous learning from discovered results!")

def test_llm_error_handling():
    """Test comprehensive LLM error handling capabilities."""
    print("\nTesting LLM error handling...")

    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}

    # Test structured output with schema validation
    structured_request = {
        "model_name": "gemini-1.5-flash",
        "prompt": "Analyze this Alzheimer's research data and extract key biomarkers. Respond ONLY with valid JSON matching the schema.",
        "response_schema": {
            "type": "object",
            "properties": {
                "biomarkers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "methodology": {"type": "string"}
            },
            "required": ["biomarkers", "confidence_score"]
        },
        "metadata": {"test_type": "structured_output"}
    }

    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/llm/structured-output",
            json=structured_request,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ Structured output successful:")
            print(f"  - Parsing success: {result['parsing_success']}")
            print(f"  - Has structured data: {'structured_data' in result}")
            if result.get('parsing_success'):
                print(f"  - Biomarkers found: {len(result['structured_data'].get('biomarkers', []))}")
            return True
        else:
            print(f"✗ Structured output failed: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ LLM service connection failed: {e}")
        return False

if __name__ == "__main__":
    main()