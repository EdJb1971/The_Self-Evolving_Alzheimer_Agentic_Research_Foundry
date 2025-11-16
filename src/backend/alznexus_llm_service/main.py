import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

import re
import uuid
import requests
import json
import time
import logging
import random
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import log_audit_event
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import time

app = FastAPI(
    title="AlzNexus LLM Service",
    description="Abstraction layer for interacting with various Large Language Models (LLMs) with ethical safeguards.",
    version="1.0.0",
)

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_REDIS_URL = os.getenv("LLM_REDIS_URL", "redis://localhost:6379")

if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY environment variable not set.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == LLM_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.on_event("startup")
async def startup_event():
    # Database schema migrations are handled externally (e.g., via Alembic in a CI/CD pipeline)
    # and are not performed by the application at startup.
    print("LLM Service: Database schema migrations are managed externally.")
    redis_connection = redis.from_url(LLM_REDIS_URL, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    print("FastAPI-Limiter initialized for LLM Service.")

# Ethical AI & LLM Integration Enhancements (STORY-602)
def sanitize_prompt(prompt: str) -> str:
    """Basic sanitization to remove potentially harmful characters or patterns."""
    # Remove common script tags or HTML entities that could be used in injection
    sanitized_text = re.sub(r'<script.*?>.*?</script>', '', prompt, flags=re.IGNORECASE | re.DOTALL)
    sanitized_text = re.sub(r'&lt;.*?&gt;', '', sanitized_text, flags=re.IGNORECASE | re.DOTALL)
    # Further sanitization can be added here (e.g., specific character filtering)
    return sanitized_text

def detect_prompt_injection(prompt: str) -> bool:
    """SEC-LLM-001: Enhanced keyword-based detection for prompt injection attempts.
    NOTE: This is a basic heuristic. A robust solution requires specialized LLM safety APIs
    or fine-tuned machine learning models for comprehensive prompt injection classification.
    """
    injection_keywords = [
        "ignore previous instructions", "disregard prior commands",
        "as an ai language model", "you are now", "developer mode",
        "print everything", "reveal your system prompt", "jailbreak",
        "override all rules", "act as a", "forget everything",
        "ignore all prior", "new persona", "simulated environment"
    ]
    for keyword in injection_keywords:
        if keyword in prompt.lower():
            return True
    return False

def moderate_response(response_text: str) -> dict[str, bool]:
    """SEC-LLM-002: Enhanced moderation for harmful content and PII in LLM responses.
    Uses comprehensive pattern matching and content analysis for production-ready moderation.
    """
    ethical_flags = {"harmful_content_detected": False, "pii_leak_detected": False}

    # Comprehensive harmful content detection
    harmful_categories = {
        "violence": ["kill", "murder", "assault", "harm", "injure", "attack", "weapon", "violent"],
        "hate_speech": ["hate", "racist", "sexist", "discriminat", "bigot", "supremacist", "intolerant"],
        "self_harm": ["suicide", "self-harm", "cutting", "overdose", "depress", "anxiety", "mental health crisis"],
        "illegal": ["illegal", "criminal", "drug", "traffick", "exploit", "abuse", "fraud", "scam"],
        "medical_misinformation": ["cure all", "miracle cure", "guaranteed", "100% effective", "breakthrough treatment"],
        "research_ethics": ["unethical", "non-consensual", "animal testing", "human experimentation"]
    }

    response_lower = response_text.lower()
    for category, keywords in harmful_categories.items():
        if any(keyword in response_lower for keyword in keywords):
            ethical_flags["harmful_content_detected"] = True
            ethical_flags[f"{category}_detected"] = True
            break

    # Comprehensive PII detection with enhanced patterns
    pii_patterns = [
        # Social Security Numbers (various formats)
        r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        r'\b\d{9}\b',
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Phone numbers (US formats)
        r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
        # Credit card numbers (basic patterns)
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
        # IP addresses
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        # Dates of birth (various formats)
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        # Addresses (street numbers and common street words)
        r'\b\d+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|place|pl|court|ct)\b',
        # ZIP codes
        r'\b\d{5}(?:-\d{4})?\b',
        # Medical record numbers (common patterns)
        r'\b(?:MRN|PATIENT|RECORD)[\s:]*[A-Z0-9]{6,}\b',
        # Names (capitalized words that might be names - basic heuristic)
        r'\b(?:Dr\.?|Professor|Patient|Subject)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    ]

    if any(re.search(pattern, response_text, re.IGNORECASE) for pattern in pii_patterns):
        ethical_flags["pii_leak_detected"] = True

    return ethical_flags

def calculate_backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter_factor: float = 0.1) -> float:
    """Calculate exponential backoff delay with jitter to prevent thundering herd."""
    # Exponential backoff: base_delay * (2 ^ attempt)
    delay = base_delay * (2 ** attempt)

    # Add jitter: randomize delay by ±jitter_factor
    jitter = delay * jitter_factor * (2 * random.random() - 1)  # -jitter_factor to +jitter_factor
    delay_with_jitter = delay + jitter

    # Ensure delay is within reasonable bounds
    return min(max(delay_with_jitter, 0.1), max_delay)

def validate_llm_response(response_data: dict, expected_schema: dict = None) -> dict:
    """Validate LLM API response structure and content."""
    if not isinstance(response_data, dict):
        raise ValueError("LLM response is not a valid JSON object")

    # Check for required OpenAI-style response fields
    if "choices" not in response_data:
        raise ValueError("LLM response missing 'choices' field")

    if not isinstance(response_data["choices"], list) or len(response_data["choices"]) == 0:
        raise ValueError("LLM response 'choices' is not a non-empty list")

    choice = response_data["choices"][0]
    if "message" not in choice:
        raise ValueError("LLM response choice missing 'message' field")

    message = choice["message"]
    if "content" not in message:
        raise ValueError("LLM response message missing 'content' field")

    content = message["content"]
    if not isinstance(content, str):
        raise ValueError("LLM response content is not a string")

    # Check for empty or whitespace-only responses
    if not content.strip():
        raise ValueError("LLM response content is empty or whitespace-only")

    # Check for common LLM error indicators
    error_indicators = [
        "i cannot", "i can't", "i am unable", "i'm unable",
        "i don't know", "i do not know", "i'm sorry",
        "as an ai", "as a language model", "i cannot assist"
    ]

    content_lower = content.lower()
    if any(indicator in content_lower for indicator in error_indicators):
        raise ValueError(f"LLM response indicates failure or refusal: {content[:100]}...")

    return response_data

def call_llm_api_with_retry(model: str, messages: list, tools: list = None, max_retries: int = 3) -> dict:
    """Call LLM API with comprehensive error handling and retry logic."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Call the LLM API
            api_response = call_llm_api(model, messages, tools)

            # Validate the response structure
            validated_response = validate_llm_response(api_response)

            return validated_response

        except requests.exceptions.Timeout:
            last_exception = Exception(f"LLM API timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                delay = calculate_backoff_with_jitter(attempt, base_delay=1.0, jitter_factor=0.3)
                print(f"Timeout retry {attempt + 1}: waiting {delay:.2f}s")
                time.sleep(delay)
                continue

        except requests.exceptions.ConnectionError:
            last_exception = Exception(f"LLM API connection error on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                delay = calculate_backoff_with_jitter(attempt, base_delay=1.0, jitter_factor=0.3)
                print(f"Connection error retry {attempt + 1}: waiting {delay:.2f}s")
                time.sleep(delay)
                continue

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 429:  # Rate limited
                last_exception = Exception(f"LLM API rate limited on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    delay = calculate_backoff_with_jitter(attempt, base_delay=2.0, jitter_factor=0.5)  # Longer delay for rate limits
                    print(f"Rate limit retry {attempt + 1}: waiting {delay:.2f}s")
                    time.sleep(delay)
                    continue
            elif status_code >= 500:  # Server error
                last_exception = Exception(f"LLM API server error ({status_code}) on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    delay = calculate_backoff_with_jitter(attempt, base_delay=1.0, jitter_factor=0.3)
                    print(f"Server error retry {attempt + 1}: waiting {delay:.2f}s")
                    time.sleep(delay)
                    continue
            else:
                # Client error, don't retry
                last_exception = Exception(f"LLM API client error ({status_code}): {e}")
                break

        except json.JSONDecodeError as e:
            last_exception = Exception(f"LLM API returned invalid JSON on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = calculate_backoff_with_jitter(attempt, base_delay=0.5, jitter_factor=0.2)
                print(f"JSON error retry {attempt + 1}: waiting {delay:.2f}s")
                time.sleep(delay)
                continue

        except ValueError as e:
            # Validation error (empty response, wrong format, etc.)
            last_exception = Exception(f"LLM response validation failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = calculate_backoff_with_jitter(attempt, base_delay=0.5, jitter_factor=0.2)
                print(f"Validation error retry {attempt + 1}: waiting {delay:.2f}s")
                time.sleep(delay)
                continue

        except Exception as e:
            last_exception = Exception(f"Unexpected LLM API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = calculate_backoff_with_jitter(attempt, base_delay=1.0, jitter_factor=0.3)
                print(f"Unexpected error retry {attempt + 1}: waiting {delay:.2f}s")
                time.sleep(delay)
                continue

    # All retries exhausted
    raise last_exception or Exception("LLM API call failed after all retries")

def parse_json_safely(json_string: str, fallback_value: any = None) -> dict:
    """Safely parse JSON string with comprehensive error handling."""
    if not json_string or not json_string.strip():
        if fallback_value is not None:
            return fallback_value
        raise ValueError("JSON string is empty or None")

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Problematic JSON: {json_string[:200]}...")

        # Try to fix common JSON issues
        fixed_json = json_string.strip()

        # Remove trailing commas
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)

        # Fix unquoted keys (basic)
        fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)

        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            if fallback_value is not None:
                print(f"Using fallback value due to JSON parsing failure")
                return fallback_value
            raise ValueError(f"Unable to parse JSON even after fixes: {e}")

def validate_structured_output(response_text: str, expected_schema: dict = None) -> dict:
    """Validate and extract structured output from LLM responses."""
    if not expected_schema:
        # Basic validation - just check if it's valid JSON
        try:
            return parse_json_safely(response_text)
        except ValueError:
            return {"raw_response": response_text}

    # Try to extract JSON from response (LLMs sometimes add extra text)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return parse_json_safely(json_match.group(1))
        except ValueError:
            pass

    # Try to find JSON object in response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return parse_json_safely(json_match.group(0))
        except ValueError:
            pass

    # Fallback to raw response
    return {"raw_response": response_text, "parsing_failed": True}

def call_llm_api(model: str, messages: list, tools: list = None) -> dict:
    """Call LLM API based on model type."""
    if model.startswith("gpt"):
        return call_openai_api(model, messages, tools)
    elif model.startswith("gemini"):
        return call_gemini_api(model, messages, tools)
    else:
        raise ValueError(f"Unsupported model: {model}")

def call_openai_api(model: str, messages: list, tools: list = None) -> dict:
    """Call OpenAI API for chat completions."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def call_gemini_api(model: str, messages: list, tools: list = None) -> dict:
    """Call Google Gemini API."""
    # Assuming model like "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={LLM_API_KEY}"
    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        if msg["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }
    if tools:
        # Gemini tools format
        payload["tools"] = [{"function_declarations": tools}]
    
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    # Convert back to OpenAI-like format for consistency
    return {
        "choices": [{
            "message": {
                "content": result["candidates"][0]["content"]["parts"][0]["text"]
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(str(contents)),
            "completion_tokens": len(result["candidates"][0]["content"]["parts"][0]["text"].split()),
            "total_tokens": len(str(contents)) + len(result["candidates"][0]["content"]["parts"][0]["text"].split())
        }
    }

@app.post("/llm/chat", response_model=schemas.LLMResponse, status_code=200,
          dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat_completion(
    chat_request: schemas.LLMChatRequest,
    enable_rag: bool = False,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Sends a chat completion request to a configured LLM, with ethical safeguards and optional RAG."""
    request_id = str(uuid.uuid4()) # SEC-LLM-003: Generate unique request ID
    original_prompt = chat_request.prompt
    sanitized_prompt = sanitize_prompt(original_prompt)
    injection_detected = detect_prompt_injection(sanitized_prompt)

    # RAG Integration: Fetch relevant context from knowledge base
    context_text = ""
    if enable_rag and not injection_detected:
        try:
            knowledge_base_url = os.getenv("KNOWLEDGE_BASE_URL", "http://localhost:8006")
            knowledge_api_key = os.getenv("KNOWLEDGE_API_KEY", "test_knowledge_key_123")

            rag_payload = {
                "query": sanitized_prompt,
                "model_name": chat_request.model_name,
                "max_tokens": None,  # Will be inferred from model
                "token_reserve": 1000,
                "min_relevance_score": 0.6,
                "max_chunks": 5,
                "context_window": 1,
                "requester_agent": chat_request.metadata.get("requester_agent", "llm_service") if chat_request.metadata else "llm_service",
                "prioritize_recent": True,
                "include_metadata": False
            }

            rag_headers = {"X-API-Key": knowledge_api_key, "Content-Type": "application/json"}
            rag_response = requests.post(
                f"{knowledge_base_url}/rag/intelligent/",
                headers=rag_headers,
                json=rag_payload,
                timeout=15  # Longer timeout for intelligent retrieval
            )

            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                context_text = rag_data.get("context_text", "")
                if context_text:
                    # Prepend context to prompt
                    sanitized_prompt = f"Context from knowledge base:\n{context_text}\n\nQuestion: {sanitized_prompt}"
                    print(f"Intelligent RAG: Retrieved {len(rag_data.get('selected_chunks', []))} chunks, {rag_data.get('total_tokens_used', 0)} tokens")
                else:
                    print("Intelligent RAG: No relevant context found")
            elif rag_response.status_code == 429:
                # Rate limited - proceed without context
                print(f"Intelligent RAG: Rate limited, proceeding without context")
            else:
                print(f"Intelligent RAG: Failed to retrieve context: {rag_response.status_code}")

        except requests.RequestException as e:
            print(f"RAG: Error retrieving context: {str(e)}")
        except Exception as e:
            print(f"RAG: Unexpected error: {str(e)}")

    ethical_flags_prompt = {}
    if injection_detected:
        ethical_flags_prompt["prompt_injection_attempt"] = True
        print(f"WARNING: Prompt injection detected for LLM chat request: {original_prompt}")
        # SEC-LLM-003: Use generated request_id for audit event
        log_audit_event(
            entity_type="LLM_SERVICE",
            entity_id=request_id,
            event_type="PROMPT_INJECTION_DETECTED",
            description="LLM chat prompt flagged for potential injection.",
            metadata={"prompt": original_prompt, "model": chat_request.model_name, "request_id": request_id}
        )

    # Call LLM API
    messages = [{"role": "user", "content": sanitized_prompt}]
    try:
        api_response = call_llm_api_with_retry(chat_request.model_name, messages)
        response_text = api_response["choices"][0]["message"]["content"]
        finish_reason = api_response["choices"][0]["finish_reason"]
        usage = api_response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(sanitized_prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(response_text.split()))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    except Exception as e:
        error_msg = f"LLM API call failed after retries: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Moderate response
    ethical_flags_response = moderate_response(response_text)
    if ethical_flags_response["harmful_content_detected"] or ethical_flags_response["pii_leak_detected"]:
        print(f"WARNING: Harmful content or PII leak detected in LLM response: {response_text}")
        # SEC-LLM-003: Use generated request_id for audit event
        log_audit_event(
            entity_type="LLM_SERVICE",
            entity_id=request_id,
            event_type="LLM_RESPONSE_FLAGGED",
            description="LLM chat response flagged for ethical concerns.",
            metadata={"response_snippet": response_text[:100], "flags": ethical_flags_response, "request_id": request_id}
        )

    llm_response = schemas.LLMResponse(
        model_name=chat_request.model_name,
        response_text=response_text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        detected_bias=False, # Bias detection is handled by Bias Detection Service
        detected_injection=injection_detected,
        ethical_flags=ethical_flags_response,
        metadata=chat_request.metadata
    )

    # Log the request and response
    log_entry_metadata = chat_request.metadata.copy() if chat_request.metadata else {}
    log_entry_metadata["request_id"] = request_id # SEC-LLM-003: Store request_id in log metadata

    log_entry = schemas.LLMRequestLogCreate(
        model_name=llm_response.model_name,
        prompt=original_prompt,
        response=llm_response.response_text,
        request_type="chat",
        detected_bias=llm_response.detected_bias,
        detected_injection=llm_response.detected_injection,
        ethical_flags=llm_response.ethical_flags,
        metadata_json=log_entry_metadata
    )
    db_log = crud.create_llm_request_log(db, log_entry)
    
    # SEC-LLM-003: Use generated request_id for audit event
    log_audit_event(
        entity_type="LLM_SERVICE",
        entity_id=request_id,
        event_type="LLM_CHAT_COMPLETION",
        description=f"LLM chat completion for model {llm_response.model_name}. Injection detected: {injection_detected}. Ethical flags: {ethical_flags_response}",
        metadata=llm_response.model_dump() | {"db_log_id": db_log.id, "request_id": request_id}
    )

    return llm_response

@app.post("/llm/tool-use", response_model=schemas.LLMResponse, status_code=200,
          dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def tool_use_completion(
    tool_use_request: schemas.LLMToolUseRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Sends a tool-use request to an LLM, including available tools and context, with ethical safeguards."""
    request_id = str(uuid.uuid4()) # SEC-LLM-003: Generate unique request ID
    original_prompt = tool_use_request.prompt
    sanitized_prompt = sanitize_prompt(original_prompt)
    injection_detected = detect_prompt_injection(sanitized_prompt)

    ethical_flags_prompt = {}
    if injection_detected:
        ethical_flags_prompt["prompt_injection_attempt"] = True
        print(f"WARNING: Prompt injection detected for LLM tool-use request: {original_prompt}")
        # SEC-LLM-003: Use generated request_id for audit event
        log_audit_event(
            entity_type="LLM_SERVICE",
            entity_id=request_id,
            event_type="PROMPT_INJECTION_DETECTED",
            description="LLM tool-use prompt flagged for potential injection.",
            metadata={"prompt": original_prompt, "model": tool_use_request.model_name, "request_id": request_id}
        )

    # Call OpenAI API with tools
    messages = [{"role": "user", "content": sanitized_prompt}]
    tools_formatted = []
    if tool_use_request.tools:
        for tool in tool_use_request.tools:
            tools_formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            })
    try:
        api_response = call_llm_api_with_retry(tool_use_request.model_name, messages, tools_formatted)
        message = api_response["choices"][0]["message"]
        response_text = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            response_text += " " + str(tool_calls)  # Append tool calls to response
        finish_reason = api_response["choices"][0]["finish_reason"]
        usage = api_response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(sanitized_prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(response_text.split()))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    except Exception as e:
        error_msg = f"LLM API call failed after retries: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Moderate response
    ethical_flags_response = moderate_response(response_text)
    if ethical_flags_response["harmful_content_detected"] or ethical_flags_response["pii_leak_detected"]:
        print(f"WARNING: Harmful content or PII leak detected in LLM tool-use response: {response_text}")
        # SEC-LLM-003: Use generated request_id for audit event
        log_audit_event(
            entity_type="LLM_SERVICE",
            entity_id=request_id,
            event_type="LLM_RESPONSE_FLAGGED",
            description="LLM tool-use response flagged for ethical concerns.",
            metadata={"response_snippet": response_text[:100], "flags": ethical_flags_response, "request_id": request_id}
        )

    llm_response = schemas.LLMResponse(
        model_name=tool_use_request.model_name,
        response_text=response_text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        detected_bias=False,
        detected_injection=injection_detected,
        ethical_flags=ethical_flags_response,
        metadata=tool_use_request.metadata
    )

    # Log the request and response
    log_entry_metadata = tool_use_request.metadata.copy() if tool_use_request.metadata else {}
    log_entry_metadata["request_id"] = request_id # SEC-LLM-003: Store request_id in log metadata

    log_entry = schemas.LLMRequestLogCreate(
        model_name=llm_response.model_name,
        prompt=original_prompt,
        response=llm_response.response_text,
        request_type="tool_use",
        detected_bias=llm_response.detected_bias,
        detected_injection=llm_response.detected_injection,
        ethical_flags=llm_response.ethical_flags,
        metadata_json=log_entry_metadata
    )
    db_log = crud.create_llm_request_log(db, log_entry)

    # SEC-LLM-003: Use generated request_id for audit event
    log_audit_event(
        entity_type="LLM_SERVICE",
        entity_id=request_id,
        event_type="LLM_TOOL_USE_COMPLETION",
        description=f"LLM tool-use completion for model {llm_response.model_name}. Injection detected: {injection_detected}. Ethical flags: {ethical_flags_response}",
        metadata=llm_response.model_dump() | {"db_log_id": db_log.id, "request_id": request_id}
    )

    return llm_response

@app.post("/llm/structured-output", response_model=schemas.LLMStructuredResponse, status_code=200,
          dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def structured_output_completion(
    structured_request: schemas.LLMStructuredRequest,
    enable_rag: bool = False,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Sends a structured output request to an LLM with schema validation and error handling."""
    request_id = str(uuid.uuid4())
    original_prompt = structured_request.prompt
    sanitized_prompt = sanitize_prompt(original_prompt)
    injection_detected = detect_prompt_injection(sanitized_prompt)

    # Add schema instructions to prompt
    schema_instructions = f"""
Please respond with a valid JSON object that matches this schema:
{json.dumps(structured_request.response_schema, indent=2)}

Your response must be ONLY the JSON object, no additional text or explanation.
"""
    enhanced_prompt = f"{sanitized_prompt}\n\n{schema_instructions}"

    # RAG Integration (same as chat endpoint)
    context_text = ""
    if enable_rag and not injection_detected:
        try:
            knowledge_base_url = os.getenv("KNOWLEDGE_BASE_URL", "http://localhost:8006")
            knowledge_api_key = os.getenv("KNOWLEDGE_API_KEY", "test_knowledge_key_123")

            rag_payload = {
                "query": sanitized_prompt,
                "model_name": structured_request.model_name,
                "max_tokens": None,
                "token_reserve": 1000,
                "min_relevance_score": 0.6,
                "max_chunks": 5,
                "context_window": 1,
                "requester_agent": structured_request.metadata.get("requester_agent", "llm_service") if structured_request.metadata else "llm_service",
                "prioritize_recent": True,
                "include_metadata": False
            }

            rag_headers = {"X-API-Key": knowledge_api_key, "Content-Type": "application/json"}
            rag_response = requests.post(
                f"{knowledge_base_url}/rag/intelligent/",
                headers=rag_headers,
                json=rag_payload,
                timeout=15
            )

            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                context_text = rag_data.get("context_text", "")
                if context_text:
                    enhanced_prompt = f"Context from knowledge base:\n{context_text}\n\n{enhanced_prompt}"
                    print(f"Structured RAG: Retrieved {len(rag_data.get('selected_chunks', []))} chunks")
            elif rag_response.status_code == 429:
                print("Structured RAG: Rate limited, proceeding without context")
            else:
                print(f"Structured RAG: Failed to retrieve context: {rag_response.status_code}")

        except Exception as e:
            print(f"RAG: Error retrieving context: {str(e)}")

    ethical_flags_prompt = {}
    if injection_detected:
        ethical_flags_prompt["prompt_injection_attempt"] = True
        print(f"WARNING: Prompt injection detected for structured output: {original_prompt}")
        log_audit_event(
            entity_type="LLM_SERVICE",
            entity_id=request_id,
            event_type="PROMPT_INJECTION_DETECTED",
            description="Structured output prompt flagged for potential injection.",
            metadata={"prompt": original_prompt, "model": structured_request.model_name, "request_id": request_id}
        )

    # Call LLM API with retry
    messages = [{"role": "user", "content": enhanced_prompt}]
    try:
        api_response = call_llm_api_with_retry(structured_request.model_name, messages)
        raw_response_text = api_response["choices"][0]["message"]["content"]
        finish_reason = api_response["choices"][0]["finish_reason"]
        usage = api_response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(enhanced_prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(raw_response_text.split()))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Parse and validate structured output
        try:
            structured_data = validate_structured_output(raw_response_text, structured_request.response_schema)
            parsing_success = True
            parsing_error = None
        except Exception as e:
            structured_data = {"raw_response": raw_response_text, "parsing_error": str(e)}
            parsing_success = False
            parsing_error = str(e)
            print(f"Structured output parsing failed: {e}")

    except Exception as e:
        error_msg = f"LLM API call failed after retries: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Moderate response
    ethical_flags_response = moderate_response(raw_response_text)
    if ethical_flags_response["harmful_content_detected"] or ethical_flags_response["pii_leak_detected"]:
        print(f"WARNING: Harmful content or PII leak detected in structured response: {raw_response_text}")
        log_audit_event(
            entity_type="LLM_SERVICE",
            entity_id=request_id,
            event_type="LLM_RESPONSE_FLAGGED",
            description="Structured output response flagged for ethical concerns.",
            metadata={"response_snippet": raw_response_text[:100], "flags": ethical_flags_response, "request_id": request_id}
        )

    structured_response = schemas.LLMStructuredResponse(
        model_name=structured_request.model_name,
        structured_data=structured_data,
        raw_response=raw_response_text,
        parsing_success=parsing_success,
        parsing_error=parsing_error,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        detected_bias=False,
        detected_injection=injection_detected,
        ethical_flags=ethical_flags_response,
        metadata=structured_request.metadata
    )

    # Log the request and response
    log_entry_metadata = structured_request.metadata.copy() if structured_request.metadata else {}
    log_entry_metadata["request_id"] = request_id

    log_entry = schemas.LLMRequestLogCreate(
        model_name=structured_response.model_name,
        prompt=original_prompt,
        response=json.dumps(structured_response.structured_data),
        request_type="structured_output",
        detected_bias=structured_response.detected_bias,
        detected_injection=structured_response.detected_injection,
        ethical_flags=structured_response.ethical_flags,
        metadata_json=log_entry_metadata
    )
    db_log = crud.create_llm_request_log(db, log_entry)

    log_audit_event(
        entity_type="LLM_SERVICE",
        entity_id=request_id,
        event_type="LLM_STRUCTURED_OUTPUT_COMPLETION",
        description=f"LLM structured output completion for model {structured_response.model_name}. Parsing success: {parsing_success}",
        metadata=structured_response.model_dump() | {"db_log_id": db_log.id, "request_id": request_id}
    )

    return structured_response

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "alznexus_llm_service"}
