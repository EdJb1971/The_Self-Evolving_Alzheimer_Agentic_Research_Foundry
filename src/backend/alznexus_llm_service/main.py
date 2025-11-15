import os
import re
import uuid
import requests
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
    NOTE: This is a basic heuristic. A robust solution requires dedicated content moderation
    services (e.g., OpenAI Moderation, Google Perspective API) and specialized PII detection libraries.
    """
    ethical_flags = {"harmful_content_detected": False, "pii_leak_detected": False}

    # Simulate detection of harmful content
    harmful_keywords = ["unethical_term", "hate speech", "violence", "self-harm", "discrimination"]
    if any(keyword in response_text.lower() for keyword in harmful_keywords):
        ethical_flags["harmful_content_detected"] = True

    # Simulate PII detection with expanded regex patterns
    pii_patterns = [
        r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b', # SSN-like pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', # Email address
        r'\b(?:\d{3}[-.]?|\(\d{3}\)\s?)\d{3}[-.]?\d{4}\b', # Phone number
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b' # Basic credit card pattern (Visa, MC, Amex, Discover, JCB, Diners Club)
    ]
    if any(re.search(pattern, response_text) for pattern in pii_patterns):
        ethical_flags["pii_leak_detected"] = True

    return ethical_flags

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
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Sends a chat completion request to a configured LLM, with ethical safeguards."""
    request_id = str(uuid.uuid4()) # SEC-LLM-003: Generate unique request ID
    original_prompt = chat_request.prompt
    sanitized_prompt = sanitize_prompt(original_prompt)
    injection_detected = detect_prompt_injection(sanitized_prompt)

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
        api_response = call_llm_api(chat_request.model_name, messages)
        response_text = api_response["choices"][0]["message"]["content"]
        finish_reason = api_response["choices"][0]["finish_reason"]
        usage = api_response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(sanitized_prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(response_text.split()))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")

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
        api_response = call_llm_api(tool_use_request.model_name, messages, tools_formatted)
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
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")

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

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "alznexus_llm_service"}
