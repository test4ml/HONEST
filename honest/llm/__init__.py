"""
LLM Client Module

This module provides unified sync/async LLM client functionality for multiple LLM protocols.

Example usage (Async):
    from honest.llm import AsyncLLMClient, create_async_llm_client

    client = create_async_llm_client(
        base_url="http://localhost:8000/v1",
        api_key="your-key",
        model_name="Qwen2.5-7B-Instruct"
    )
    answer = await client.generate_answer("What is 2+2?")

Example usage (Sync):
    from honest.llm import SyncLLMClient, create_sync_llm_client

    client = create_sync_llm_client(
        base_url="http://localhost:8000/v1",
        api_key="your-key",
        model_name="Qwen2.5-7B-Instruct"
    )
    answer = client.generate_answer("What is 2+2?")
"""

from .client import (
    # Async
    AsyncLLMClient,
    AsyncRateLimiter,
    create_async_llm_client,
    create_async_llm_client_from_config,
    # Sync
    SyncLLMClient,
    SyncRateLimiter,
    create_sync_llm_client,
    create_sync_llm_client_from_config,
    # Shared
    PromptTemplate,
    DEFAULT_SIMPLE_PROMPT,
    DEFAULT_THINKING_PROMPT,
    DEFAULT_QA_PROMPT,
    DEFAULT_PROTOCOL,
    SUPPORTED_PROTOCOLS,
    # Legacy (aliases for async version)
    LLMClient,
    RateLimiter,
    create_llm_client,
    create_llm_client_from_config,
)

__all__ = [
    # Async
    'AsyncLLMClient',
    'AsyncRateLimiter',
    'create_async_llm_client',
    'create_async_llm_client_from_config',
    # Sync
    'SyncLLMClient',
    'SyncRateLimiter',
    'create_sync_llm_client',
    'create_sync_llm_client_from_config',
    # Shared
    'PromptTemplate',
    'DEFAULT_SIMPLE_PROMPT',
    'DEFAULT_THINKING_PROMPT',
    'DEFAULT_QA_PROMPT',
    'DEFAULT_PROTOCOL',
    'SUPPORTED_PROTOCOLS',
    # Legacy
    'LLMClient',
    'RateLimiter',
    'create_llm_client',
    'create_llm_client_from_config',
]
