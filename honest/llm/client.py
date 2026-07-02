#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LLM Client Module

This module provides unified sync/async LLM clients for multiple LLM protocols.
It includes rate limiting, concurrency control, and retry logic for both
synchronous and asynchronous use cases.
"""

import asyncio
import inspect
import logging
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI

from configs import get_config

logger = logging.getLogger(__name__)

DEFAULT_PROTOCOL = "openai"
SUPPORTED_PROTOCOLS = {"openai", "openai_responses", "anthropic", "gemini"}


class AsyncRateLimiter:
    """Async token bucket rate limiter for API requests"""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.max_tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request (blocks if rate limit exceeded)"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 1.0
                self.last_update = now

            self.tokens -= 1.0


class SyncRateLimiter:
    """Synchronous token bucket rate limiter for API requests"""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.max_tokens = rate
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """Acquire permission to make a request (blocks if rate limit exceeded)"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                sleep_time = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 1.0
                self.last_update = now

            self.tokens -= 1.0


@dataclass
class PromptTemplate:
    """Prompt template for LLM requests"""
    system_prompt: str = "You are a helpful assistant."
    user_prompt_template: str = "{question}"

    def format(self, question: str, **kwargs) -> tuple[str, str]:
        """Format the prompt with the given question and additional parameters"""
        return self.system_prompt, self.user_prompt_template.format(question=question, **kwargs)


@dataclass
class NormalizedPrompt:
    system: str
    user: str


DEFAULT_SIMPLE_PROMPT = PromptTemplate(
    system_prompt="You are a helpful assistant that answers questions accurately and concisely. Provide your answer directly without unnecessary elaboration.",
    user_prompt_template="Answer the following question:\n\n{question}\n\nProvide your answer:"
)

DEFAULT_THINKING_PROMPT = PromptTemplate(
    system_prompt="You are a helpful assistant that answers questions thoughtfully and thoroughly. Think step by step and explain your reasoning before providing your final answer.",
    user_prompt_template="{question}\n\nPlease think through this question carefully. You can:\n- Break down the problem into steps\n- Consider different aspects or perspectives\n- Explain your reasoning process\n- Then provide your conclusion\n\nTake your time and think it through."
)

DEFAULT_QA_PROMPT = PromptTemplate(
    system_prompt="You are a helpful assistant specialized in question answering.",
    user_prompt_template="Question: {question}\n\nAnswer:"
)


def _load_anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError as exc:
        raise ImportError(
            "protocol='anthropic' requires the 'anthropic' package. "
            "Install dependencies from requirements.txt."
        ) from exc


def _load_genai():
    try:
        from google import genai
        return genai
    except ImportError as exc:
        raise ImportError(
            "protocol='gemini' requires the 'google-genai' package. "
            "Install dependencies from requirements.txt."
        ) from exc


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _iter_text_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(_iter_text_values(item))
        return texts

    text = _get_attr_or_key(value, "text")
    if isinstance(text, str):
        return [text]

    content = _get_attr_or_key(value, "content")
    parts = _get_attr_or_key(value, "parts")
    output = _get_attr_or_key(value, "output")
    candidates = _get_attr_or_key(value, "candidates")

    texts = []
    for nested in (content, parts, output, candidates):
        texts.extend(_iter_text_values(nested))
    return texts


class _BaseLLMClientMixin:
    """Shared functionality for both sync and async clients"""

    def _normalize_protocol(self, protocol: Optional[str]) -> str:
        normalized = (protocol or DEFAULT_PROTOCOL).lower()
        if normalized not in SUPPORTED_PROTOCOLS:
            supported = ", ".join(sorted(SUPPORTED_PROTOCOLS))
            raise ValueError(f"Unsupported LLM protocol '{protocol}'. Supported protocols: {supported}")
        return normalized

    def _is_qwen3_model(self) -> bool:
        qwen3_patterns = [
            'qwen3', 'Qwen3', 'Qwen/Qwen3', 'Qwen-3', 'qwen-3'
        ]
        model_name_lower = self.model_name.lower()
        return any(pattern.lower() in model_name_lower for pattern in qwen3_patterns)

    def _is_deepseek_model(self) -> bool:
        """True if the configured model is a DeepSeek reasoning model.

        DeepSeek-V4 series enables thinking mode by default; the way to disable it differs
        from Qwen (see the DeepSeek docs: under the OpenAI format, use
        ``extra_body={"thinking": {"type": "disabled"}}``).
        """
        return 'deepseek' in (self.model_name or '').lower()

    def _should_disable_thinking(self) -> bool:
        if self.enable_thinking is not None:
            return not self.enable_thinking
        # Disable thinking mode by default: for reasoning models such as DeepSeek-V4 / Qwen3,
        # unless explicitly enabled, always treat them as "no thinking" so that they remain
        # comparable to other non-reasoning models under the same setting, and to prevent the
        # thinking chain from consuming all of max_tokens and leaving the final text empty.
        return self._is_qwen3_model() or self._is_deepseek_model()

    def _build_normalized_prompt(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> NormalizedPrompt:
        if system_prompt is None or user_prompt is None:
            sys_prompt, usr_prompt = self.prompt_template.format(question)
        else:
            sys_prompt, usr_prompt = system_prompt, user_prompt
        return NormalizedPrompt(system=sys_prompt, user=usr_prompt)

    def _build_request_params(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> dict:
        prompt = self._build_normalized_prompt(question, system_prompt, user_prompt)
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_temperature = temperature if temperature is not None else self.temperature

        if self.protocol == "openai":
            return self._build_openai_chat_params(prompt, effective_max_tokens, effective_temperature)
        if self.protocol == "openai_responses":
            return self._build_openai_responses_params(prompt, effective_max_tokens, effective_temperature)
        if self.protocol == "anthropic":
            return self._build_anthropic_params(prompt, effective_max_tokens, effective_temperature)
        if self.protocol == "gemini":
            return self._build_gemini_params(prompt, effective_max_tokens, effective_temperature)
        raise ValueError(f"Unsupported LLM protocol '{self.protocol}'")

    def _build_openai_chat_params(
        self,
        prompt: NormalizedPrompt,
        max_tokens: int,
        temperature: float
    ) -> dict:
        request_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self._should_disable_thinking():
            # Disable thinking mode: DeepSeek and Qwen use different extra_body formats.
            # DeepSeek (OpenAI format): extra_body={"thinking": {"type": "disabled"}}
            # Qwen (vLLM): extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            if self._is_deepseek_model():
                request_params["extra_body"] = {"thinking": {"type": "disabled"}}
            else:
                request_params["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

        return request_params

    def _build_openai_responses_params(
        self,
        prompt: NormalizedPrompt,
        max_tokens: int,
        temperature: float
    ) -> dict:
        return {
            "model": self.model_name,
            "input": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

    def _build_anthropic_params(
        self,
        prompt: NormalizedPrompt,
        max_tokens: int,
        temperature: float
    ) -> dict:
        request_params = {
            "model": self.model_name,
            "system": prompt.system,
            "messages": [
                {"role": "user", "content": prompt.user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Disable thinking mode under the Anthropic format (DeepSeek docs):
        # {"thinking": {"type": "disabled"}}. Previously this parameter was not passed, so
        # DeepSeek-V4-Flash defaulted to thinking=enabled, the thinking chain overflowed
        # max_tokens, the text block became empty -> the answer was empty -> NaN.
        if self._should_disable_thinking():
            request_params["thinking"] = {"type": "disabled"}

        return request_params

    def _build_gemini_params(
        self,
        prompt: NormalizedPrompt,
        max_tokens: int,
        temperature: float
    ) -> dict:
        return {
            "model": self.model_name,
            "contents": prompt.user,
            "config": {
                "system_instruction": prompt.system,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        }

    def _extract_text_from_response(self, response: Any) -> str:
        if self.protocol == "openai":
            return self._extract_openai_chat_text(response)
        if self.protocol == "openai_responses":
            return self._extract_openai_responses_text(response)
        if self.protocol == "anthropic":
            return self._extract_anthropic_text(response)
        if self.protocol == "gemini":
            return self._extract_gemini_text(response)
        raise ValueError(f"Unsupported LLM protocol '{self.protocol}'")

    def _extract_openai_chat_text(self, response: Any) -> str:
        choices = _get_attr_or_key(response, "choices", [])
        if choices and len(choices) > 0:
            message = _get_attr_or_key(choices[0], "message")
            content = _get_attr_or_key(message, "content", "")
            return str(content).strip() if content else ""
        return ""

    def _extract_openai_responses_text(self, response: Any) -> str:
        output_text = _get_attr_or_key(response, "output_text")
        if output_text:
            return str(output_text).strip()
        return "".join(_iter_text_values(_get_attr_or_key(response, "output"))).strip()

    def _extract_anthropic_text(self, response: Any) -> str:
        content = _get_attr_or_key(response, "content", [])
        texts = []
        for block in content or []:
            block_type = _get_attr_or_key(block, "type")
            text = _get_attr_or_key(block, "text")
            if text and (block_type in {None, "text"}):
                texts.append(str(text))
        return "".join(texts).strip()

    def _extract_gemini_text(self, response: Any) -> str:
        text = _get_attr_or_key(response, "text")
        if text:
            return str(text).strip()
        return "".join(_iter_text_values(_get_attr_or_key(response, "candidates"))).strip()

    def _is_critical_api_error(self, exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        if self.protocol == "anthropic":
            try:
                anthropic = _load_anthropic()
            except ImportError:
                return False
            critical_errors = tuple(
                error for error in (
                    getattr(anthropic, "APIConnectionError", None),
                    getattr(anthropic, "APITimeoutError", None),
                ) if error is not None
            )
            return bool(critical_errors) and isinstance(exc, critical_errors)
        return False

    def _init_sync_client(self, base_url: Optional[str], api_key: str, timeout: int, timeout_config: httpx.Timeout):
        if self.protocol in {"openai", "openai_responses"}:
            return OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.Client(timeout=timeout_config)
            )
        if self.protocol == "anthropic":
            anthropic = _load_anthropic()
            params = {"auth_token": api_key, "timeout": timeout}
            if base_url:
                params["base_url"] = base_url
            return anthropic.Anthropic(**params)
        if self.protocol == "gemini":
            genai = _load_genai()
            return genai.Client(api_key=api_key)
        raise ValueError(f"Unsupported LLM protocol '{self.protocol}'")

    def _init_async_client(self, base_url: Optional[str], api_key: str, timeout: int, timeout_config: httpx.Timeout):
        if self.protocol in {"openai", "openai_responses"}:
            return AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.AsyncClient(timeout=timeout_config)
            )
        if self.protocol == "anthropic":
            anthropic = _load_anthropic()
            params = {"auth_token": api_key, "timeout": timeout}
            if base_url:
                params["base_url"] = base_url
            return anthropic.AsyncAnthropic(**params)
        if self.protocol == "gemini":
            genai = _load_genai()
            return genai.Client(api_key=api_key)
        raise ValueError(f"Unsupported LLM protocol '{self.protocol}'")


class AsyncLLMClient(_BaseLLMClientMixin):
    """Async LLM client for supported LLM protocols"""

    def __init__(
        self,
        base_url: Optional[str],
        api_key: str,
        model_name: str,
        max_concurrent: int = 10,
        rate_limit: float = 10.0,
        timeout: int = 600,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        prompt_template: Optional[PromptTemplate] = None,
        enable_thinking: bool = None,
        protocol: str = DEFAULT_PROTOCOL
    ):
        self.protocol = self._normalize_protocol(protocol)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = AsyncRateLimiter(rate_limit)
        self.request_count = 0
        self.lock = asyncio.Lock()
        self.prompt_template = prompt_template or DEFAULT_SIMPLE_PROMPT
        self.enable_thinking = enable_thinking

        timeout_config = httpx.Timeout(timeout=timeout, connect=60.0)
        self.client = self._init_async_client(base_url, api_key, timeout, timeout_config)

    async def _create_response(self, request_params: dict):
        if self.protocol == "openai":
            return await self.client.chat.completions.create(**request_params)
        if self.protocol == "openai_responses":
            return await self.client.responses.create(**request_params)
        if self.protocol == "anthropic":
            return await self.client.messages.create(**request_params)
        if self.protocol == "gemini":
            return await asyncio.to_thread(self.client.models.generate_content, **request_params)
        raise ValueError(f"Unsupported LLM protocol '{self.protocol}'")

    async def generate_answer(
        self,
        question: str,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        async with self.semaphore:
            await self.rate_limiter.acquire()

            for retry_attempt in range(max_retries):
                try:
                    async with self.lock:
                        self.request_count += 1

                    request_params = self._build_request_params(
                        question, system_prompt, user_prompt
                    )
                    response = await self._create_response(request_params)
                    return self._extract_text_from_response(response)

                except Exception as e:
                    if self._is_critical_api_error(e):
                        logger.error(f"Critical API error: {e}")
                        raise

                    logger.error(f"Error generating answer (attempt {retry_attempt + 1}/{max_retries}): {e}")

                    if retry_attempt < max_retries - 1:
                        wait_time = (2 ** retry_attempt) * 1.0
                        await asyncio.sleep(wait_time)
                        continue
                    return f"ERROR: {str(e)}"

            return "ERROR: Max retries exceeded"

    async def generate_batch(
        self,
        questions: List[str],
        max_retries: int = 3
    ) -> List[str]:
        tasks = [
            self.generate_answer(q, max_retries)
            for q in questions
        ]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Close the async client if supported"""
        close = getattr(self.client, "close", None)
        if not close:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class SyncLLMClient(_BaseLLMClientMixin):
    """Synchronous LLM client for supported LLM protocols"""

    def __init__(
        self,
        base_url: Optional[str],
        api_key: str,
        model_name: str,
        rate_limit: float = 10.0,
        timeout: int = 600,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        prompt_template: Optional[PromptTemplate] = None,
        enable_thinking: bool = None,
        protocol: str = DEFAULT_PROTOCOL
    ):
        self.protocol = self._normalize_protocol(protocol)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limiter = SyncRateLimiter(rate_limit)
        self.request_count = 0
        self.lock = threading.Lock()
        self.prompt_template = prompt_template or DEFAULT_SIMPLE_PROMPT
        self.enable_thinking = enable_thinking

        timeout_config = httpx.Timeout(timeout=timeout, connect=60.0)
        self.client = self._init_sync_client(base_url, api_key, timeout, timeout_config)

    def _create_response(self, request_params: dict):
        if self.protocol == "openai":
            return self.client.chat.completions.create(**request_params)
        if self.protocol == "openai_responses":
            return self.client.responses.create(**request_params)
        if self.protocol == "anthropic":
            return self.client.messages.create(**request_params)
        if self.protocol == "gemini":
            return self.client.models.generate_content(**request_params)
        raise ValueError(f"Unsupported LLM protocol '{self.protocol}'")

    def generate_answer(
        self,
        question: str,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        self.rate_limiter.acquire()

        for retry_attempt in range(max_retries):
            try:
                with self.lock:
                    self.request_count += 1

                request_params = self._build_request_params(
                    question, system_prompt, user_prompt
                )
                response = self._create_response(request_params)
                return self._extract_text_from_response(response)

            except Exception as e:
                if self._is_critical_api_error(e):
                    logger.error(f"Critical API error: {e}")
                    raise

                logger.error(f"Error generating answer (attempt {retry_attempt + 1}/{max_retries}): {e}")

                if retry_attempt < max_retries - 1:
                    wait_time = (2 ** retry_attempt) * 1.0
                    time.sleep(wait_time)
                    continue
                return f"ERROR: {str(e)}"

        return "ERROR: Max retries exceeded"

    def generate_batch(
        self,
        questions: List[str],
        max_retries: int = 3
    ) -> List[str]:
        return [
            self.generate_answer(q, max_retries)
            for q in questions
        ]

    def close(self):
        """Close the client if supported"""
        close = getattr(self.client, "close", None)
        if not close:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = close()
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop.run_until_complete(result)
                except RuntimeError:
                    asyncio.run(result)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


LLMClient = AsyncLLMClient


def create_async_llm_client_from_config(
    config_key: str = 'local',
    **kwargs
) -> AsyncLLMClient:
    """Create async LLM client from configuration"""
    llm_config = get_config(f'llm.{config_key}', {})

    base_url = kwargs.pop('base_url', llm_config.get('base_url', 'http://localhost:8000/v1'))
    api_key = kwargs.pop('api_key', llm_config.get('api_key', 'your-api-key-here'))
    model_name = kwargs.pop('model_name', None) or kwargs.pop('model', None) or llm_config.get('default_model', 'Qwen2.5-7B-Instruct')
    protocol = kwargs.pop('protocol', llm_config.get('protocol', DEFAULT_PROTOCOL))

    params = {
        'base_url': base_url,
        'api_key': api_key,
        'model_name': model_name,
        'protocol': protocol,
        'max_concurrent': kwargs.pop('max_concurrent', llm_config.get('max_concurrent', 10)),
        'rate_limit': kwargs.pop('rate_limit', llm_config.get('rate_limit', 10.0)),
        'timeout': kwargs.pop('timeout', llm_config.get('timeout', 600)),
        'max_tokens': kwargs.pop('max_tokens', llm_config.get('max_tokens', 2048)),
        'temperature': kwargs.pop('temperature', llm_config.get('temperature', 0.0)),
    }
    params.update(kwargs)

    return AsyncLLMClient(**params)


def create_sync_llm_client_from_config(
    config_key: str = 'local',
    **kwargs
) -> SyncLLMClient:
    """Create sync LLM client from configuration"""
    llm_config = get_config(f'llm.{config_key}', {})

    base_url = kwargs.pop('base_url', llm_config.get('base_url', 'http://localhost:8000/v1'))
    api_key = kwargs.pop('api_key', llm_config.get('api_key', 'your-api-key-here'))
    model_name = kwargs.pop('model_name', None) or kwargs.pop('model', None) or llm_config.get('default_model', 'Qwen2.5-7B-Instruct')
    protocol = kwargs.pop('protocol', llm_config.get('protocol', DEFAULT_PROTOCOL))

    params = {
        'base_url': base_url,
        'api_key': api_key,
        'model_name': model_name,
        'protocol': protocol,
        'rate_limit': kwargs.pop('rate_limit', llm_config.get('rate_limit', 10.0)),
        'timeout': kwargs.pop('timeout', llm_config.get('timeout', 600)),
        'max_tokens': kwargs.pop('max_tokens', llm_config.get('max_tokens', 2048)),
        'temperature': kwargs.pop('temperature', llm_config.get('temperature', 0.0)),
    }
    params.update(kwargs)

    return SyncLLMClient(**params)


def create_async_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    protocol: Optional[str] = None,
    **kwargs
) -> AsyncLLMClient:
    """Create async LLM client with automatic config fallback"""
    llm_config = get_config('llm.local', {})

    base_url = base_url if base_url is not None else llm_config.get('base_url', 'http://localhost:8000/v1')
    api_key = api_key or llm_config.get('api_key', 'your-api-key-here')
    model_name = model_name or llm_config.get('default_model', 'Qwen2.5-7B-Instruct')
    protocol = protocol or llm_config.get('protocol', DEFAULT_PROTOCOL)

    if api_key == 'your-api-key-here':
        raise ValueError(
            "API key not configured. Please provide --api-key parameter, "
            "set LLM_API_KEY environment variable, or configure it in configs/default.yaml"
        )

    params = {
        'base_url': base_url,
        'api_key': api_key,
        'model_name': model_name,
        'protocol': protocol,
        **kwargs
    }

    return AsyncLLMClient(**params)


def create_sync_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    protocol: Optional[str] = None,
    **kwargs
) -> SyncLLMClient:
    """Create sync LLM client with automatic config fallback"""
    llm_config = get_config('llm.local', {})

    base_url = base_url if base_url is not None else llm_config.get('base_url', 'http://localhost:8000/v1')
    api_key = api_key or llm_config.get('api_key', 'your-api-key-here')
    model_name = model_name or llm_config.get('default_model', 'Qwen2.5-7B-Instruct')
    protocol = protocol or llm_config.get('protocol', DEFAULT_PROTOCOL)

    if api_key == 'your-api-key-here':
        raise ValueError(
            "API key not configured. Please provide --api-key parameter, "
            "set LLM_API_KEY environment variable, or configure it in configs/default.yaml"
        )

    params = {
        'base_url': base_url,
        'api_key': api_key,
        'model_name': model_name,
        'protocol': protocol,
        **kwargs
    }

    return SyncLLMClient(**params)


def create_llm_client_from_config(config_key: str = 'local', **kwargs) -> AsyncLLMClient:
    """Legacy alias for create_async_llm_client_from_config"""
    return create_async_llm_client_from_config(config_key, **kwargs)


def create_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    protocol: Optional[str] = None,
    **kwargs
) -> AsyncLLMClient:
    """Legacy alias for create_async_llm_client"""
    return create_async_llm_client(base_url, api_key, model_name, protocol, **kwargs)


RateLimiter = AsyncRateLimiter
LLMClient = AsyncLLMClient


__all__ = [
    'AsyncLLMClient',
    'AsyncRateLimiter',
    'create_async_llm_client',
    'create_async_llm_client_from_config',
    'SyncLLMClient',
    'SyncRateLimiter',
    'create_sync_llm_client',
    'create_sync_llm_client_from_config',
    'PromptTemplate',
    'DEFAULT_SIMPLE_PROMPT',
    'DEFAULT_THINKING_PROMPT',
    'DEFAULT_QA_PROMPT',
    'DEFAULT_PROTOCOL',
    'SUPPORTED_PROTOCOLS',
    'LLMClient',
    'RateLimiter',
    'create_llm_client',
    'create_llm_client_from_config',
]
