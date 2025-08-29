"""
LLM Providers Module

This module contains all LLM provider implementations for the unified orchestration system.
Providers implement the BaseLLMProvider interface and are managed by the ProviderRegistry.
"""

from .base_provider import (
    BaseLLMProvider,
    MockLLMProvider,
    ProviderStatus,
    Message,
    Conversation,
    GenerationRequest,
    GenerationResponse
)

from .registry import ProviderRegistry, get_provider_registry, reset_provider_registry

# Import specific providers
try:
    from .claude_provider import ClaudeProvider
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    from .openai_provider import OpenAIProvider
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

__all__ = [
    # Base classes and interfaces
    "BaseLLMProvider",
    "MockLLMProvider",
    "ProviderStatus",
    "Message",
    "Conversation",
    "GenerationRequest",
    "GenerationResponse",
    
    # Registry
    "ProviderRegistry",
    "get_provider_registry",
    "reset_provider_registry",
    
    # Provider availability flags
    "CLAUDE_AVAILABLE",
    "OPENAI_AVAILABLE"
]

# Add providers to __all__ if available
if CLAUDE_AVAILABLE:
    __all__.append("ClaudeProvider")

if OPENAI_AVAILABLE:
    __all__.append("OpenAIProvider")
