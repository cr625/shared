"""
Unified LLM Orchestration Module

A comprehensive LLM service that combines the best features from ProEthica,
OntExtract, and OntServe systems to provide intelligent language model
orchestration with ontological context awareness.

Features:
- Multi-provider architecture with intelligent fallback
- Context-aware prompt building with ontological enrichment
- Conversation management and threading
- MCP integration for rich semantic context
- Provider health monitoring and metrics collection
- Caching and performance optimization

Primary Goals:
- Enable LLMs to interact directly with MCP servers for complex ontology information
- Provide responsive and reliable LLM services across all systems
- Maintain consistent behavior while allowing for system-specific customizations
"""

from .core.orchestrator import LLMOrchestrator, OrchestratorConfig, get_llm_orchestrator, reset_llm_orchestrator
from .providers.registry import ProviderRegistry, get_provider_registry, reset_provider_registry
from .providers import (
    BaseLLMProvider, MockLLMProvider, ProviderStatus,
    Message, Conversation, GenerationRequest, GenerationResponse,
    CLAUDE_AVAILABLE, OPENAI_AVAILABLE
)
from .integrations.mcp_context import MCPContextManager, get_mcp_context_manager, reset_mcp_context_manager

# Import specific providers if available
if CLAUDE_AVAILABLE:
    from .providers.claude_provider import ClaudeProvider

if OPENAI_AVAILABLE:
    from .providers.openai_provider import OpenAIProvider

__version__ = "1.0.0"
__author__ = "ProEthica Development Team"

# Export main components
__all__ = [
    # Core orchestration
    "LLMOrchestrator",
    "OrchestratorConfig", 
    "get_llm_orchestrator",
    "reset_llm_orchestrator",
    
    # Provider management
    "ProviderRegistry",
    "get_provider_registry",
    "reset_provider_registry",
    
    # Provider interfaces
    "BaseLLMProvider",
    "MockLLMProvider",
    "ProviderStatus",
    
    # Message and conversation handling
    "Message",
    "Conversation",
    "GenerationRequest",
    "GenerationResponse",
    
    # MCP integration
    "MCPContextManager",
    "get_mcp_context_manager",
    "reset_mcp_context_manager",
    
    # Provider availability
    "CLAUDE_AVAILABLE",
    "OPENAI_AVAILABLE"
]

# Add providers to __all__ if available
if CLAUDE_AVAILABLE:
    __all__.append("ClaudeProvider")

if OPENAI_AVAILABLE:
    __all__.append("OpenAIProvider")
