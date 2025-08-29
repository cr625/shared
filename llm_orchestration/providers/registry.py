"""
Provider Registry

Manages multiple LLM providers with intelligent fallback, health monitoring,
and load balancing. Combines the provider management patterns from OntExtract
with the sophisticated orchestration needs identified in the analysis.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass

from .base_provider import BaseLLMProvider, GenerationRequest, GenerationResponse, ProviderStatus, MockLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a provider in the registry."""
    provider_class: Type[BaseLLMProvider]
    priority: int  # Lower number = higher priority
    config: Dict[str, Any]
    enabled: bool = True
    max_retries: int = 3
    timeout: float = 30.0


class ProviderRegistry:
    """
    Registry for managing multiple LLM providers with intelligent fallback.
    
    Features:
    - Priority-based provider selection
    - Automatic failover on errors
    - Health monitoring and circuit breaking
    - Load balancing and retry logic
    - Provider performance metrics
    """
    
    def __init__(self, 
                 enable_fallback: bool = True,
                 default_priority: Optional[List[str]] = None,
                 enable_mock_fallback: bool = True):
        """
        Initialize the provider registry.
        
        Args:
            enable_fallback: Enable automatic fallback between providers
            default_priority: Default provider priority list
            enable_mock_fallback: Enable mock provider as final fallback
        """
        self.enable_fallback = enable_fallback
        self.enable_mock_fallback = enable_mock_fallback
        
        # Provider storage
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
        
        # Get provider priority from environment or use defaults
        self.provider_priority = default_priority or self._get_default_priority()
        
        # Registry state
        self._initialized = False
        self._registry_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "provider_switches": 0
        }
        
        # Initialize the registry
        self._setup_registry()
    
    def _get_default_priority(self) -> List[str]:
        """Get default provider priority from environment variables."""
        # Check for explicit provider priority configuration
        priority_env = os.environ.get("LLM_PROVIDER_PRIORITY", "").strip()
        if priority_env:
            return [p.strip().lower() for p in priority_env.split(',')]
        
        # Check for legacy DEFAULT_LLM_PROVIDER
        default_provider = os.environ.get("DEFAULT_LLM_PROVIDER", "").strip().lower()
        if default_provider:
            # Normalize anthropic to claude
            if default_provider == "anthropic":
                default_provider = "claude"
            # Put default first, then other common providers
            other_providers = ["claude", "openai"]
            if default_provider in other_providers:
                other_providers.remove(default_provider)
            return [default_provider] + other_providers
        
        # Fall back to standard priority
        return ["claude", "openai"]
    
    def _setup_registry(self):
        """Set up the provider registry with available providers."""
        logger.info("Initializing LLM provider registry...")
        
        # Import providers here to avoid circular imports
        from .claude_provider import ClaudeProvider
        from .openai_provider import OpenAIProvider
        
        # Register Claude provider
        if "claude" in self.provider_priority:
            self.register_provider(
                "claude",
                ClaudeProvider,
                priority=self.provider_priority.index("claude"),
                config=self._get_claude_config()
            )
        
        # Register OpenAI provider
        if "openai" in self.provider_priority:
            self.register_provider(
                "openai",
                OpenAIProvider,
                priority=self.provider_priority.index("openai"),
                config=self._get_openai_config()
            )
        
        # Register mock provider as final fallback if enabled
        if self.enable_mock_fallback:
            self.register_provider(
                "mock",
                MockLLMProvider,
                priority=999,  # Lowest priority
                config={"model": "mock-unified-llm"}
            )
        
        # Initialize all providers
        self._initialize_providers()
        
        # Log registry status (skip async availability check during init)
        logger.info(f"Provider registry initialized with {len(self.providers)} providers")
        logger.info(f"Provider priority: {self.provider_priority}")
        # Note: Provider availability will be checked when they're actually used
        
        self._initialized = True
    
    def _get_claude_config(self) -> Dict[str, Any]:
        """Get Claude provider configuration."""
        return {
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "model": os.environ.get("CLAUDE_DEFAULT_MODEL", "claude-3-5-sonnet-20241022"),
            "api_base": os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"),
            "max_tokens": int(os.environ.get("CLAUDE_MAX_TOKENS", 4096)),
            "timeout": float(os.environ.get("CLAUDE_TIMEOUT", 30.0))
        }
    
    def _get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI provider configuration."""
        return {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model": os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", 4096)),
            "timeout": float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        }
    
    def register_provider(self, 
                         name: str,
                         provider_class: Type[BaseLLMProvider],
                         priority: int,
                         config: Dict[str, Any],
                         enabled: bool = True) -> None:
        """
        Register a provider in the registry.
        
        Args:
            name: Unique provider name
            provider_class: Provider class to instantiate
            priority: Provider priority (lower = higher priority)
            config: Provider configuration
            enabled: Whether provider is enabled
        """
        # Store provider configuration
        self.provider_configs[name] = ProviderConfig(
            provider_class=provider_class,
            priority=priority,
            config=config,
            enabled=enabled
        )
        
        logger.info(f"Registered provider: {name} (priority: {priority})")
    
    def _initialize_providers(self) -> None:
        """Initialize all registered providers."""
        for name, config in self.provider_configs.items():
            if not config.enabled:
                continue
                
            try:
                # Instantiate provider with configuration
                provider = config.provider_class(
                    provider_name=name,
                    **config.config
                )
                
                self.providers[name] = provider
                logger.info(f"Initialized provider: {name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {name}: {e}")
                continue
    
    async def get_available_providers(self) -> List[str]:
        """Get list of currently available providers in priority order."""
        available = []
        
        # Sort providers by priority
        sorted_providers = sorted(
            [(name, config.priority) for name, config in self.provider_configs.items() 
             if config.enabled and name in self.providers],
            key=lambda x: x[1]
        )
        
        # Check availability for each provider
        for name, _ in sorted_providers:
            provider = self.providers.get(name)
            if provider and await provider.is_available():
                available.append(name)
        
        return available
    
    async def select_provider(self, preferred_provider: Optional[str] = None) -> Optional[BaseLLMProvider]:
        """
        Select the best available provider.
        
        Args:
            preferred_provider: Specific provider to use (if available)
            
        Returns:
            Selected provider or None if none available
        """
        # Use preferred provider if specified and available
        if preferred_provider:
            provider = self.providers.get(preferred_provider)
            if provider and await provider.is_available():
                return provider
            elif preferred_provider != "mock":
                logger.warning(f"Preferred provider {preferred_provider} not available, falling back")
        
        # Get available providers in priority order
        available_providers = await self.get_available_providers()
        
        if not available_providers:
            logger.error("No LLM providers are available")
            return None
        
        # Return highest priority available provider
        selected_provider = available_providers[0]
        return self.providers[selected_provider]
    
    async def generate_text(self, 
                          request: GenerationRequest,
                          preferred_provider: Optional[str] = None,
                          max_retries: int = None) -> GenerationResponse:
        """
        Generate text using the best available provider with fallback.
        
        Args:
            request: Generation request
            preferred_provider: Preferred provider name
            max_retries: Maximum number of provider retries
            
        Returns:
            Generated response
            
        Raises:
            RuntimeError: If all providers fail
        """
        self._registry_stats["total_requests"] += 1
        
        if max_retries is None:
            max_retries = 3
        
        # Get available providers
        available_providers = await self.get_available_providers()
        if not available_providers:
            self._registry_stats["failed_requests"] += 1
            raise RuntimeError("No LLM providers are available")
        
        # Try preferred provider first if specified
        if preferred_provider and preferred_provider in available_providers:
            available_providers.remove(preferred_provider)
            available_providers.insert(0, preferred_provider)
        
        last_error = None
        
        # Try each provider
        for provider_name in available_providers:
            if not self.enable_fallback and provider_name != available_providers[0]:
                break  # Only try first provider if fallback disabled
                
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            # Try this provider with retries
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempting generation with {provider_name} (attempt {attempt + 1})")
                    
                    response = await provider.generate_text(request)
                    
                    # Success! Update stats and return
                    self._registry_stats["successful_requests"] += 1
                    if provider_name != available_providers[0]:
                        self._registry_stats["provider_switches"] += 1
                    
                    return response
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Provider {provider_name} failed (attempt {attempt + 1}): {e}")
                    
                    # Wait before retrying same provider
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    
                    continue
            
            logger.error(f"Provider {provider_name} exhausted all retries")
        
        # All providers failed
        self._registry_stats["failed_requests"] += 1
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    
    async def send_message(self,
                          message: str,
                          conversation: Optional[Any] = None,
                          system_prompt: Optional[str] = None,
                          world_context: Optional[Dict[str, Any]] = None,
                          application_context: Optional[str] = None,
                          preferred_provider: Optional[str] = None,
                          **kwargs) -> GenerationResponse:
        """
        High-level interface for sending messages with full context.
        
        This provides the same interface as ProEthica's ClaudeService while
        using the unified provider registry underneath.
        
        Args:
            message: User message to send
            conversation: Conversation history (optional)
            system_prompt: Base system prompt (optional)
            world_context: Ontology/world context from MCP (optional)
            application_context: Application-specific context (optional)
            preferred_provider: Preferred provider to use (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Import here to avoid circular imports
        from .base_provider import Conversation
        
        # Create conversation if not provided
        if conversation is None:
            conversation = Conversation(messages=[])
        
        # Build generation request
        request = GenerationRequest(
            prompt=message,
            conversation=conversation,
            system_prompt=system_prompt,
            world_context=world_context,
            application_context=application_context,
            **kwargs
        )
        
        # Generate response using provider registry
        return await self.generate_text(request, preferred_provider)
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers in the registry."""
        status = {
            "registry_stats": self._registry_stats.copy(),
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            status["providers"][name] = provider.get_provider_info()
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
        results = {}
        
        for name, provider in self.providers.items():
            try:
                provider_status = await provider.health_check()
                results[name] = {
                    "status": provider_status.value,
                    "available": provider_status == ProviderStatus.AVAILABLE,
                    "info": provider.get_provider_info()
                }
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "available": False,
                    "error": str(e)
                }
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        stats = {
            "registry": self._registry_stats.copy(),
            "providers": {
                name: provider.get_provider_info()
                for name, provider in self.providers.items()
            },
            "configuration": {
                "provider_priority": self.provider_priority,
                "fallback_enabled": self.enable_fallback,
                "mock_fallback_enabled": self.enable_mock_fallback,
                "registered_providers": list(self.provider_configs.keys()),
                "active_providers": list(self.providers.keys())
            }
        }
        
        # Calculate aggregate success rate
        total_requests = sum(p["total_requests"] for p in stats["providers"].values())
        successful_requests = sum(p["successful_requests"] for p in stats["providers"].values())
        stats["aggregate_success_rate"] = (successful_requests / max(total_requests, 1)) * 100
        
        return stats


# Singleton instance for global access
_provider_registry = None


def get_provider_registry(**kwargs) -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _provider_registry
    if _provider_registry is None:
        _provider_registry = ProviderRegistry(**kwargs)
    return _provider_registry


def reset_provider_registry():
    """Reset the global provider registry (for testing)."""
    global _provider_registry
    _provider_registry = None
