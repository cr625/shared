"""
LLM Orchestrator

Main orchestration class that coordinates multiple LLM providers with ontological context.
This is the primary interface for all LLM operations in the unified system.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time

from ..providers import (
    ProviderRegistry, get_provider_registry, reset_provider_registry,
    Conversation, Message, GenerationRequest, GenerationResponse
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the LLM orchestrator."""
    provider_priority: Optional[List[str]] = None
    enable_fallback: bool = True
    enable_mock_fallback: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_retries: int = 3
    default_model: Optional[str] = None
    mcp_server_url: Optional[str] = None


class LLMOrchestrator:
    """
    Main LLM orchestration class.
    
    This class provides a unified interface for LLM operations that:
    - Manages multiple providers with intelligent fallback
    - Integrates with MCP servers for ontological context
    - Handles conversation management and threading
    - Provides caching and performance optimization
    - Maintains compatibility with existing services (ProEthica, OntExtract)
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the LLM orchestrator.
        
        Args:
            config: Orchestrator configuration (uses defaults if None)
        """
        self.config = config or OrchestratorConfig()
        
        # Initialize provider registry
        self.provider_registry = get_provider_registry(
            enable_fallback=self.config.enable_fallback,
            default_priority=self.config.provider_priority,
            enable_mock_fallback=self.config.enable_mock_fallback
        )
        
        # MCP client (will be initialized when needed)
        self._mcp_client = None
        
        # Simple in-memory cache (can be replaced with Redis)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Orchestrator statistics
        self._stats = {
            "total_requests": 0,
            "cached_responses": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        
        logger.info("LLM Orchestrator initialized")
    
    @property
    def mcp_client(self):
        """Lazy-load MCP client when needed."""
        if self._mcp_client is None:
            try:
                # Import here to avoid circular dependencies
                from ..integrations.mcp_context import MCPContextManager
                mcp_url = self.config.mcp_server_url or os.environ.get("ONTSERVE_MCP_URL", "http://localhost:8082")
                self._mcp_client = MCPContextManager(mcp_url)
                logger.info(f"MCP client initialized: {mcp_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP client: {e}")
                self._mcp_client = None
        
        return self._mcp_client
    
    def _generate_cache_key(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           world_context: Optional[Dict[str, Any]] = None,
                           **kwargs) -> str:
        """Generate cache key for request."""
        import hashlib
        
        # Create hash from key components
        key_components = [
            prompt,
            system_prompt or "",
            str(world_context) if world_context else "",
            str(sorted(kwargs.items()))
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid."""
        if not self.config.enable_caching or cache_key not in self._cache:
            return False
        
        # Check TTL
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.config.cache_ttl
    
    def _cache_response(self, cache_key: str, response: GenerationResponse) -> None:
        """Cache a response."""
        if not self.config.enable_caching:
            return
        
        # Simple cache eviction - remove oldest entries if cache gets too large
        if len(self._cache) > 1000:
            # Remove oldest 10% of entries
            sorted_keys = sorted(self._cache_timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:100]]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        
        # Cache the response
        self._cache[cache_key] = {
            "content": response.content,
            "provider": response.provider,
            "model": response.model,
            "tokens_used": response.tokens_used,
            "metadata": response.metadata
        }
        self._cache_timestamps[cache_key] = time.time()
    
    async def get_world_context(self, world_id: Optional[Union[str, int]] = None,
                              world_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get world context from MCP server.
        
        Args:
            world_id: World ID to get context for
            world_name: World name to get context for
            
        Returns:
            Dictionary containing world context (guidelines, entities, etc.)
        """
        if not self.mcp_client:
            return {}
        
        try:
            context = {}
            
            # Get guidelines for the world
            if world_id or world_name:
                guidelines = await self.mcp_client.get_guidelines(world_id=world_id, world_name=world_name)
                if guidelines:
                    context["guidelines"] = guidelines
            
            # Get entities for the world
            if world_id:
                entities = await self.mcp_client.get_world_entities(world_id)
                if entities:
                    context["entities"] = entities
                    context["world_name"] = world_name or f"World {world_id}"
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to get world context: {e}")
            return {}
    
    async def send_message(self,
                          message: str,
                          conversation: Optional[Conversation] = None,
                          world_id: Optional[Union[str, int]] = None,
                          world_name: Optional[str] = None,
                          system_prompt: Optional[str] = None,
                          application_context: Optional[str] = None,
                          preferred_provider: Optional[str] = None,
                          temperature: float = 0.7,
                          max_tokens: int = 1024,
                          **kwargs) -> GenerationResponse:
        """
        Send a message with full context awareness.
        
        This is the primary interface for LLM interactions, providing:
        - Multi-provider support with fallback
        - Ontological context integration
        - Conversation management
        - Caching and performance optimization
        
        Args:
            message: User message to send
            conversation: Existing conversation (optional)
            world_id: World ID for context (optional)
            world_name: World name for context (optional)
            system_prompt: Base system prompt (optional)
            application_context: Application-specific context (optional)
            preferred_provider: Preferred provider to use (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated response
        """
        start_time = time.time()
        self._stats["total_requests"] += 1
        
        try:
            # Get world context if world is specified
            world_context = None
            if world_id or world_name:
                world_context = await self.get_world_context(world_id, world_name)
            
            # Check cache first
            cache_key = self._generate_cache_key(
                message, system_prompt, world_context, 
                temperature=temperature, max_tokens=max_tokens, **kwargs
            )
            
            if self._is_cache_valid(cache_key):
                cached_data = self._cache[cache_key]
                self._stats["cached_responses"] += 1
                
                return GenerationResponse(
                    content=cached_data["content"],
                    provider=cached_data["provider"],
                    model=cached_data["model"],
                    tokens_used=cached_data.get("tokens_used"),
                    processing_time=time.time() - start_time,
                    cached=True,
                    metadata=cached_data.get("metadata", {})
                )
            
            # Generate new response
            response = await self.provider_registry.send_message(
                message=message,
                conversation=conversation,
                system_prompt=system_prompt,
                world_context=world_context,
                application_context=application_context,
                preferred_provider=preferred_provider,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Cache the response
            self._cache_response(cache_key, response)
            
            # Update statistics
            self._stats["successful_requests"] += 1
            processing_time = time.time() - start_time
            
            # Update average response time (simple moving average)
            current_avg = self._stats["average_response_time"]
            total_successful = self._stats["successful_requests"]
            self._stats["average_response_time"] = (current_avg * (total_successful - 1) + processing_time) / total_successful
            
            return response
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Orchestrator request failed: {e}")
            raise
    
    async def send_message_with_conversation(self,
                                           message: str,
                                           conversation: Optional[Conversation] = None,
                                           world_id: Optional[Union[str, int]] = None,
                                           world_name: Optional[str] = None,
                                           **kwargs) -> GenerationResponse:
        """
        Send a message and automatically manage conversation history.
        
        This method replicates ProEthica's send_message_with_context interface.
        
        Args:
            message: User message
            conversation: Existing conversation (will create if None)
            world_id: World ID for context
            world_name: World name for context
            **kwargs: Additional parameters
            
        Returns:
            Generated response with conversation updated
        """
        # Create conversation if not provided
        if conversation is None:
            conversation = Conversation(messages=[])
        
        # Send message and get response
        response = await self.send_message(
            message=message,
            conversation=conversation,
            world_id=world_id,
            world_name=world_name,
            **kwargs
        )
        
        # Add user message to conversation if not already added
        if not any(m.content == message and m.role == "user" for m in conversation.messages[-1:]):
            conversation.add_message(message, role="user")
        
        # Add assistant response to conversation
        conversation.add_message(response.content, role="assistant", 
                               provider=response.provider, model=response.model)
        
        return response
    
    def get_prompt_options(self,
                          conversation: Optional[Conversation] = None,
                          world_id: Optional[Union[str, int]] = None,
                          world_name: Optional[str] = None,
                          preferred_provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get suggested prompt options based on conversation and world context.
        
        This replicates ProEthica's get_prompt_options interface.
        
        Args:
            conversation: Current conversation
            world_id: World ID for context
            world_name: World name for context
            preferred_provider: Preferred provider to use
            
        Returns:
            List of suggested prompt options
        """
        try:
            # Get world context synchronously (simplified)
            world_context = {}
            if world_name:
                world_context["world_name"] = world_name
            
            # Get provider to generate options
            available_providers = asyncio.run(self.provider_registry.get_available_providers())
            
            if preferred_provider and preferred_provider in available_providers:
                provider = self.provider_registry.providers.get(preferred_provider)
            elif available_providers:
                provider = self.provider_registry.providers.get(available_providers[0])
            else:
                # Fallback static options
                return [
                    {"id": 1, "text": "Tell me more about this topic"},
                    {"id": 2, "text": "What are the key considerations?"},
                    {"id": 3, "text": "How should I approach this decision?"}
                ]
            
            # Use provider's prompt option generation
            if provider and hasattr(provider, 'get_prompt_options'):
                return provider.get_prompt_options(conversation, world_context)
            else:
                # Fallback static options
                return [
                    {"id": 1, "text": "Can you elaborate on that?"},
                    {"id": 2, "text": "What are the implications?"},
                    {"id": 3, "text": "What would you recommend?"}
                ]
                
        except Exception as e:
            logger.error(f"Failed to get prompt options: {e}")
            return [
                {"id": 1, "text": "Tell me more"},
                {"id": 2, "text": "What should I consider?"},
                {"id": 3, "text": "How do I proceed?"}
            ]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status of orchestrator and all providers
        """
        health_data = {
            "orchestrator_status": "ok",
            "mcp_client_available": self.mcp_client is not None,
            "cache_enabled": self.config.enable_caching,
            "cache_size": len(self._cache),
            "statistics": self._stats.copy()
        }
        
        # Check provider registry health
        try:
            provider_health = await self.provider_registry.health_check()
            health_data["providers"] = provider_health
            health_data["available_providers"] = await self.provider_registry.get_available_providers()
        except Exception as e:
            health_data["provider_health_error"] = str(e)
        
        # Check MCP client health if available
        if self.mcp_client:
            try:
                mcp_health = await self.mcp_client.health_check()
                health_data["mcp_status"] = mcp_health
            except Exception as e:
                health_data["mcp_error"] = str(e)
        
        return health_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = self._stats.copy()
        
        # Add provider statistics
        provider_stats = self.provider_registry.get_registry_stats()
        stats["provider_registry"] = provider_stats
        
        # Calculate additional metrics
        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["cache_hit_rate"] = (stats["cached_responses"] / total_requests) * 100
            stats["success_rate"] = (stats["successful_requests"] / total_requests) * 100
            stats["failure_rate"] = (stats["failed_requests"] / total_requests) * 100
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Response cache cleared")
    
    def reset_statistics(self) -> None:
        """Reset orchestrator statistics."""
        self._stats = {
            "total_requests": 0,
            "cached_responses": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        logger.info("Orchestrator statistics reset")


# Global orchestrator instance for backward compatibility
_global_orchestrator = None


def get_llm_orchestrator(config: Optional[OrchestratorConfig] = None) -> LLMOrchestrator:
    """Get the global LLM orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = LLMOrchestrator(config)
    return _global_orchestrator


def reset_llm_orchestrator():
    """Reset the global orchestrator (for testing)."""
    global _global_orchestrator
    _global_orchestrator = None
    reset_provider_registry()
