"""
Base LLM Provider Interface

Abstract base class and common utilities for all LLM providers in the unified
orchestration system. Defines the standard interface that all providers must
implement while allowing for provider-specific optimizations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncContextManager
from dataclasses import dataclass
from enum import Enum
import logging
import time
import asyncio

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class Message:
    """Standard message format for conversations."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Conversation:
    """Container for conversation history."""
    messages: List[Message]
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
    
    def add_message(self, content: str, role: str = "user", **metadata) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        return message
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent messages from the conversation."""
        return self.messages[-count:] if count > 0 else self.messages


@dataclass
class GenerationRequest:
    """Standard request format for text generation."""
    prompt: str
    conversation: Optional[Conversation] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    world_context: Optional[Dict[str, Any]] = None
    application_context: Optional[str] = None
    provider_specific: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Standard response format from providers."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Defines the standard interface that all providers must implement,
    including text generation, conversation management, and health monitoring.
    """
    
    def __init__(self, 
                 provider_name: str,
                 model: str,
                 api_key: Optional[str] = None,
                 **config):
        """
        Initialize the provider.
        
        Args:
            provider_name: Unique name for this provider
            model: Model identifier to use
            api_key: API key for the service
            **config: Provider-specific configuration
        """
        self.provider_name = provider_name
        self.model = model
        self.api_key = api_key
        self.config = config
        
        # Operational state
        self._status = ProviderStatus.UNAVAILABLE
        self._last_health_check = 0
        self._error_count = 0
        self._total_requests = 0
        self._successful_requests = 0
        
        # Initialize provider-specific components
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize provider-specific components."""
        pass
    
    @abstractmethod
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from the given request.
        
        Args:
            request: Generation request with prompt and parameters
            
        Returns:
            Generated response
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> ProviderStatus:
        """
        Check provider health and availability.
        
        Returns:
            Current provider status
        """
        pass
    
    async def is_available(self) -> bool:
        """Check if provider is currently available."""
        # Use cached status if recent
        if time.time() - self._last_health_check < 30:  # 30 second cache
            return self._status == ProviderStatus.AVAILABLE
        
        # Perform health check
        try:
            self._status = await self.health_check()
            self._last_health_check = time.time()
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_name}: {e}")
            self._status = ProviderStatus.ERROR
        
        return self._status == ProviderStatus.AVAILABLE
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this provider."""
        return {
            "provider_name": self.provider_name,
            "model": self.model,
            "status": self._status.value,
            "error_count": self._error_count,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "success_rate": (self._successful_requests / max(self._total_requests, 1)) * 100,
            "last_health_check": self._last_health_check
        }
    
    async def send_message_with_conversation(self, 
                                           message: str,
                                           conversation: Optional[Conversation] = None,
                                           system_prompt: Optional[str] = None,
                                           world_context: Optional[Dict[str, Any]] = None,
                                           **kwargs) -> GenerationResponse:
        """
        Send a message with full conversation context.
        
        This is a higher-level interface that wraps generate_text with
        conversation management, similar to ProEthica's pattern.
        
        Args:
            message: User message to send
            conversation: Existing conversation (will create if None)
            system_prompt: System prompt to use
            world_context: Contextual information from MCP/ontology
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Create conversation if not provided
        if conversation is None:
            conversation = Conversation(messages=[])
        
        # Add user message to conversation
        conversation.add_message(message, role="user")
        
        # Build generation request
        request = GenerationRequest(
            prompt=message,
            conversation=conversation,
            system_prompt=system_prompt,
            world_context=world_context,
            **kwargs
        )
        
        # Generate response
        start_time = time.time()
        try:
            self._total_requests += 1
            response = await self.generate_text(request)
            
            # Add assistant response to conversation
            conversation.add_message(response.content, role="assistant")
            
            # Update metrics
            self._successful_requests += 1
            response.processing_time = time.time() - start_time
            
            return response
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Generation failed for {self.provider_name}: {e}")
            raise
    
    def format_conversation_for_provider(self, conversation: Conversation) -> List[Dict[str, str]]:
        """
        Convert conversation to provider-specific format.
        
        Default implementation returns standard chat format.
        Providers can override for specific requirements.
        
        Args:
            conversation: Conversation to format
            
        Returns:
            List of message dictionaries
        """
        formatted_messages = []
        
        for message in conversation.messages:
            # Most providers support user/assistant roles
            if message.role in ["user", "assistant"]:
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            else:
                # Handle other roles by prefixing content
                formatted_messages.append({
                    "role": "user",
                    "content": f"[{message.role}]: {message.content}"
                })
        
        return formatted_messages
    
    def build_system_prompt(self, 
                           base_prompt: Optional[str] = None,
                           world_context: Optional[Dict[str, Any]] = None,
                           application_context: Optional[str] = None) -> str:
        """
        Build a comprehensive system prompt.
        
        Combines base prompt with contextual information from MCP and application state.
        This replicates ProEthica's system prompt building pattern.
        
        Args:
            base_prompt: Base system prompt
            world_context: Context from MCP/ontology services
            application_context: Additional application-specific context
            
        Returns:
            Complete system prompt
        """
        if base_prompt is None:
            base_prompt = """You are an AI assistant helping with ethical decision-making scenarios.
Provide thoughtful, nuanced responses that consider multiple ethical perspectives.
When appropriate, reference relevant ethical frameworks and principles."""
        
        system_prompt = base_prompt
        
        # Add world context if available
        if world_context:
            system_prompt += "\n\nIMPORTANT: You are operating within a specific ethical context."
            system_prompt += "\nYou must constrain your responses to be relevant to this context."
            
            # Add guidelines if available
            if "guidelines" in world_context:
                system_prompt += f"\n\nGuidelines:\n{world_context['guidelines']}"
            
            # Add key entities/concepts if available
            if "entities" in world_context and world_context["entities"]:
                entity_list = list(world_context["entities"].keys())[:15]  # Limit for token efficiency
                if entity_list:
                    system_prompt += "\n\nKey concepts in this context:"
                    for entity in entity_list:
                        entity_name = entity.split('/')[-1].replace('_', ' ')
                        system_prompt += f"\n- {entity_name}"
        
        # Add application context if available
        if application_context:
            system_prompt += f"\n\nAPPLICATION CONTEXT:\n{application_context}"
        
        return system_prompt


class MockLLMProvider(BaseLLMProvider):
    """
    Mock provider for development and testing.
    
    Provides realistic responses without requiring API keys,
    similar to ProEthica's mock fallback system.
    """
    
    def __init__(self, 
                 provider_name: str = "mock",
                 model: str = "mock-model", 
                 **config):
        super().__init__(
            provider_name=provider_name,
            model=model,
            api_key=None,
            **config
        )
    
    def _initialize(self) -> None:
        """Initialize mock provider."""
        self._status = ProviderStatus.AVAILABLE
    
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate mock response."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        prompt = request.prompt
        content = f"This is a mock response to: '{prompt[:50]}...'"
        
        # Add context-aware elements
        if request.world_context:
            content += f"\n\nI'm responding in a specific ethical context."
        
        if request.conversation and len(request.conversation.messages) > 1:
            content += f"\n\nBased on our conversation history of {len(request.conversation.messages)} messages."
        
        content += "\n\n[Note: This is a mock response. Enable API access for real responses.]"
        
        return GenerationResponse(
            content=content,
            provider=self.provider_name,
            model=self.model,
            tokens_used=len(content.split()),
            processing_time=0.1,
            cached=False,
            metadata={"mock": True}
        )
    
    async def health_check(self) -> ProviderStatus:
        """Mock provider is always available."""
        return ProviderStatus.AVAILABLE
