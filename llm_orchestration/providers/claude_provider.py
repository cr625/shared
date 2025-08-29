"""
Claude LLM Provider

Anthropic Claude provider implementation for the unified LLM orchestration system.
Integrates patterns from ProEthica's ClaudeService with the new unified interface.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, Any, Optional

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base_provider import BaseLLMProvider, GenerationRequest, GenerationResponse, ProviderStatus

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """
    Anthropic Claude provider implementation.
    
    Features:
    - Full Claude API integration
    - System prompt support
    - Conversation history management
    - World context integration (from ProEthica pattern)
    - Automatic fallback to mock mode
    """
    
    def __init__(self, 
                 provider_name: str = "claude",
                 model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None,
                 api_base: str = "https://api.anthropic.com/v1",
                 max_tokens: int = 4096,
                 timeout: float = 30.0,
                 **config):
        """
        Initialize Claude provider.
        
        Args:
            provider_name: Provider name
            model: Claude model to use
            api_key: Anthropic API key
            api_base: API base URL
            max_tokens: Maximum tokens per request
            timeout: Request timeout in seconds
            **config: Additional configuration
        """
        # Extract these from config if they're passed that way
        self.api_base = config.pop('api_base', api_base)
        self.max_tokens = config.pop('max_tokens', max_tokens)
        self.timeout = config.pop('timeout', timeout)
        
        # Call parent constructor
        super().__init__(provider_name, model, api_key, **config)
        
        # Claude client (will be None if not available)
        self.client: Optional[Anthropic] = None
        
        # Default system prompt (from ProEthica pattern)
        self.default_system_prompt = """
        You are an AI assistant helping users with ethical decision-making scenarios.
        Provide thoughtful, nuanced responses that consider multiple ethical perspectives.
        When appropriate, reference relevant ethical frameworks and principles.
        """
    
    def _initialize(self) -> None:
        """Initialize Claude client and check availability."""
        # Check if anthropic package is available
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic package not available, Claude provider will use mock mode")
            self._status = ProviderStatus.UNAVAILABLE
            return
        
        # Check for API key
        if not self.api_key or self.api_key.startswith("your-") or len(self.api_key) < 20:
            logger.warning("No valid Claude API key found, provider will be unavailable")
            self._status = ProviderStatus.UNAVAILABLE
            return
        
        # Initialize the Anthropic client
        try:
            # Set environment variable for SDK
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
            
            # Initialize client
            self.client = Anthropic(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"Claude provider initialized with model {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            self._status = ProviderStatus.ERROR
            self.client = None
    
    async def health_check(self) -> ProviderStatus:
        """Check Claude API health."""
        if not ANTHROPIC_AVAILABLE or not self.client:
            return ProviderStatus.UNAVAILABLE
        
        try:
            # Make a minimal request to test connectivity
            test_request = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "system": "Respond with just 'Hi'"
            }
            
            # Run in thread pool to avoid blocking
            def make_request():
                return self.client.messages.create(**test_request)
            
            # Use asyncio to run the synchronous call
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, make_request),
                timeout=5.0  # Quick health check
            )
            
            return ProviderStatus.AVAILABLE
            
        except asyncio.TimeoutError:
            logger.warning("Claude health check timed out")
            return ProviderStatus.DEGRADED
        except Exception as e:
            logger.error(f"Claude health check failed: {e}")
            return ProviderStatus.ERROR
    
    def _build_system_prompt_with_context(self, 
                                        request: GenerationRequest) -> str:
        """
        Build system prompt with world context.
        
        Replicates ProEthica's system prompt building pattern.
        """
        # Start with provided system prompt or default
        system_prompt = request.system_prompt or self.default_system_prompt
        
        # Add world context if available
        if request.world_context:
            system_prompt += "\n\nIMPORTANT: You are operating within a specific ethical context."
            system_prompt += "\nYou must constrain your responses to be relevant to this context."
            
            # Add guidelines if available
            if "guidelines" in request.world_context:
                system_prompt += f"\n\nGuidelines for this context:\n{request.world_context['guidelines']}"
            
            # Add key entities/concepts if available
            if "entities" in request.world_context and request.world_context["entities"]:
                entity_list = list(request.world_context["entities"].keys())[:15]  # Limit for efficiency
                if entity_list:
                    system_prompt += "\n\nKey concepts in this context:"
                    for entity in entity_list:
                        # Clean up URI format
                        entity_name = entity.split('/')[-1].replace('_', ' ')
                        system_prompt += f"\n- {entity_name}"
        
        # Add application context if available
        if request.application_context:
            system_prompt += f"\n\nAPPLICATION CONTEXT:\n{request.application_context}"
        
        return system_prompt
    
    def _format_messages_for_claude(self, request: GenerationRequest) -> list:
        """
        Format conversation messages for Claude API.
        
        Based on ProEthica's _format_messages_for_claude method.
        """
        claude_messages = []
        
        # Add conversation history if available
        if request.conversation:
            for message in request.conversation.messages:
                # Map standard roles to Claude format
                if message.role in ['user', 'assistant']:
                    claude_messages.append({
                        "role": message.role,
                        "content": message.content
                    })
                else:
                    # Handle other roles as user messages with prefix
                    claude_messages.append({
                        "role": "user",
                        "content": f"[{message.role}]: {message.content}"
                    })
        
        # Add current prompt if not already in conversation
        if not any(m.get('content') == request.prompt and m.get('role') == 'user' 
                  for m in claude_messages):
            claude_messages.append({
                "role": "user",
                "content": request.prompt
            })
        
        return claude_messages
    
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using Claude API.
        
        Args:
            request: Generation request with all parameters
            
        Returns:
            Generated response
            
        Raises:
            Exception: If generation fails
        """
        if not self.client:
            raise RuntimeError("Claude client not available")
        
        start_time = time.time()
        
        try:
            # Build system prompt with context
            system_prompt = self._build_system_prompt_with_context(request)
            
            # Format messages for Claude
            claude_messages = self._format_messages_for_claude(request)
            
            # Prepare request parameters
            claude_params = {
                "model": self.model,
                "system": system_prompt,
                "messages": claude_messages,
                "max_tokens": min(request.max_tokens, self.max_tokens),
                "temperature": request.temperature
            }
            
            # Add provider-specific parameters if provided
            if request.provider_specific:
                claude_params.update(request.provider_specific)
            
            # Make the API call in a thread pool
            def make_request():
                return self.client.messages.create(**claude_params)
            
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, make_request),
                timeout=self.timeout
            )
            
            # Extract content from response
            content = response.content[0].text
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get token usage if available
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return GenerationResponse(
                content=content,
                provider=self.provider_name,
                model=self.model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                cached=False,
                metadata={
                    "system_prompt_length": len(system_prompt),
                    "message_count": len(claude_messages),
                    "has_world_context": bool(request.world_context),
                    "response_usage": response.usage._asdict() if hasattr(response, 'usage') and response.usage else None
                }
            )
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Claude API request timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise RuntimeError(f"Claude API error: {str(e)}")
    
    def get_prompt_options(self, 
                          conversation: Optional[Any] = None,
                          world_context: Optional[Dict[str, Any]] = None) -> list:
        """
        Generate suggested prompt options based on conversation and world context.
        
        This replicates ProEthica's get_prompt_options method but uses the unified interface.
        For now, returns static options. Could be enhanced to use Claude for dynamic generation.
        """
        # Default options for no conversation
        if not conversation or len(conversation.messages) == 0:
            if world_context and "world_name" in world_context:
                world_name = world_context["world_name"]
                return [
                    {"id": 1, "text": f"Tell me more about the {world_name} context"},
                    {"id": 2, "text": f"What ethical principles apply in {world_name}?"},
                    {"id": 3, "text": f"How should I approach decisions in {world_name}?"}
                ]
            else:
                return [
                    {"id": 1, "text": "Tell me more about ethical decision-making"},
                    {"id": 2, "text": "What are some key ethical frameworks?"},
                    {"id": 3, "text": "How can I apply ethics to everyday decisions?"}
                ]
        
        # Context-aware options for ongoing conversation
        return [
            {"id": 1, "text": "Can you elaborate on that point?"},
            {"id": 2, "text": "What are the potential consequences?"},
            {"id": 3, "text": "How do other ethical frameworks view this?"},
            {"id": 4, "text": "What would you recommend in this situation?"}
        ]


# For backward compatibility with ProEthica imports
ClaudeService = ClaudeProvider
