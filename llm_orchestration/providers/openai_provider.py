"""
OpenAI LLM Provider

OpenAI provider implementation for the unified LLM orchestration system.
Integrates patterns from OntExtract's multi-provider architecture with the new unified interface.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, Any, Optional, List

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_provider import BaseLLMProvider, GenerationRequest, GenerationResponse, ProviderStatus, Conversation

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation.
    
    Features:
    - Full OpenAI API integration (GPT-4, GPT-3.5, etc.)
    - System prompt support
    - Conversation history management
    - World context integration
    - Automatic fallback handling
    """
    
    def __init__(self, 
                 provider_name: str = "openai",
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 api_base: str = "https://api.openai.com/v1",
                 max_tokens: int = 4096,
                 timeout: float = 30.0,
                 **config):
        """
        Initialize OpenAI provider.
        
        Args:
            provider_name: Provider name
            model: OpenAI model to use
            api_key: OpenAI API key
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
        
        # OpenAI client (will be None if not available)
        self.client: Optional[openai.AsyncOpenAI] = None
        
        # Default system prompt
        self.default_system_prompt = """
        You are an AI assistant helping users with ethical decision-making scenarios.
        Provide thoughtful, nuanced responses that consider multiple ethical perspectives.
        When appropriate, reference relevant ethical frameworks and principles.
        """
    
    def _initialize(self) -> None:
        """Initialize OpenAI client and check availability."""
        # Check if openai package is available
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available, OpenAI provider will be unavailable")
            self._status = ProviderStatus.UNAVAILABLE
            return
        
        # Check for API key
        if not self.api_key or self.api_key.startswith("your-") or len(self.api_key) < 20:
            logger.warning("No valid OpenAI API key found, provider will be unavailable")
            self._status = ProviderStatus.UNAVAILABLE
            return
        
        # Initialize the OpenAI client
        try:
            # Set environment variable for SDK
            os.environ["OPENAI_API_KEY"] = self.api_key
            
            # Initialize async client
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"OpenAI provider initialized with model {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._status = ProviderStatus.ERROR
            self.client = None
    
    async def health_check(self) -> ProviderStatus:
        """Check OpenAI API health."""
        if not OPENAI_AVAILABLE or not self.client:
            return ProviderStatus.UNAVAILABLE
        
        try:
            # Make a minimal request to test connectivity
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Respond with just 'Hi'"},
                        {"role": "user", "content": "Hello"}
                    ],
                    max_tokens=5,
                    temperature=0
                ),
                timeout=5.0  # Quick health check
            )
            
            return ProviderStatus.AVAILABLE
            
        except asyncio.TimeoutError:
            logger.warning("OpenAI health check timed out")
            return ProviderStatus.DEGRADED
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return ProviderStatus.ERROR
    
    def _build_system_message_with_context(self, request: GenerationRequest) -> str:
        """
        Build system message with world context.
        
        Similar to Claude's system prompt building but adapted for OpenAI's format.
        """
        # Start with provided system prompt or default
        system_content = request.system_prompt or self.default_system_prompt
        
        # Add world context if available
        if request.world_context:
            system_content += "\n\nIMPORTANT: You are operating within a specific ethical context."
            system_content += "\nYou must constrain your responses to be relevant to this context."
            
            # Add guidelines if available
            if "guidelines" in request.world_context:
                system_content += f"\n\nGuidelines for this context:\n{request.world_context['guidelines']}"
            
            # Add key entities/concepts if available
            if "entities" in request.world_context and request.world_context["entities"]:
                entity_list = list(request.world_context["entities"].keys())[:15]  # Limit for efficiency
                if entity_list:
                    system_content += "\n\nKey concepts in this context:"
                    for entity in entity_list:
                        # Clean up URI format
                        entity_name = entity.split('/')[-1].replace('_', ' ')
                        system_content += f"\n- {entity_name}"
        
        # Add application context if available
        if request.application_context:
            system_content += f"\n\nAPPLICATION CONTEXT:\n{request.application_context}"
        
        return system_content
    
    def _format_messages_for_openai(self, request: GenerationRequest) -> List[Dict[str, str]]:
        """
        Format conversation messages for OpenAI API.
        
        OpenAI uses a different format than Claude - system messages are part of the messages array.
        """
        openai_messages = []
        
        # Add system message first
        system_content = self._build_system_message_with_context(request)
        openai_messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history if available
        if request.conversation:
            for message in request.conversation.messages:
                # Map standard roles to OpenAI format
                if message.role in ['user', 'assistant', 'system']:
                    openai_messages.append({
                        "role": message.role,
                        "content": message.content
                    })
                else:
                    # Handle other roles as user messages with prefix
                    openai_messages.append({
                        "role": "user",
                        "content": f"[{message.role}]: {message.content}"
                    })
        
        # Add current prompt if not already in conversation
        if not any(m.get('content') == request.prompt and m.get('role') == 'user' 
                  for m in openai_messages):
            openai_messages.append({
                "role": "user",
                "content": request.prompt
            })
        
        return openai_messages
    
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using OpenAI API.
        
        Args:
            request: Generation request with all parameters
            
        Returns:
            Generated response
            
        Raises:
            Exception: If generation fails
        """
        if not self.client:
            raise RuntimeError("OpenAI client not available")
        
        start_time = time.time()
        
        try:
            # Format messages for OpenAI
            openai_messages = self._format_messages_for_openai(request)
            
            # Prepare request parameters
            openai_params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": min(request.max_tokens, self.max_tokens),
                "temperature": request.temperature
            }
            
            # Add provider-specific parameters if provided
            if request.provider_specific:
                # Filter out parameters that OpenAI doesn't support
                supported_params = {
                    'frequency_penalty', 'presence_penalty', 'top_p', 'stop',
                    'logit_bias', 'user', 'response_format', 'seed', 'tools', 'tool_choice'
                }
                for key, value in request.provider_specific.items():
                    if key in supported_params:
                        openai_params[key] = value
            
            # Make the API call
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**openai_params),
                timeout=self.timeout
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get token usage if available
            tokens_used = None
            if response.usage:
                tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
            
            return GenerationResponse(
                content=content,
                provider=self.provider_name,
                model=self.model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                cached=False,
                metadata={
                    "system_message_length": len(openai_messages[0]["content"]) if openai_messages else 0,
                    "message_count": len(openai_messages),
                    "has_world_context": bool(request.world_context),
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None
                }
            )
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"OpenAI API request timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def get_prompt_options(self, 
                          conversation: Optional[Conversation] = None,
                          world_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate suggested prompt options based on conversation and world context.
        
        Returns static options for now. Could be enhanced to use OpenAI for dynamic generation.
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
    
    def supports_function_calling(self) -> bool:
        """Check if this model supports function calling."""
        function_calling_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ]
        return any(model in self.model for model in function_calling_models)
    
    def supports_vision(self) -> bool:
        """Check if this model supports vision/image inputs."""
        vision_models = ["gpt-4-vision", "gpt-4o", "gpt-4-turbo"]
        return any(model in self.model for model in vision_models)


# For backward compatibility with OntExtract imports
OpenAILLMProvider = OpenAIProvider
