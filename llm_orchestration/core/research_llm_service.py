"""
Research-focused unified LLM service with MCP tool integration.

This service provides clean LLM orchestration for research purposes with direct 
integration to OntServe MCP tools for ontological context enrichment.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Provider imports
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# LangChain imports for structured workflows
try:
    from langchain.prompts import PromptTemplate
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    """Request object for LLM operations."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    world_id: Optional[int] = None
    domain: Optional[str] = None
    conversation_id: Optional[str] = None
    use_mcp_context: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Response object from LLM operations."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    request_time: float = 0.0
    mcp_context_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPContextBuilder:
    """Builds context from MCP ontology tools."""
    
    def __init__(self, mcp_client=None):
        """Initialize with optional MCP client."""
        self.mcp_client = mcp_client
        
    async def build_ontology_context(self, world_id: Optional[int] = None, 
                                   domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Build ontological context using MCP tools.
        
        Args:
            world_id: World/domain ID for context
            domain: Domain name for context
            
        Returns:
            Dictionary containing ontological context
        """
        if not self.mcp_client:
            return {}
            
        context = {}
        
        try:
            # Get entities by category (primary MCP integration)
            if world_id or domain:
                # Get different types of entities
                entities = await self._get_entities_by_category(world_id or domain)
                if entities:
                    context['entities'] = entities
                    
                # Get domain information
                domain_info = await self._get_domain_info(world_id or domain)
                if domain_info:
                    context['domain'] = domain_info
                    
            return context
            
        except Exception as e:
            logger.warning(f"Failed to build MCP context: {e}")
            return {}
    
    async def _get_entities_by_category(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get entities by category from MCP service."""
        try:
            # Call MCP get_entities_by_category tool
            result = await self.mcp_client.call_tool(
                "get_entities_by_category",
                {"domain": str(identifier)}
            )
            return result
        except Exception as e:
            logger.debug(f"MCP entities retrieval failed: {e}")
            return {}
    
    async def _get_domain_info(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get domain information from MCP service."""
        try:
            # Call MCP get_domain_info tool
            result = await self.mcp_client.call_tool(
                "get_domain_info",
                {"domain": str(identifier)}
            )
            return result
        except Exception as e:
            logger.debug(f"MCP domain info retrieval failed: {e}")
            return {}

class PromptBuilder:
    """Builds enriched prompts with ontological context."""
    
    def __init__(self, mcp_context_builder: MCPContextBuilder):
        self.mcp_context_builder = mcp_context_builder
        
    async def build_enriched_prompt(self, request: LLMRequest) -> str:
        """
        Build prompt enriched with MCP ontological context.
        
        Args:
            request: LLM request object
            
        Returns:
            Enhanced prompt string
        """
        base_prompt = request.prompt
        
        # Skip MCP context if disabled
        if not request.use_mcp_context:
            return base_prompt
            
        # Get ontological context from MCP
        context = await self.mcp_context_builder.build_ontology_context(
            world_id=request.world_id,
            domain=request.domain
        )
        
        if not context:
            return base_prompt
            
        # Build enriched prompt with ontological context
        enriched_prompt = self._format_prompt_with_context(base_prompt, context)
        return enriched_prompt
    
    def _format_prompt_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        """Format prompt with ontological context."""
        context_parts = []
        
        # Add domain information
        if 'domain' in context:
            domain_info = context['domain']
            context_parts.append(f"Domain: {domain_info.get('name', 'Unknown')}")
            if domain_info.get('description'):
                context_parts.append(f"Description: {domain_info['description']}")
                
        # Add entity categories
        if 'entities' in context:
            entities = context['entities']
            
            # Format different entity types
            for category, items in entities.items():
                if items and isinstance(items, list):
                    context_parts.append(f"\n{category.title()}:")
                    for item in items[:10]:  # Limit to avoid token overflow
                        label = item.get('label', item.get('name', 'Unknown'))
                        description = item.get('description', '')
                        if description:
                            context_parts.append(f"  - {label}: {description}")
                        else:
                            context_parts.append(f"  - {label}")
        
        if not context_parts:
            return prompt
            
        # Construct enriched prompt
        context_text = '\n'.join(context_parts)
        enriched_prompt = f"""
ONTOLOGICAL CONTEXT:
{context_text}

USER REQUEST:
{prompt}

Please provide a response that takes into account the ontological context above.
"""
        
        return enriched_prompt

class ResearchLLMService:
    """
    Research-focused LLM service with MCP integration.
    
    Designed for research purposes with clean architecture and direct
    integration to OntServe MCP tools for ontological enrichment.
    """
    
    def __init__(self, mcp_client=None):
        """
        Initialize the research LLM service.
        
        Args:
            mcp_client: MCP client for ontology tool access
        """
        self.mcp_client = mcp_client
        self.mcp_context_builder = MCPContextBuilder(mcp_client)
        self.prompt_builder = PromptBuilder(self.mcp_context_builder)
        
        # Initialize providers
        self.providers = {}
        self._init_providers()
        
        # Track usage for research metrics
        self.request_count = 0
        self.total_tokens = 0
        
    def _init_providers(self):
        """Initialize available LLM providers."""
        # Claude/Anthropic provider
        if ANTHROPIC_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.providers['claude'] = Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                logger.info("Claude provider initialized")
            except Exception as e:
                logger.warning(f"Claude provider failed: {e}")
                
        # OpenAI provider
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                self.providers['openai'] = openai
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"OpenAI provider failed: {e}")
                
        # LangChain Claude for structured workflows
        if LANGCHAIN_AVAILABLE and os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.providers['langchain_claude'] = ChatAnthropic(
                    model=os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514'),
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                    temperature=0.2
                )
                logger.info("LangChain Claude provider initialized")
            except Exception as e:
                logger.warning(f"LangChain Claude provider failed: {e}")
        
        if not self.providers:
            logger.error("No LLM providers available - check API keys and installations")
    
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using the best available provider with MCP context.
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response object
        """
        if not self.providers:
            raise RuntimeError("No LLM providers available")
            
        # Build enriched prompt with MCP context
        start_time = asyncio.get_event_loop().time()
        enriched_prompt = await self.prompt_builder.build_enriched_prompt(request)
        
        # Determine provider (prefer Claude for research work)
        provider_name = self._select_provider()
        provider = self.providers[provider_name]
        
        try:
            # Generate text with selected provider
            if provider_name == 'claude':
                response = await self._generate_with_claude(provider, request, enriched_prompt)
            elif provider_name == 'openai':
                response = await self._generate_with_openai(provider, request, enriched_prompt)
            elif provider_name == 'langchain_claude':
                response = await self._generate_with_langchain_claude(provider, request, enriched_prompt)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
                
            # Update metrics
            self.request_count += 1
            self.total_tokens += response.tokens_used or 0
            
            # Mark if MCP context was used
            response.mcp_context_used = (enriched_prompt != request.prompt)
            response.request_time = asyncio.get_event_loop().time() - start_time
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed with {provider_name}: {e}")
            raise
    
    def _select_provider(self) -> str:
        """Select the best available provider."""
        # Prefer Claude for research work
        if 'claude' in self.providers:
            return 'claude'
        elif 'langchain_claude' in self.providers:
            return 'langchain_claude'
        elif 'openai' in self.providers:
            return 'openai'
        else:
            raise RuntimeError("No providers available")
    
    async def _generate_with_claude(self, provider, request: LLMRequest, prompt: str) -> LLMResponse:
        """Generate text using direct Claude API."""
        try:
            message = provider.messages.create(
                model=os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514'),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt or "You are a helpful research assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = message.content[0].text
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
            
            return LLMResponse(
                content=content,
                provider='claude',
                model=os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514'),
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise
    
    async def _generate_with_openai(self, provider, request: LLMRequest, prompt: str) -> LLMResponse:
        """Generate text using OpenAI API."""
        try:
            response = provider.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": request.system_prompt or "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=content,
                provider='openai', 
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def _generate_with_langchain_claude(self, provider, request: LLMRequest, prompt: str) -> LLMResponse:
        """Generate text using LangChain Claude integration."""
        try:
            # For structured workflows, we can use LangChain
            response = provider.invoke(prompt)
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            return LLMResponse(
                content=content,
                provider='langchain_claude',
                model=os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514'),
                tokens_used=None  # LangChain doesn't always provide token counts
            )
            
        except Exception as e:
            logger.error(f"LangChain Claude generation failed: {e}")
            raise
    
    def create_structured_chain(self, template: str, input_variables: List[str]):
        """
        Create a structured LangChain for complex workflows.
        
        Args:
            template: Prompt template string
            input_variables: List of input variable names
            
        Returns:
            LangChain chain object
        """
        if 'langchain_claude' not in self.providers:
            raise RuntimeError("LangChain Claude not available for structured chains")
            
        prompt_template = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        
        return prompt_template | self.providers['langchain_claude']
    
    async def run_chain(self, chain, **kwargs) -> str:
        """
        Run a structured chain with variables.
        
        Args:
            chain: LangChain chain object
            **kwargs: Variables for the chain
            
        Returns:
            Chain output content
        """
        try:
            response = chain.invoke(kwargs)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Chain execution failed: {e}")
            raise
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of available providers."""
        return {
            'available_providers': list(self.providers.keys()),
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'mcp_client_available': self.mcp_client is not None
        }