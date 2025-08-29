"""
Tests for the research LLM service.

Tests both unit functionality and integration with MCP services.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
import json

# Add the shared directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_orchestration.core.research_llm_service import (
    ResearchLLMService, 
    LLMRequest, 
    LLMResponse,
    MCPContextBuilder,
    PromptBuilder
)
from llm_orchestration.integrations.mcp_client import OntServeMCPClient, MCPClientManager

class TestLLMRequest:
    """Test LLM request data structures."""
    
    def test_llm_request_creation(self):
        """Test creating LLM requests."""
        request = LLMRequest(
            prompt="Test prompt",
            max_tokens=500,
            temperature=0.7,
            world_id=123,
            domain="test-domain",
            use_mcp_context=True
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 500
        assert request.temperature == 0.7
        assert request.world_id == 123
        assert request.domain == "test-domain"
        assert request.use_mcp_context == True
        
    def test_llm_request_defaults(self):
        """Test LLM request defaults."""
        request = LLMRequest(prompt="Test")
        
        assert request.system_prompt is None
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
        assert request.world_id is None
        assert request.domain is None
        assert request.conversation_id is None
        assert request.use_mcp_context == True
        assert request.metadata == {}

class TestMCPContextBuilder:
    """Test MCP context building functionality."""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client for testing."""
        client = AsyncMock()
        client.call_tool = AsyncMock()
        return client
    
    @pytest.fixture
    def context_builder(self, mock_mcp_client):
        """Context builder with mock client."""
        return MCPContextBuilder(mock_mcp_client)
    
    @pytest.mark.asyncio
    async def test_build_ontology_context_success(self, context_builder, mock_mcp_client):
        """Test successful context building."""
        # Setup mock responses
        mock_mcp_client.call_tool.side_effect = [
            {
                "entities": {
                    "obligation": [{"label": "Hold paramount public safety", "description": "Engineers must prioritize public safety"}],
                    "principle": [{"label": "Professional Competence", "description": "Act within areas of competence"}]
                }
            },
            {
                "name": "Professional Ethics",
                "description": "Engineering professional ethics domain",
                "entity_count": 94
            }
        ]
        
        # Test context building
        context = await context_builder.build_ontology_context(world_id=123)
        
        # Verify calls were made
        assert mock_mcp_client.call_tool.call_count == 2
        
        # Verify context structure
        assert "entities" in context
        assert "domain" in context
        assert len(context["entities"]["obligation"]) == 1
        assert len(context["entities"]["principle"]) == 1
        assert context["domain"]["name"] == "Professional Ethics"
    
    @pytest.mark.asyncio
    async def test_build_ontology_context_no_client(self):
        """Test context building without MCP client."""
        builder = MCPContextBuilder(None)
        context = await builder.build_ontology_context(world_id=123)
        
        assert context == {}
    
    @pytest.mark.asyncio
    async def test_build_ontology_context_error(self, context_builder, mock_mcp_client):
        """Test context building with MCP errors."""
        mock_mcp_client.call_tool.side_effect = Exception("MCP connection failed")
        
        context = await context_builder.build_ontology_context(world_id=123)
        
        assert context == {}

class TestPromptBuilder:
    """Test prompt building with MCP context."""
    
    @pytest.fixture
    def mock_context_builder(self):
        """Mock context builder."""
        builder = AsyncMock()
        builder.build_ontology_context = AsyncMock()
        return builder
    
    @pytest.fixture
    def prompt_builder(self, mock_context_builder):
        """Prompt builder with mock context builder."""
        return PromptBuilder(mock_context_builder)
    
    @pytest.mark.asyncio
    async def test_build_enriched_prompt_with_context(self, prompt_builder, mock_context_builder):
        """Test building enriched prompt with ontological context."""
        # Setup mock context
        mock_context_builder.build_ontology_context.return_value = {
            "domain": {
                "name": "Professional Ethics",
                "description": "Engineering ethics domain"
            },
            "entities": {
                "obligation": [
                    {"label": "Public Safety", "description": "Hold paramount public safety"}
                ],
                "principle": [
                    {"label": "Competence", "description": "Work within competence areas"}
                ]
            }
        }
        
        # Create request
        request = LLMRequest(
            prompt="What are the key obligations?",
            world_id=123,
            use_mcp_context=True
        )
        
        # Build enriched prompt
        enriched_prompt = await prompt_builder.build_enriched_prompt(request)
        
        # Verify enrichment
        assert "ONTOLOGICAL CONTEXT:" in enriched_prompt
        assert "Professional Ethics" in enriched_prompt
        assert "Public Safety" in enriched_prompt
        assert "What are the key obligations?" in enriched_prompt
        
    @pytest.mark.asyncio
    async def test_build_enriched_prompt_no_context(self, prompt_builder, mock_context_builder):
        """Test building prompt without MCP context."""
        request = LLMRequest(
            prompt="What are the key obligations?",
            use_mcp_context=False
        )
        
        enriched_prompt = await prompt_builder.build_enriched_prompt(request)
        
        # Should return original prompt unchanged
        assert enriched_prompt == "What are the key obligations?"
        mock_context_builder.build_ontology_context.assert_not_called()

class TestOntServeMCPClient:
    """Test MCP client functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock()
        session.post = AsyncMock()
        session.post.return_value.__aenter__ = AsyncMock(return_value=response)
        session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        session.get = AsyncMock()
        session.get.return_value.__aenter__ = AsyncMock(return_value=response)
        session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        return session, response
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_session):
        """Test successful MCP tool call."""
        session, response = mock_session
        
        # Setup response
        response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "text": {
                            "entities": [
                                {"label": "Public Safety", "type": "obligation"}
                            ]
                        }
                    }
                ]
            }
        }
        
        # Create client and patch session
        client = OntServeMCPClient("http://localhost:8082")
        client.session = session
        
        # Call tool
        result = await client.call_tool("get_entities_by_category", {"domain": "test"})
        
        # Verify result
        assert "entities" in result
        assert len(result["entities"]) == 1
        assert result["entities"][0]["label"] == "Public Safety"
        
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_session):
        """Test successful health check."""
        session, response = mock_session
        response.status = 200
        
        client = OntServeMCPClient("http://localhost:8082")
        client.session = session
        
        is_healthy = await client.health_check()
        
        assert is_healthy == True
        
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_session):
        """Test failed health check."""
        session, response = mock_session
        response.status = 500
        
        client = OntServeMCPClient("http://localhost:8082")
        client.session = session
        
        is_healthy = await client.health_check()
        
        assert is_healthy == False

class TestResearchLLMService:
    """Test the main research LLM service."""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client."""
        client = AsyncMock()
        client.call_tool = AsyncMock()
        return client
    
    def test_init_without_providers(self):
        """Test initialization when no providers are available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('llm_orchestration.core.research_llm_service.ANTHROPIC_AVAILABLE', False):
                with patch('llm_orchestration.core.research_llm_service.OPENAI_AVAILABLE', False):
                    service = ResearchLLMService()
                    assert len(service.providers) == 0
    
    def test_init_with_claude(self):
        """Test initialization with Claude provider."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('llm_orchestration.core.research_llm_service.ANTHROPIC_AVAILABLE', True):
                with patch('llm_orchestration.core.research_llm_service.Anthropic') as mock_anthropic:
                    service = ResearchLLMService()
                    assert 'claude' in service.providers
                    mock_anthropic.assert_called_with(api_key='test-key')
    
    @pytest.mark.asyncio
    async def test_generate_text_no_providers(self):
        """Test text generation when no providers are available."""
        service = ResearchLLMService()
        service.providers = {}  # Clear providers
        
        request = LLMRequest(prompt="Test prompt")
        
        with pytest.raises(RuntimeError, match="No LLM providers available"):
            await service.generate_text(request)
    
    @pytest.mark.asyncio
    async def test_select_provider_preference(self):
        """Test provider selection preference."""
        service = ResearchLLMService()
        
        # Test Claude preference
        service.providers = {'claude': Mock(), 'openai': Mock()}
        assert service._select_provider() == 'claude'
        
        # Test LangChain Claude as second choice
        service.providers = {'langchain_claude': Mock(), 'openai': Mock()}
        assert service._select_provider() == 'langchain_claude'
        
        # Test OpenAI as last choice
        service.providers = {'openai': Mock()}
        assert service._select_provider() == 'openai'
        
        # Test error when no providers
        service.providers = {}
        with pytest.raises(RuntimeError):
            service._select_provider()
    
    def test_get_provider_status(self, mock_mcp_client):
        """Test getting provider status."""
        service = ResearchLLMService(mcp_client=mock_mcp_client)
        service.providers = {'claude': Mock(), 'openai': Mock()}
        service.request_count = 5
        service.total_tokens = 1500
        
        status = service.get_provider_status()
        
        assert status['available_providers'] == ['claude', 'openai']
        assert status['request_count'] == 5
        assert status['total_tokens'] == 1500
        assert status['mcp_client_available'] == True

class TestIntegration:
    """Integration tests requiring actual or mock services."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock_workflow(self):
        """Test complete workflow with all mocked dependencies."""
        
        # Mock MCP client
        mock_mcp_client = AsyncMock()
        mock_mcp_client.call_tool.side_effect = [
            {
                "entities": {
                    "obligation": [{"label": "Public Safety", "description": "Hold paramount public safety"}]
                }
            },
            {
                "name": "Professional Ethics",
                "description": "Engineering ethics domain"
            }
        ]
        
        # Mock Claude provider
        mock_claude_provider = Mock()
        mock_message = Mock()
        mock_message.content = [Mock()]
        mock_message.content[0].text = "Engineers must prioritize public safety above all other considerations."
        mock_message.usage.input_tokens = 50
        mock_message.usage.output_tokens = 25
        mock_claude_provider.messages.create.return_value = mock_message
        
        # Create service with mocks
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key', 'CLAUDE_MODEL': 'claude-3-sonnet'}):
            with patch('llm_orchestration.core.research_llm_service.ANTHROPIC_AVAILABLE', True):
                with patch('llm_orchestration.core.research_llm_service.Anthropic', return_value=mock_claude_provider):
                    
                    service = ResearchLLMService(mcp_client=mock_mcp_client)
                    
                    # Create request
                    request = LLMRequest(
                        prompt="What are the key obligations for engineers?",
                        domain="proethica-intermediate",
                        use_mcp_context=True,
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    # Generate response
                    response = await service.generate_text(request)
                    
                    # Verify response
                    assert response.provider == 'claude'
                    assert response.model == 'claude-3-sonnet'
                    assert response.content == "Engineers must prioritize public safety above all other considerations."
                    assert response.tokens_used == 75
                    assert response.mcp_context_used == True
                    
                    # Verify MCP calls were made
                    assert mock_mcp_client.call_tool.call_count == 2
                    
                    # Verify Claude was called with enriched prompt
                    claude_call_args = mock_claude_provider.messages.create.call_args
                    assert claude_call_args[1]['model'] == 'claude-3-sonnet'
                    assert 'ONTOLOGICAL CONTEXT:' in claude_call_args[1]['messages'][0]['content']
                    assert 'Public Safety' in claude_call_args[1]['messages'][0]['content']

# Test runner and utilities
def run_tests():
    """Run all tests with proper async handling."""
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    run_tests()