#!/usr/bin/env python3
"""
Basic functionality test for the unified LLM orchestration system.

This test verifies that the core components can be imported and initialized
without errors, and that the mock provider works correctly.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_imports():
    """Test that all core components can be imported."""
    print("Testing basic imports...")
    
    try:
        from llm_orchestration import (
            LLMOrchestrator, 
            OrchestratorConfig,
            get_llm_orchestrator,
            ProviderRegistry,
            get_provider_registry,
            MockLLMProvider,
            Conversation,
            Message,
            MCPContextManager
        )
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_mock_provider():
    """Test the mock provider functionality."""
    print("\nTesting mock provider...")
    
    try:
        from llm_orchestration import MockLLMProvider, GenerationRequest, Conversation
        
        # Create mock provider
        provider = MockLLMProvider()
        
        # Create test conversation
        conversation = Conversation(messages=[])
        conversation.add_message("Hello", role="user")
        
        # Create test request
        request = GenerationRequest(
            prompt="Test message",
            conversation=conversation,
            system_prompt="Test system prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        # Generate response
        response = await provider.generate_text(request)
        
        print(f"✅ Mock provider response: {response.content[:50]}...")
        print(f"✅ Provider: {response.provider}, Model: {response.model}")
        return True
        
    except Exception as e:
        print(f"❌ Mock provider test failed: {e}")
        return False


async def test_provider_registry():
    """Test the provider registry functionality."""
    print("\nTesting provider registry...")
    
    try:
        from llm_orchestration import get_provider_registry, reset_provider_registry
        
        # Reset registry for clean test
        reset_provider_registry()
        
        # Get registry (should initialize with mock provider)
        registry = get_provider_registry()
        
        # Check available providers
        available = await registry.get_available_providers()
        print(f"✅ Available providers: {available}")
        
        # Test health check
        health = await registry.health_check()
        print(f"✅ Registry health check: {len(health)} providers checked")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider registry test failed: {e}")
        return False


async def test_orchestrator():
    """Test the main orchestrator functionality."""
    print("\nTesting LLM orchestrator...")
    
    try:
        from llm_orchestration import LLMOrchestrator, OrchestratorConfig, Conversation
        
        # Create orchestrator config (with mock fallback enabled)
        config = OrchestratorConfig(
            enable_mock_fallback=True,
            enable_caching=True,
            mcp_server_url="http://localhost:8082"  # Will fail gracefully if not available
        )
        
        # Create orchestrator
        orchestrator = LLMOrchestrator(config)
        
        # Test basic message sending
        response = await orchestrator.send_message(
            message="Hello, this is a test message",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"✅ Orchestrator response: {response.content[:50]}...")
        print(f"✅ Response cached: {response.cached}")
        print(f"✅ Provider used: {response.provider}")
        
        # Test conversation management
        conversation = Conversation(messages=[])
        response2 = await orchestrator.send_message_with_conversation(
            message="Follow-up message",
            conversation=conversation
        )
        
        print(f"✅ Conversation has {len(conversation.messages)} messages")
        
        # Test health check
        health = await orchestrator.health_check()
        print(f"✅ Orchestrator health: {health['orchestrator_status']}")
        
        # Test statistics
        stats = orchestrator.get_statistics()
        print(f"✅ Total requests: {stats['total_requests']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False


async def test_mcp_context_manager():
    """Test the MCP context manager (will fail gracefully if server not available)."""
    print("\nTesting MCP context manager...")
    
    try:
        from llm_orchestration import MCPContextManager
        
        # Create MCP manager
        mcp_manager = MCPContextManager("http://localhost:8082")
        
        try:
            # Test health check (will fail if server not running)
            health = await mcp_manager.health_check()
            print(f"✅ MCP server health: {health.get('status', 'unknown')}")
            
            # Test tool listing
            tools = await mcp_manager.list_available_tools()
            print(f"✅ MCP tools available: {len(tools)}")
            
        except Exception as e:
            print(f"⚠️  MCP server not available (expected): {e}")
            print("✅ MCP context manager created successfully")
        
        # Clean up
        await mcp_manager.close()
        
        return True
        
    except Exception as e:
        print(f"❌ MCP context manager test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting unified LLM orchestration system tests...\n")
    
    tests = [
        test_basic_imports,
        test_mock_provider,
        test_provider_registry,
        test_orchestrator,
        test_mcp_context_manager
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! The unified LLM orchestration system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all(results)


if __name__ == "__main__":
    # Set environment variable to enable mock fallback
    os.environ["USE_MOCK_FALLBACK"] = "true"
    
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
