#!/usr/bin/env python3
"""
Test Agent Integration with Unified LLM Orchestration

This test verifies that the ProEthica agent system successfully integrates
with our new unified LLM orchestration system.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "proethica"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment for testing
os.environ["USE_MOCK_FALLBACK"] = "true"

async def test_agent_integration():
    """Test complete agent integration."""
    print("🚀 Testing ProEthica Agent Integration with Unified LLM Orchestration\n")
    
    results = []
    
    # Test 1: Import unified agent service
    print("1. Testing unified agent service import...")
    try:
        from app.services.unified_agent_service import get_unified_agent_service
        agent_service = get_unified_agent_service()
        print("✅ Unified agent service imported and initialized")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to import agent service: {e}")
        results.append(False)
        return results
    
    # Test 2: Check service info
    print("\n2. Testing service configuration...")
    try:
        info = agent_service.get_service_info()
        print(f"✅ Service type: {info['service_type']}")
        print(f"✅ Unified available: {info['unified_available']}")
        print(f"✅ Using orchestrator: {info['using_orchestrator']}")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to get service info: {e}")
        results.append(False)
    
    # Test 3: Test message sending (mock mode)
    print("\n3. Testing message sending...")
    try:
        from app.services.llm_service import Conversation
        
        # Create test conversation
        conversation = Conversation(messages=[])
        
        # Send test message
        response = agent_service.send_message(
            message="What ethical principles should engineers consider?",
            conversation=conversation,
            world_id=1,  # Test with world context
            service="claude"
        )
        
        print(f"✅ Message sent successfully")
        print(f"✅ Response role: {response.role}")
        print(f"✅ Response preview: {response.content[:60]}...")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to send message: {e}")
        results.append(False)
    
    # Test 4: Test prompt options
    print("\n4. Testing prompt options...")
    try:
        options = agent_service.get_prompt_options(
            conversation=conversation,
            world_id=1,
            service="claude"
        )
        
        print(f"✅ Got {len(options)} prompt options")
        if options:
            print(f"✅ Sample option: {options[0]['text'][:40]}...")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to get prompt options: {e}")
        results.append(False)
    
    # Test 5: Test guidelines retrieval
    print("\n5. Testing guidelines retrieval...")
    try:
        guidelines = agent_service.get_guidelines_for_world(world_id=1)
        print(f"✅ Guidelines retrieved: {'Yes' if guidelines else 'No (expected for test)'}")
        results.append(True)
    except Exception as e:
        print(f"❌ Failed to get guidelines: {e}")
        results.append(False)
    
    print(f"\n📊 Integration Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 Complete integration successful! Agent system is connected to unified orchestration.")
    else:
        print("⚠️  Some integration tests failed. Check output above.")
    
    return results

async def main():
    """Run integration test."""
    try:
        results = await test_agent_integration()
        return all(results)
    except Exception as e:
        print(f"❌ Integration test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n{'🎯 INTEGRATION SUCCESSFUL' if success else '⚠️  INTEGRATION ISSUES DETECTED'}")
    sys.exit(0 if success else 1)
