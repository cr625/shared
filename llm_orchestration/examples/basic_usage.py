"""
Basic usage examples for the unified LLM orchestration service.

Demonstrates how to use the research-focused LLM service with MCP integration.
"""

import asyncio
import os
from shared.llm_orchestration import ResearchLLMService, LLMRequest, MCPClientManager

async def example_basic_text_generation():
    """Example: Basic text generation without MCP context."""
    
    # Initialize service without MCP client
    llm_service = ResearchLLMService()
    
    # Create request
    request = LLMRequest(
        prompt="Explain the concept of professional ethics in engineering",
        max_tokens=500,
        temperature=0.7,
        use_mcp_context=False  # Disable MCP context for this example
    )
    
    # Generate response
    response = await llm_service.generate_text(request)
    
    print("=== Basic Text Generation ===")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Content: {response.content}")
    print(f"Tokens: {response.tokens_used}")
    print()

async def example_mcp_enriched_generation():
    """Example: Text generation with MCP ontological context."""
    
    # Initialize MCP client 
    mcp_client = MCPClientManager.get_instance("http://localhost:8082")
    
    # Initialize service with MCP client
    llm_service = ResearchLLMService(mcp_client=mcp_client)
    
    # Create request with MCP context
    request = LLMRequest(
        prompt="What are the key ethical obligations for engineers?",
        system_prompt="You are an expert in professional engineering ethics.",
        max_tokens=600,
        temperature=0.5,
        domain="proethica-intermediate",  # Use ontology domain
        use_mcp_context=True
    )
    
    # Generate response with ontological enrichment
    response = await llm_service.generate_text(request)
    
    print("=== MCP-Enriched Generation ===")
    print(f"Provider: {response.provider}")
    print(f"MCP Context Used: {response.mcp_context_used}")
    print(f"Content: {response.content}")
    print()

async def example_structured_chain():
    """Example: Using LangChain structured workflows."""
    
    # Initialize service
    llm_service = ResearchLLMService()
    
    # Create structured chain for concept extraction
    chain = llm_service.create_structured_chain(
        template="""
        Extract {concept_type} from the following text about professional ethics:
        
        Text: {text}
        
        Return the extracted concepts in a numbered list format.
        """,
        input_variables=["concept_type", "text"]
    )
    
    # Run the chain
    result = await llm_service.run_chain(
        chain,
        concept_type="ethical principles",
        text="Engineers must hold paramount the safety, health, and welfare of the public. They must perform engineering work only in areas of their competence and maintain professional integrity."
    )
    
    print("=== Structured Chain Example ===")
    print(f"Extracted Concepts: {result}")
    print()

async def example_mcp_direct_queries():
    """Example: Direct MCP queries for ontological information."""
    
    # Initialize MCP client
    mcp_client = MCPClientManager.get_instance("http://localhost:8082")
    
    # Check server health
    is_healthy = await mcp_client.health_check()
    print(f"=== MCP Server Health ===")
    print(f"Server healthy: {is_healthy}")
    
    if not is_healthy:
        print("MCP server not available, skipping MCP examples")
        return
    
    # Get domain information
    domain_info = await mcp_client.get_domain_info("proethica-intermediate")
    print(f"\n=== Domain Information ===")
    print(f"Name: {domain_info.get('name')}")
    print(f"Description: {domain_info.get('description')}")
    print(f"Entity Count: {domain_info.get('entity_count')}")
    
    # Get entities by category
    entities = await mcp_client.get_entities_by_category("proethica-intermediate")
    print(f"\n=== Entities by Category ===")
    for category, entity_list in entities.items():
        print(f"{category}: {len(entity_list)} entities")
        # Show first few entities
        for entity in entity_list[:3]:
            print(f"  - {entity.get('label', 'Unknown')}")
        if len(entity_list) > 3:
            print(f"  ... and {len(entity_list) - 3} more")
    print()

async def example_complete_workflow():
    """Example: Complete workflow combining LLM generation with MCP context."""
    
    # Initialize with MCP
    mcp_client = MCPClientManager.get_instance("http://localhost:8082")
    llm_service = ResearchLLMService(mcp_client=mcp_client)
    
    # Step 1: Get domain context
    print("=== Complete Workflow Example ===")
    print("Step 1: Retrieving ontological context...")
    
    entities = await mcp_client.get_entities_by_category("proethica-intermediate")
    obligation_count = len(entities.get("obligation", []))
    principle_count = len(entities.get("principle", []))
    
    print(f"Found {obligation_count} obligations and {principle_count} principles")
    
    # Step 2: Generate analysis with context
    print("Step 2: Generating contextual analysis...")
    
    request = LLMRequest(
        prompt=f"""
        I'm analyzing a case where an engineer discovers a design flaw that could affect public safety, 
        but fixing it would delay the project significantly. The client is pressuring for on-time delivery.
        
        Please analyze this scenario considering the professional obligations and principles 
        available in the ontology.
        """,
        system_prompt="You are an expert in engineering ethics with access to comprehensive ethical frameworks.",
        max_tokens=800,
        temperature=0.3,
        domain="proethica-intermediate",
        use_mcp_context=True
    )
    
    response = await llm_service.generate_text(request)
    
    print(f"Analysis (using {response.provider}):")
    print(response.content)
    
    # Step 3: Get service metrics
    status = llm_service.get_provider_status()
    print(f"\nService Status:")
    print(f"Providers: {status['available_providers']}")
    print(f"Requests processed: {status['request_count']}")
    print(f"Total tokens: {status['total_tokens']}")

async def main():
    """Run all examples."""
    print("Starting LLM Orchestration Examples")
    print("=" * 50)
    
    # Set environment variables if not set
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Warning: ANTHROPIC_API_KEY not set. Some examples may not work.")
    
    try:
        # Run examples
        await example_basic_text_generation()
        await example_mcp_enriched_generation()
        await example_structured_chain()
        await example_mcp_direct_queries()
        await example_complete_workflow()
        
    except Exception as e:
        print(f"Example failed: {e}")
        
    finally:
        # Cleanup connections
        await MCPClientManager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())