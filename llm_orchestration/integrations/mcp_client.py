"""
MCP client for OntServe tool integration.

Provides clean interface for accessing OntServe MCP tools from the 
unified LLM service for ontological context enrichment.
"""

import json
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OntServeMCPClient:
    """
    Client for accessing OntServe MCP tools.
    
    Provides methods for retrieving ontological context to enrich
    LLM prompts with domain-specific information.
    """
    
    def __init__(self, server_url: str = None):
        """
        Initialize MCP client.
        
        Args:
            server_url: OntServe MCP server URL
        """
        self.server_url = server_url or "http://localhost:8082"
        self.session = None
        self.request_id = 0
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool on the OntServe server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool result dictionary
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        self.request_id += 1
        
        # Construct JSON-RPC 2.0 request
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            async with self.session.post(
                self.server_url,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    logger.error(f"MCP request failed with status {response.status}")
                    return {}
                    
                result = await response.json()
                
                if "error" in result:
                    logger.error(f"MCP tool error: {result['error']}")
                    return {}
                    
                return result.get("result", {}).get("content", [{}])[0].get("text", {})
                
        except Exception as e:
            logger.warning(f"MCP request failed: {e}")
            return {}
    
    async def get_entities_by_category(self, domain: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get entities categorized by type for a domain.
        
        Args:
            domain: Domain name or identifier
            
        Returns:
            Dictionary mapping categories to entity lists
        """
        result = await self.call_tool("get_entities_by_category", {"domain": domain})
        
        if not result or not isinstance(result, dict):
            return {}
            
        # Transform result into categorized structure
        categories = {}
        
        for entity in result.get("entities", []):
            entity_type = entity.get("type", "unknown")
            if entity_type not in categories:
                categories[entity_type] = []
            categories[entity_type].append({
                "label": entity.get("label"),
                "description": entity.get("description"),
                "uri": entity.get("uri")
            })
            
        return categories
    
    async def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """
        Get information about a domain.
        
        Args:
            domain: Domain name or identifier
            
        Returns:
            Domain information dictionary
        """
        result = await self.call_tool("get_domain_info", {"domain": domain})
        
        if not result or not isinstance(result, dict):
            return {}
            
        return {
            "name": result.get("name"),
            "description": result.get("description"), 
            "entity_count": result.get("entity_count", 0),
            "categories": result.get("categories", [])
        }
    
    async def get_entity_definition(self, entity_uri: str) -> Dict[str, Any]:
        """
        Get detailed definition for a specific entity.
        
        Args:
            entity_uri: URI of the entity
            
        Returns:
            Entity definition dictionary
        """
        result = await self.call_tool("get_entity_definition", {"entity_uri": entity_uri})
        
        if not result or not isinstance(result, dict):
            return {}
            
        return {
            "label": result.get("label"),
            "description": result.get("description"),
            "type": result.get("type"),
            "properties": result.get("properties", []),
            "relationships": result.get("relationships", [])
        }
    
    async def find_related_entities(self, entity_uri: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_uri: URI of the source entity
            relationship_type: Type of relationship to search for (optional)
            
        Returns:
            List of related entities
        """
        arguments = {"entity_uri": entity_uri}
        if relationship_type:
            arguments["relationship_type"] = relationship_type
            
        result = await self.call_tool("find_related_entities", arguments)
        
        if not result or not isinstance(result, dict):
            return []
            
        return result.get("entities", [])
    
    async def natural_language_query(self, query: str, domain: str = None) -> Dict[str, Any]:
        """
        Process a natural language query about ontology entities.
        
        Args:
            query: Natural language query
            domain: Domain to search in (optional)
            
        Returns:
            Query results dictionary
        """
        arguments = {"query": query}
        if domain:
            arguments["domain"] = domain
            
        result = await self.call_tool("natural_language_query", arguments)
        
        if not result or not isinstance(result, dict):
            return {}
            
        return {
            "interpretation": result.get("interpretation"),
            "entities": result.get("entities", []),
            "confidence": result.get("confidence", 0.0)
        }
    
    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            # Try to access the health endpoint
            async with self.session.get(f"{self.server_url}/health") as response:
                return response.status == 200
                
        except Exception as e:
            logger.debug(f"MCP health check failed: {e}")
            return False

class MCPClientManager:
    """
    Manager for MCP client instances with connection pooling.
    
    Provides singleton access and connection management for
    research applications.
    """
    
    _instance = None
    _client = None
    
    @classmethod 
    def get_instance(cls, server_url: str = None) -> OntServeMCPClient:
        """
        Get singleton MCP client instance.
        
        Args:
            server_url: OntServe MCP server URL
            
        Returns:
            MCP client instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._client = OntServeMCPClient(server_url)
            
        return cls._client
    
    @classmethod
    async def cleanup(cls):
        """Cleanup client connections."""
        if cls._client and cls._client.session:
            await cls._client.session.close()
            cls._client = None
            cls._instance = None