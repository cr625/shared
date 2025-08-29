"""
MCP Context Manager

Manages communication with MCP servers to provide rich ontological context
for LLM interactions. This is a key component for enabling LLMs to interact
directly with ontology servers for complex semantic information.
"""

import os
import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0


class MCPContextManager:
    """
    Manager for MCP server communication and context extraction.
    
    This class provides a high-level interface for LLMs to interact with
    MCP servers, particularly OntServe, to get rich ontological context
    for decision-making and reasoning.
    
    Features:
    - Async communication with MCP servers
    - Automatic retry and error handling
    - Context caching for performance
    - Health monitoring and circuit breaking
    - Support for multiple MCP server types
    """
    
    def __init__(self, 
                 server_url: str = "http://localhost:8082",
                 config: Optional[MCPServerConfig] = None):
        """
        Initialize MCP context manager.
        
        Args:
            server_url: URL of the MCP server (default OntServe)
            config: Server configuration (uses defaults if None)
        """
        self.server_url = server_url
        self.config = config or MCPServerConfig(url=server_url)
        
        # HTTP session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Request ID counter for JSON-RPC
        self._request_id = 0
        
        # Context cache
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Health status
        self._last_health_check = 0
        self._server_available = True
        
        logger.info(f"MCP Context Manager initialized for: {server_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                keepalive_timeout=30
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._session
    
    def _get_next_request_id(self) -> int:
        """Get next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id
    
    def _generate_cache_key(self, method: str, **params) -> str:
        """Generate cache key for request."""
        import hashlib
        key_data = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid."""
        if cache_key not in self._context_cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self._cache_ttl
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result."""
        # Simple cache eviction
        if len(self._context_cache) > 500:
            # Remove oldest 20% of entries
            sorted_keys = sorted(self._cache_timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:100]]
            for key in keys_to_remove:
                self._context_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        
        self._context_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
    
    async def _make_jsonrpc_request(self, 
                                   method: str, 
                                   params: Optional[Dict[str, Any]] = None,
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Make a JSON-RPC request to the MCP server.
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
            use_cache: Whether to use caching for this request
            
        Returns:
            Response data
            
        Raises:
            Exception: If request fails after retries
        """
        # Check cache first
        if use_cache and params:
            cache_key = self._generate_cache_key(method, **params)
            if self._is_cache_valid(cache_key):
                return self._context_cache[cache_key]
        
        # Prepare JSON-RPC request
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._get_next_request_id()
        }
        
        session = await self._get_session()
        last_error = None
        
        # Retry loop
        for attempt in range(self.config.max_retries):
            try:
                async with session.post(self.config.url, json=request_data) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        # Check for JSON-RPC error
                        if "error" in result_data:
                            error = result_data["error"]
                            raise Exception(f"MCP server error: {error.get('message', 'Unknown error')}")
                        
                        # Extract result
                        result = result_data.get("result", {})
                        
                        # Cache successful result
                        if use_cache and params:
                            self._cache_result(cache_key, result)
                        
                        self._server_available = True
                        return result
                    
                    else:
                        raise Exception(f"HTTP error: {response.status}")
            
            except Exception as e:
                last_error = e
                logger.warning(f"MCP request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        # All retries failed
        self._server_available = False
        raise Exception(f"MCP server request failed after {self.config.max_retries} attempts. Last error: {last_error}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check MCP server health.
        
        Returns:
            Health status information
        """
        try:
            # Use the /health endpoint if available, otherwise try a simple MCP method
            session = await self._get_session()
            
            # First try REST health endpoint
            try:
                health_url = f"{self.config.url.rstrip('/')}/health"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        self._server_available = True
                        self._last_health_check = time.time()
                        return health_data
            except:
                pass  # Fall back to JSON-RPC method
            
            # Fall back to JSON-RPC list_tools method
            result = await self._make_jsonrpc_request("list_tools", use_cache=False)
            
            self._server_available = True
            self._last_health_check = time.time()
            
            return {
                "status": "ok",
                "method": "jsonrpc",
                "tools_available": len(result.get("tools", [])),
                "server_url": self.config.url
            }
            
        except Exception as e:
            self._server_available = False
            logger.error(f"MCP health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "server_url": self.config.url
            }
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.
        
        Returns:
            List of tool definitions
        """
        try:
            result = await self._make_jsonrpc_request("list_tools")
            return result.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []
    
    async def get_entities_by_category(self, 
                                     category: str,
                                     domain_id: str = "engineering-ethics",
                                     status: str = "approved") -> Dict[str, Any]:
        """
        Get ontology entities by category from the MCP server.
        
        Args:
            category: Entity category (e.g., "Role", "Principle", "Obligation")
            domain_id: Professional domain
            status: Entity status filter
            
        Returns:
            Dictionary of entities
        """
        try:
            params = {
                "category": category,
                "domain_id": domain_id,
                "status": status
            }
            
            result = await self._make_jsonrpc_request(
                "call_tool",
                {
                    "name": "get_entities_by_category",
                    "arguments": params
                }
            )
            
            # Parse the result from MCP tool response format
            if "content" in result and result["content"]:
                content_text = result["content"][0].get("text", "{}")
                return json.loads(content_text)
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get entities by category: {e}")
            return {}
    
    async def get_world_entities(self, world_id: Union[str, int]) -> Dict[str, Any]:
        """
        Get entities for a specific world/domain.
        
        Args:
            world_id: World identifier
            
        Returns:
            Dictionary of world entities
        """
        # For now, map world to domain and get all entity categories
        domain_id = str(world_id) if isinstance(world_id, int) else world_id
        
        # Common entity categories in ethical ontologies
        categories = ["Role", "Principle", "Obligation", "Virtue", "Context", "Stakeholder"]
        
        all_entities = {}
        
        for category in categories:
            try:
                entities = await self.get_entities_by_category(
                    category=category,
                    domain_id=domain_id
                )
                if entities and "entities" in entities:
                    all_entities.update(entities["entities"])
            except Exception as e:
                logger.debug(f"Failed to get {category} entities: {e}")
                continue
        
        return {"entities": all_entities} if all_entities else {}
    
    async def execute_sparql_query(self, 
                                 query: str,
                                 domain_id: str = "engineering-ethics",
                                 reasoning: bool = False,
                                 format_type: str = "json",
                                 limit: int = 100) -> Dict[str, Any]:
        """
        Execute a SPARQL query on the ontology.
        
        Args:
            query: SPARQL query string
            domain_id: Professional domain
            reasoning: Enable OWL reasoning
            format_type: Output format
            limit: Maximum results
            
        Returns:
            Query results
        """
        try:
            params = {
                "query": query,
                "domain_id": domain_id,
                "reasoning": reasoning,
                "format": format_type,
                "limit": limit
            }
            
            result = await self._make_jsonrpc_request(
                "call_tool",
                {
                    "name": "sparql_query",
                    "arguments": params
                }
            )
            
            # Parse the result
            if "content" in result and result["content"]:
                content_text = result["content"][0].get("text", "{}")
                return json.loads(content_text)
            
            return {"results": []}
            
        except Exception as e:
            logger.warning(f"Failed to execute SPARQL query: {e}")
            return {"results": [], "error": str(e)}
    
    async def get_entity_definition(self, 
                                  entity_uri: str,
                                  include_relationships: bool = True,
                                  include_hierarchy: bool = True,
                                  reasoning_depth: int = 2) -> Dict[str, Any]:
        """
        Get comprehensive definition of an entity.
        
        Args:
            entity_uri: URI of the entity
            include_relationships: Include entity relationships
            include_hierarchy: Include class hierarchy
            reasoning_depth: Depth of relationship traversal
            
        Returns:
            Entity definition with context
        """
        try:
            params = {
                "entity_uri": entity_uri,
                "include_relationships": include_relationships,
                "include_hierarchy": include_hierarchy,
                "reasoning_depth": reasoning_depth
            }
            
            result = await self._make_jsonrpc_request(
                "call_tool",
                {
                    "name": "get_entity_definition",
                    "arguments": params
                }
            )
            
            # Parse the result
            if "content" in result and result["content"]:
                content_text = result["content"][0].get("text", "{}")
                return json.loads(content_text)
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to get entity definition: {e}")
            return {"error": str(e)}
    
    async def natural_language_query(self, 
                                   question: str,
                                   domain_id: str = "engineering-ethics",
                                   context_entities: Optional[List[str]] = None,
                                   explain_query: bool = True) -> Dict[str, Any]:
        """
        Convert natural language question to SPARQL and execute.
        
        Args:
            question: Natural language question
            domain_id: Professional domain
            context_entities: Related entities for context
            explain_query: Include generated SPARQL in response
            
        Returns:
            Query results with explanation
        """
        try:
            params = {
                "question": question,
                "domain_id": domain_id,
                "context_entities": context_entities or [],
                "explain_query": explain_query
            }
            
            result = await self._make_jsonrpc_request(
                "call_tool",
                {
                    "name": "natural_language_query",
                    "arguments": params
                }
            )
            
            # Parse the result
            if "content" in result and result["content"]:
                content_text = result["content"][0].get("text", "{}")
                return json.loads(content_text)
            
            return {"results": [], "query": "", "explanation": ""}
            
        except Exception as e:
            logger.warning(f"Failed to execute natural language query: {e}")
            return {"results": [], "error": str(e)}
    
    async def get_guidelines(self, 
                           world_id: Optional[Union[str, int]] = None,
                           world_name: Optional[str] = None) -> Optional[str]:
        """
        Get guidelines for a specific world/domain.
        
        This method attempts to retrieve domain-specific guidelines that can
        be used to enhance LLM prompts with contextual information.
        
        Args:
            world_id: World identifier
            world_name: World name
            
        Returns:
            Guidelines text or None
        """
        try:
            # For now, return basic guidelines based on domain
            # In a full implementation, this would query the MCP server for
            # domain-specific guidelines stored in the ontology
            
            domain_id = str(world_id) if world_id else "engineering-ethics"
            
            # Basic engineering ethics guidelines as fallback
            if "engineering" in domain_id.lower() or "ethics" in domain_id.lower():
                return """
                Engineering Ethics Guidelines:
                1. Hold paramount the safety, health, and welfare of the public
                2. Perform services only in areas of competence
                3. Issue public statements only in an objective and truthful manner
                4. Act for each employer as faithful agents or trustees
                5. Avoid deceptive acts
                6. Conduct themselves honorably, responsibly, ethically
                """
            
            # Could be enhanced to fetch from MCP server:
            # guidelines = await self.natural_language_query(
            #     f"What are the key guidelines for {domain_id}?",
            #     domain_id=domain_id
            # )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get guidelines: {e}")
            return None
    
    async def close(self):
        """Close the MCP context manager and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("MCP context manager closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MCP context manager statistics."""
        return {
            "server_url": self.config.url,
            "server_available": self._server_available,
            "last_health_check": self._last_health_check,
            "cache_size": len(self._context_cache),
            "cached_requests": len(self._cache_timestamps),
            "session_active": self._session is not None and not self._session.closed
        }


# Global instance for singleton access pattern (similar to ProEthica's MCPClient)
_global_mcp_manager = None


def get_mcp_context_manager(server_url: Optional[str] = None) -> MCPContextManager:
    """Get the global MCP context manager instance."""
    global _global_mcp_manager
    if _global_mcp_manager is None:
        url = server_url or os.environ.get("ONTSERVE_MCP_URL", "http://localhost:8082")
        _global_mcp_manager = MCPContextManager(url)
    return _global_mcp_manager


def reset_mcp_context_manager():
    """Reset the global MCP context manager (for testing)."""
    global _global_mcp_manager
    if _global_mcp_manager:
        # Note: Should use asyncio.run(manager.close()) in real usage
        _global_mcp_manager = None
