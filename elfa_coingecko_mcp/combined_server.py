import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
import time
from datetime import datetime, timedelta

import aiohttp
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

# Configuration
class Config:
    PORT = int(os.environ.get("PORT", 8004))
    COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
    ELFA_API_BASE_URL = "https://api.elfa.ai"
    ELFA_API_KEY = "elfak_9da97adea0a74a1b78d414d846c160f8ecb180b4" # Hardcoded API Key
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOGGER_NAME = "multimcps-server"

logger = logging.getLogger(Config.LOGGER_NAME)
logger.setLevel(Config.LOG_LEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter(Config.LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Helper function for ELFA API calls
async def call_elfa_api(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper function to call the ELFA AI API."""
    url = f"{Config.ELFA_API_BASE_URL}{endpoint}"
    headers = {"X-API-Key": Config.ELFA_API_KEY}
    logger.info(f"Calling ELFA API: {url} with params: {params}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error calling ELFA API {endpoint}: {response.status} - {error_text}")
                    # Return a structured error for clarity
                    return {"success": False, "error": f"API Error {response.status}", "details": error_text}
    except Exception as e:
        logger.exception(f"Exception calling ELFA API {endpoint}: {e}")
        return {"success": False, "error": "Exception during API call", "details": str(e)}


# MCP Server Implementation
class MultiMCPServer:
    def __init__(self):
        self.server = None

    async def get_coin_price(self, coin_id: str) -> Dict[str, Any]:
        """Fetches the current price of a coin from CoinGecko."""
        url = f"{Config.COINGECKO_API_URL}/simple/price?ids={coin_id}&vs_currencies=usd"
        logger.info(f"Fetching price for {coin_id} from {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if coin_id in data and "usd" in data[coin_id]:
                            return {"price": data[coin_id]["usd"]}
                        else:
                            logger.warning(f"Could not retrieve USD price for {coin_id}")
                            return {"price": None}
                    else:
                        logger.error(f"Error fetching price for coin_id: {response.status}")
                        return {"price": None}
        except Exception as e:
            logger.exception(f"Error fetching price for coin_id: {e}")
            return {"price": None}

    async def search_twitter_mentions(self, keywords: List[str], from_timestamp: int, to_timestamp: int, limit: Optional[int] = 20, searchType: Optional[str] = None, cursor: Optional[str] = None) -> Dict[str, Any]:
        """Searches for mentions of specific keywords on Twitter using ELFA AI API."""
        endpoint = "/v1/mentions/search"
        params = {
            "keywords": ",".join(keywords), # API expects comma-separated string
            "from": from_timestamp,
            "to": to_timestamp,
            "limit": limit
        }
        if searchType:
            params["searchType"] = searchType
        if cursor:
            params["cursor"] = cursor
            
        return await call_elfa_api(endpoint, params)

    async def get_account_smart_stats(self, username: str) -> Dict[str, Any]:
        """Retrieve smart stats and social metrics for a given Twitter username."""
        endpoint = "/v1/account/smart-stats"
        params = {"username": username}
        return await call_elfa_api(endpoint, params)

    async def get_trending_tokens(self, timeWindow: Optional[str] = "24h", page: Optional[int] = 1, pageSize: Optional[int] = 50, minMentions: Optional[int] = 5) -> Dict[str, Any]:
        """Query tokens most discussed in a particular time period."""
        endpoint = "/v1/trending-tokens"
        params = {
            "timeWindow": timeWindow,
            "page": page,
            "pageSize": pageSize,
            "minMentions": minMentions
        }
        return await call_elfa_api(endpoint, params)


    async def initialize(self) -> Server:
        if not self.server: # Initialize only once
             self.server = self._create_server()
        return self.server

    def _create_server(self) -> Server:
        app = Server("multimcps-server")

        @app.list_tools()
        async def list_tools() -> List[Dict[str, Any]]:
            return [
                {
                    "name": "get_coin_price",
                    "description": "Gets the current price of a coin from CoinGecko.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "coin_id": {
                                "type": "string",
                                "description": "The CoinGecko ID of the coin (e.g., bitcoin, ethereum)."
                            }
                        },
                        "required": ["coin_id"]
                    }
                },
                {
                    "name": "search_twitter_mentions",
                    "description": "Searches for mentions of specific keywords on Twitter using ELFA AI API.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "List of keywords to search for (max 5)."
                            },
                            "from_timestamp": {
                                "type": "integer",
                                "description": "Start date (Unix timestamp)."
                            },
                            "to_timestamp": {
                                "type": "integer",
                                "description": "End date (Unix timestamp)."
                            },
                             "limit": {
                                "type": "integer",
                                "description": "Number of results to return (default 20, max 30).",
                                "default": 20
                            },
                            "searchType": {
                                "type": "string",
                                "description": "Type of search (and, or).",
                                "enum": ["and", "or"]
                            },
                            "cursor": {
                                "type": "string",
                                "description": "Cursor for pagination."
                            }
                        },
                        "required": ["keywords", "from_timestamp", "to_timestamp"]
                    }
                },
                 {
                    "name": "get_account_smart_stats",
                    "description": "Retrieve smart stats and social metrics for a given Twitter username.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "Twitter username (without @)."
                            }
                        },
                        "required": ["username"]
                    }
                },
                {
                    "name": "get_trending_tokens",
                    "description": "Query tokens most discussed in a particular time period.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "timeWindow": {
                                "type": "string",
                                "description": "Time window for trending analysis (e.g., 1h, 24h, 7d).",
                                "default": "24h"
                            },
                             "page": {
                                "type": "integer",
                                "description": "Page number for pagination.",
                                "default": 1
                            },
                            "pageSize": {
                                "type": "integer",
                                "description": "Number of items per page.",
                                "default": 50
                            },
                            "minMentions": {
                                "type": "integer",
                                "description": "Minimum number of mentions required.",
                                "default": 5
                            }
                        },
                        "required": [] # All parameters have defaults
                    }
                }
            ]

        @app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                if name == "get_coin_price":
                    coin_id = arguments.get("coin_id")
                    if not coin_id: raise ValueError("coin_id is required")
                    result = await self.get_coin_price(coin_id)
                elif name == "search_twitter_mentions":
                    keywords = arguments.get("keywords")
                    from_timestamp = arguments.get("from_timestamp")
                    to_timestamp = arguments.get("to_timestamp")
                    if not keywords: raise ValueError("keywords is required")
                    if from_timestamp is None: raise ValueError("from_timestamp is required")
                    if to_timestamp is None: raise ValueError("to_timestamp is required")
                    
                    result = await self.search_twitter_mentions(
                        keywords, 
                        from_timestamp, 
                        to_timestamp,
                        arguments.get("limit"),
                        arguments.get("searchType"),
                        arguments.get("cursor")
                    )
                elif name == "get_account_smart_stats":
                     username = arguments.get("username")
                     if not username: raise ValueError("username is required")
                     result = await self.get_account_smart_stats(username)
                elif name == "get_trending_tokens":
                     result = await self.get_trending_tokens(
                         arguments.get("timeWindow"),
                         arguments.get("page"),
                         arguments.get("pageSize"),
                         arguments.get("minMentions")
                     )
                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Return result as TextContent
                return [{"type": "text", "text": str(result)}]
                
            except Exception as e:
                 logger.exception(f"Error calling tool {name}: {e}")
                 # Return error information as TextContent
                 return [{"type": "text", "text": f"Error executing tool {name}: {str(e)}"}]


        return app

    async def run_sse(self, port: int):
        await self.initialize() # Ensure server is initialized before starting Starlette

        messages_path = "/messages/"
        sse = SseServerTransport(messages_path)

        async def handle_sse(request):
            # Server should already be initialized here by run_sse
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await self.server.run(streams[0], streams[1], self.server.create_initialization_options())

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )

        logger.info(f"Starting SSE server on port {port}")
        config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port)
        server = uvicorn.Server(config)
        await server.serve()

async def main():
    server = MultiMCPServer()
    await server.run_sse(Config.PORT)

if __name__ == "__main__":
    asyncio.run(main())
