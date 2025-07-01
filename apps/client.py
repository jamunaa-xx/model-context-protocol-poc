import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, List

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

nest_asyncio.apply()

load_dotenv("../.env")

class MCPOpenAIClient:
    """Client for interacting with Ollama models(using OpenAI) using MCP tools."""
    
    def __init__(self, model: str = "llama3.1"):
        """
            Initialize the Ollama MCP Client.
            
            Args:
                model (str, optional): The Ollama model to use. Defaults to "llama3.1".
        """
        
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = AsyncOpenAI(base_url="http://localhost:11434/v1/")
        self.model = model
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """
            Connect to an MCP server.
            
            Args:
                server_script_path: The path to the server script.
        """
        
        # Server configuration
        server_params = StdioServerParameters(
            command = "python",
            args=[server_script_path]
        )
        
        # Connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport        
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        # Initialize connection
        await self.session.initialize()
        
        # List available tools
        tools_result = await self.session.list_tools()
        print("Connected to server with tools:")
        for tool in tools_result.tools:
            print(f"- {tool.name}: {tool.description}")
            
    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
            Get available MCP tools.
            
            Returns:
                List[Dict[str, Any]]: A list of available MCP tools in OpenAI format.
        """
        
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in tools_result.tools
        ]
        
    async def process_query(self, query: str) -> str:
        """
            Process a query using Ollama and avaliable MCP tools and return the response.
            
            Args:
                query (str): The user query to process.
                
            Returns:
                str: The response from the Ollama model.
        """
        
        # Get available tools
        tools = await self.get_mcp_tools()
        
        # Initial OpenAI API call
        response = await self.openai_client.chat.completions.create(
             model = self.model,
             messages = [{"role": "user", "content": query}],
             tools=tools,
             tool_choice = "auto"
        )
        
        # Get assistant's response
        assistant_message = response.choices[0].message
        
        # Initialize conversation with user query and assistant response
        messages = [
            {"role": "user", "content": query},
            assistant_message
        ]
        
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                result = await self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                )                
                # Add tool response to conversation
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "content": result.content[0].text
                })
                
                # For knowledge base, also add as system message for context (unnecessary since we are appending tool reponse)
                # if tool_call.function.name == "get_knowledge_base":
                #     messages.append({"role": "system", "content": result.content[0].text})
            
            # Get final response from OpenAI with tool results
            final_response = await self.openai_client.chat.completions.create(
                model = self.model,
                messages = messages,
                tools=tools,
                tool_choice="none"
            )
                        
            return final_response.choices[0].message.content
        
        # No tool calls, just return the direct response
        return assistant_message.content
    
    async def chat_loop(self):
        """
            Starts an interactive chat loop with the user.
            
            Waits for user input, processes the query using the Ollama model,
            and prints out the response. If the user types 'quit', the loop exits.
            
            This function is an infinite loop and should be called inside an async context.
        """
        
        print("\nMCP Ollama Client Started! Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                response = await self.process_query(query)
                print("\nAnswer: " + response)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        
        
async def main():
    """"Main entrypoint for the client."""
    
    client = MCPOpenAIClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()
    
if __name__ == "__main__":
    asyncio.run(main())
