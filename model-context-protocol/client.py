import asyncio

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # Start MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    # Create MCP connection
    async with stdio_client(server_params) as (read, write):

        # Create session
        async with ClientSession(read, write) as session:

            # Initialize session
            await session.initialize()

            print("\n==============================")
            print("MCP CLIENT CONNECTED")
            print("==============================")

            # List available tools
            tools = await session.list_tools()

            print("\nAvailable Tools:\n")

            for tool in tools.tools:
                print("-", tool.name)

            print("\n==============================")
            print("CALLING TOOLS")
            print("==============================")

            # Call add tool
            result1 = await session.call_tool(
                "add",
                arguments={
                    "a": 10,
                    "b": 20
                }
            )

            print("\nAdd Tool Result:")
            print(result1.content[0].text)

            # Call multiply tool
            result2 = await session.call_tool(
                "multiply",
                arguments={
                    "a": 5,
                    "b": 6
                }
            )

            print("\nMultiply Tool Result:")
            print(result2.content[0].text)

            # Call greet tool
            result3 = await session.call_tool(
                "greet",
                arguments={
                    "name": "Guru"
                }
            )

            print("\nGreet Tool Result:")
            print(result3.content[0].text)

            print("\n==============================")
            print("MCP DEMO FINISHED")
            print("==============================\n")


# Run client
if __name__ == "__main__":
    asyncio.run(main())