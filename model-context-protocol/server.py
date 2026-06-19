from mcp.server.fastmcp import FastMCP

# Create MCP Server
mcp = FastMCP("Simple MCP Demo Server")

# Tool 1
@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers
    """
    return a + b

# Tool 2
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers
    """
    return a * b

# Tool 3
@mcp.tool()
def greet(name: str) -> str:
    """
    Greet a person
    """
    return f"Hello {name}, welcome to MCP!"

# Run MCP Server
if __name__ == "__main__":
    mcp.run()