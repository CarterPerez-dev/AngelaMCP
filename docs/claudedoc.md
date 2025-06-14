# Set up MCP for Claude Code

Model Context Protocol (MCP) is an open protocol that enables LLMs to access external tools and data sources. For more details, see the [MCP documentation](https://modelcontextprotocol.io/introduction).

<Warning>
  Use third party MCP servers at your own risk. Make sure you trust the MCP
  servers, and be especially careful when using MCP servers that talk to the
  internet, as these can expose you to prompt injection risk.
</Warning>

## Configure MCP servers

**When to use:** You want to enhance Claude's capabilities by connecting it to specialized tools and external servers using the Model Context Protocol.

<Steps>
  <Step title="Add an MCP Studio Server">
    ```bash
    # Basic syntax
    claude mcp add <name> <command> [args...]

    # Example: Adding a local server
    claude mcp add my-server -e API_KEY=123 -- /path/to/server arg1 arg2
    ```
  </Step>

  <Step title="Add an MCP SSE Server">
    ```bash
    # Basic syntax
    claude mcp add --transport sse <name> <url>

    # Example: Adding an SSE server
    claude mcp add --transport sse sse-server https://example.com/sse-endpoint
    ```
  </Step>

  <Step title="Manage your MCP servers">
    ```bash
    # List all configured servers
    claude mcp list

    # Get details for a specific server
    claude mcp get my-server

    # Remove a server
    claude mcp remove my-server
    ```
  </Step>
</Steps>

**Tips:**

* Use the `-s` or `--scope` flag to specify where the configuration is stored:
  * `local` (default): Available only to you in the current project (was called `project` in older versions)
  * `project`: Shared with everyone in the project via `.mcp.json` file
  * `user`: Available to you across all projects (was called `global` in older versions)
* Set environment variables with `-e` or `--env` flags (e.g., `-e KEY=value`)
* Configure MCP server startup timeout using the MCP\_TIMEOUT environment variable (e.g., `MCP_TIMEOUT=10000 claude` sets a 10-second timeout)
* Check MCP server status any time using the `/mcp` command within Claude Code
* MCP follows a client-server architecture where Claude Code (the client) can connect to multiple specialized servers

## Understanding MCP server scopes

**When to use:** You want to understand how different MCP scopes work and how to share servers with your team.

<Steps>
  <Step title="Local-scoped MCP servers">
    The default scope (`local`) stores MCP server configurations in your project-specific user settings. These servers are only available to you while working in the current project.

    ```bash
    # Add a local-scoped server (default)
    claude mcp add my-private-server /path/to/server

    # Explicitly specify local scope
    claude mcp add my-private-server -s local /path/to/server
    ```
  </Step>

  <Step title="Project-scoped MCP servers (.mcp.json)">
    Project-scoped servers are stored in a `.mcp.json` file at the root of your project. This file should be checked into version control to share servers with your team.

    ```bash
    # Add a project-scoped server
    claude mcp add shared-server -s project /path/to/server
    ```

    This creates or updates a `.mcp.json` file with the following structure:

    ```json
    {
      "mcpServers": {
        "shared-server": {
          "command": "/path/to/server",
          "args": [],
          "env": {}
        }
      }
    }
    ```
  </Step>

  <Step title="User-scoped MCP servers">
    User-scoped servers are available to you across all projects on your machine, and are private to you.

    ```bash
    # Add a user server
    claude mcp add my-user-server -s user /path/to/server
    ```
  </Step>
</Steps>

**Tips:**

* Local-scoped servers take precedence over project-scoped and user-scoped servers with the same name
* Project-scoped servers (in `.mcp.json`) take precedence over user-scoped servers with the same name
* Before using project-scoped servers from `.mcp.json`, Claude Code will prompt you to approve them for security
* The `.mcp.json` file is intended to be checked into version control to share MCP servers with your team
* Project-scoped servers make it easy to ensure everyone on your team has access to the same MCP tools
* If you need to reset your choices for which project-scoped servers are enabled or disabled, use the `claude mcp reset-project-choices` command

## Connect to a Postgres MCP server

**When to use:** You want to give Claude read-only access to a PostgreSQL database for querying and schema inspection.

<Steps>
  <Step title="Add the Postgres MCP server">
    ```bash
    claude mcp add postgres-server /path/to/postgres-mcp-server --connection-string "postgresql://user:pass@localhost:5432/mydb"
    ```
  </Step>

  <Step title="Query your database with Claude">
    ```
    # In your Claude session, you can ask about your database

    > describe the schema of our users table
    > what are the most recent orders in the system?
    > show me the relationship between customers and invoices
    ```
  </Step>
</Steps>

**Tips:**

* The Postgres MCP server provides read-only access for safety
* Claude can help you explore database structure and run analytical queries
* You can use this to quickly understand database schemas in unfamiliar projects
* Make sure your connection string uses appropriate credentials with minimum required permissions

## Add MCP servers from JSON configuration

**When to use:** You have a JSON configuration for a single MCP server that you want to add to Claude Code.

<Steps>
  <Step title="Add an MCP server from JSON">
    ```bash
    # Basic syntax
    claude mcp add-json <name> '<json>'

    # Example: Adding a stdio server with JSON configuration
    claude mcp add-json weather-api '{"type":"stdio","command":"/path/to/weather-cli","args":["--api-key","abc123"],"env":{"CACHE_DIR":"/tmp"}}'
    ```
  </Step>

  <Step title="Verify the server was added">
    ```bash
    claude mcp get weather-api
    ```
  </Step>
</Steps>

**Tips:**

* Make sure the JSON is properly escaped in your shell
* The JSON must conform to the MCP server configuration schema
* You can use `-s global` to add the server to your global configuration instead of the project-specific one

## Import MCP servers from Claude Desktop

**When to use:** You have already configured MCP servers in Claude Desktop and want to use the same servers in Claude Code without manually reconfiguring them.

<Steps>
  <Step title="Import servers from Claude Desktop">
    ```bash
    # Basic syntax 
    claude mcp add-from-claude-desktop 
    ```
  </Step>

  <Step title="Select which servers to import">
    After running the command, you'll see an interactive dialog that allows you to select which servers you want to import.
  </Step>

  <Step title="Verify the servers were imported">
    ```bash
    claude mcp list 
    ```
  </Step>
</Steps>

**Tips:**

* This feature only works on macOS and Windows Subsystem for Linux (WSL)
* It reads the Claude Desktop configuration file from its standard location on those platforms
* Use the `-s global` flag to add servers to your global configuration
* Imported servers will have the same names as in Claude Desktop
* If servers with the same names already exist, they will get a numerical suffix (e.g., `server_1`)

## Use Claude Code as an MCP server

**When to use:** You want to use Claude Code itself as an MCP server that other applications can connect to, providing them with Claude's tools and capabilities.

<Steps>
  <Step title="Start Claude as an MCP server">
    ```bash
    # Basic syntax
    claude mcp serve
    ```
  </Step>

  <Step title="Connect from another application">
    You can connect to Claude Code MCP server from any MCP client, such as Claude Desktop. If you're using Claude Desktop, you can add the Claude Code MCP server using this configuration:

    ```json
    {
      "command": "claude",
      "args": ["mcp", "serve"],
      "env": {}
    }
    ```
  </Step>
</Steps>

**Tips:**

* The server provides access to Claude's tools like View, Edit, LS, etc.
* In Claude Desktop, try asking Claude to read files in a directory, make edits, and more.
* Note that this MCP server is simply exposing Claude Code's tools to your MCP client, so your own client is responsible for implementing user confirmation for individual tool calls.
