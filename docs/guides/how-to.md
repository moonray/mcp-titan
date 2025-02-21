# Setting Up MCP with Claude and Cursor: A Comprehensive Guide

## Prerequisites

### System Requirements

- **Operating System**:
  - Windows 10/11 (64-bit)
  - macOS 11.0 or later
- **Node.js**: Version 16.x or later (LTS recommended)
- **Claude Desktop App**: Latest version
- **Cursor**: Latest version
- **Disk Space**: At least 1GB free space
- **Memory**: Minimum 8GB RAM recommended

### Required Software Installation

1. **Node.js and npm**

   ```bash
   # Visit https://nodejs.org/
   # Download and install the LTS version

   # Verify installation
   node --version
   npm --version
   ```

2. **TypeScript**

   ```bash
   npm install -g typescript
   tsc --version
   ```

3. **Claude Desktop App**

   - Download from Anthropic's official website
   - Complete account setup and authentication

4. **Cursor**
   - Download from https://cursor.sh
   - Install and set up your development environment

## Setting Up Your MCP Development Environment

### 1. Create Your Project Structure

```bash
mkdir my-mcp-project
cd my-mcp-project
npm init -y

# Install required dependencies
npm install 
```

### 2. Configure TypeScript

Create `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node",
    "moduleResolution": "Node16",
    "outDir": "./build",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

### 3. MCP Server Configuration

Create `claude_desktop_config.json` in your home directory:

**Windows**:

```json
{
  "servers": {
    "my-mcp-server": {
      "command": "node",
      "args": ["C:\\Users\\YourUsername\\my-mcp-project\\build\\index.js"]
    }
  }
}
```

**macOS**:

```json
{
  "servers": {
    "my-mcp-server": {
      "command": "node",
      "args": ["/Users/YourUsername/my-mcp-project/build/index.js"]
    }
  }
}
```

## Configuring Claude Desktop

### 1. Enable MCP in Claude Desktop

1. Open Claude Desktop
2. Go to Settings
3. Navigate to "Developer Settings"
4. Enable "MCP Tools"
5. Restart Claude Desktop

### 2. Verify MCP Server Connection

1. Look for the hammer icon in Claude Desktop
2. Click it to see available MCP servers
3. Your server should appear in the list
4. Test the connection using a simple command

## Using MCP with Cursor

### 1. Cursor Configuration

1. Open Cursor
2. Go to Settings > Extensions
3. Enable MCP integration
4. Configure server paths if needed

### 2. Testing the Setup

Create a test file in Cursor and try this interaction:

```typescript
// test.ts
console.log("Testing MCP connection");
```

## Prompt Instructions for Claude

When working with Claude, use these prompt templates for optimal results:

### 1. Basic MCP Command Template

```
Please execute the following MCP command:
[Command description]
Parameters:
- param1: value1
- param2: value2
```

### 2. File Operation Template

```
Using MCP, please:
1. Open [file path]
2. [Describe operation]
3. Save the changes
```

### 3. Project Analysis Template

```
Using MCP tools, analyze this project:
- Check for [specific aspects]
- Suggest improvements for [area]
- Provide a summary of [feature]
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **"Could not attach to MCP server" Error**

   - Check server path in configuration
   - Verify Node.js installation
   - Run Claude Desktop as administrator (Windows)
   - Check file permissions (macOS)

2. **Server Not Appearing in Claude**

   - Restart Claude Desktop
   - Verify config file syntax
   - Check absolute paths
   - Review server logs

3. **TypeScript Compilation Errors**
   - Verify tsconfig.json settings
   - Check for missing dependencies
   - Update TypeScript version

### Getting Logs

**Windows**:

```bash
type "%APPDATA%\Claude\logs\mcp*.log"
```

**macOS**:

```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

## Best Practices

1. **Version Control**

   - Use Git for your MCP projects
   - Include .gitignore for node_modules
   - Document server configurations

2. **Security**

   - Never commit sensitive data
   - Use environment variables
   - Implement proper error handling

3. **Development Workflow**
   - Use TypeScript strict mode
   - Implement proper logging
   - Follow MCP protocol specifications

## Advanced Configuration

### Custom Tool Development

```typescript
interface CustomTool {
  name: string;
  description: string;
  parameters: {
    [key: string]: {
      type: string;
      description: string;
      required: boolean;
    };
  };
  handler: (params: any) => Promise<any>;
}
```

### Memory Management

```typescript
// Implement proper memory management
const cleanup = () => {
  // Dispose of resources
  // Clear caches
  // Reset states
};

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);
```

## Updating and Maintenance

1. **Regular Updates**

   ```bash
   npm update @modelcontextprotocol/sdk
   npm audit fix
   ```

2. **Monitoring**
   - Check server logs regularly
   - Monitor memory usage
   - Track API response times

## Resources and Documentation

- [Official MCP Documentation](https://modelcontextprotocol.io)
- [Claude API Reference](https://docs.anthropic.com/claude/reference)
- [TypeScript SDK Documentation](https://github.com/modelcontextprotocol/typescript-sdk)
- [Cursor Documentation](https://cursor.sh/docs)

## Community and Support

- Join the MCP Discord community
- Participate in GitHub discussions
- Report issues on the respective repositories
- Share your tools and experiences

Remember to keep your development environment updated and regularly check for new versions of Claude Desktop, Cursor, and the MCP SDK.

## Advanced Integration Features

### Browser Automation with MCP

```typescript
// Example of browser automation tool setup
interface BrowserTool {
  navigate: (url: string) => Promise<void>;
  screenshot: (selector?: string) => Promise<Buffer>;
  click: (selector: string) => Promise<void>;
  type: (selector: string, text: string) => Promise<void>;
}

// Implementation with Puppeteer
const browserTool: BrowserTool = {
  async navigate(url) {
    // Navigation implementation
  },
  async screenshot(selector) {
    // Screenshot implementation
  },
  async click(selector) {
    // Click implementation
  },
  async type(selector, text) {
    // Type implementation
  },
};
```

### Git Integration

```typescript
interface GitTool {
  clone: (repo: string) => Promise<void>;
  commit: (message: string) => Promise<void>;
  push: () => Promise<void>;
  analyze: (path: string) => Promise<CodeAnalysis>;
}
```

## Latest Best Practices (2024)

### 1. Security Enhancements

- Implement rate limiting for tool calls
- Add request validation middleware
- Use secure WebSocket connections
- Implement proper error boundaries

### 2. Performance Optimization

```typescript
// Example of optimized tool registration
class OptimizedMcpServer {
  private toolCache = new Map<string, ToolImplementation>();

  registerTool(name: string, implementation: ToolImplementation) {
    this.toolCache.set(name, implementation);
  }

  async executeTool(name: string, params: unknown) {
    const tool = this.toolCache.get(name);
    if (!tool) throw new Error(`Tool ${name} not found`);
    return await tool.execute(params);
  }
}
```

### 3. Cross-Platform Compatibility

- Use path.join for file paths
- Handle line endings (CRLF/LF)
- Use cross-platform environment variables
- Implement platform-specific error handling

### 4. Integration Testing

```typescript
// Example test suite
describe("MCP Server Integration", () => {
  test("should handle tool registration", async () => {
    const server = new McpServer();
    await server.registerTool("test", testImplementation);
    expect(server.hasToolRegistered("test")).toBe(true);
  });

  test("should execute tools correctly", async () => {
    const result = await server.executeTool("test", testParams);
    expect(result).toMatchExpectedOutput();
  });
});
```

### 5. Error Recovery Strategies

```typescript
class ResilientMcpServer {
  private retryCount = 3;
  private backoffMs = 1000;

  async executeWithRetry(tool: string, params: unknown) {
    for (let i = 0; i < this.retryCount; i++) {
      try {
        return await this.executeTool(tool, params);
      } catch (error) {
        if (i === this.retryCount - 1) throw error;
        await this.wait(this.backoffMs * Math.pow(2, i));
      }
    }
  }
}
```

## Emerging Patterns and Recommendations

### 1. Tool Organization

```typescript
// Group related tools into namespaces
interface ToolNamespace {
  name: string;
  description: string;
  tools: Record<string, Tool>;
}

const fileSystemNamespace: ToolNamespace = {
  name: "fs",
  description: "File system operations",
  tools: {
    read: readImplementation,
    write: writeImplementation,
    delete: deleteImplementation,
  },
};
```

### 2. Context Management

```typescript
// Maintain tool context
class ContextAwareTool {
  private context: Map<string, unknown> = new Map();

  async execute(params: unknown, context?: unknown) {
    this.context.set("lastExecution", Date.now());
    this.context.set("params", params);
    // Tool implementation
  }

  getContext(): unknown {
    return Object.fromEntries(this.context);
  }
}
```

### 3. Monitoring and Analytics

```typescript
// Implement tool usage analytics
interface ToolMetrics {
  executionCount: number;
  averageExecutionTime: number;
  errorRate: number;
  lastExecuted: Date;
}

class MonitoredMcpServer {
  private metrics: Map<string, ToolMetrics> = new Map();

  async trackExecution(tool: string, startTime: number) {
    const endTime = Date.now();
    const duration = endTime - startTime;
    this.updateMetrics(tool, duration);
  }
}
```

Remember to check the [official MCP documentation](https://modelcontextprotocol.io) regularly for updates and new best practices as the ecosystem continues to evolve.

## LLM Prompting Best Practices with MCP

### 1. Structured Tool Requests

When asking Claude to use MCP tools, follow these patterns:

```
I need to [action] using the MCP tools. Please:
1. Check if [precondition]
2. Use the [tool name] to [specific action]
3. Verify the result by [verification step]
```

Example:

```
I need to analyze this codebase using the MCP tools. Please:
1. Check if the repository is accessible
2. Use the git_analyze tool to scan the main branch
3. Verify the result by checking the output format
```

### 2. Context-Aware Commands

```
Given the current [context/state], please:
- Consider [specific aspects]
- Use [tool] to [action]
- Handle any [potential issues]
```

Example:

```
Given the current TypeScript project structure, please:
- Consider our existing dependencies
- Use npm_tool to install required packages
- Handle any version conflicts
```

### 3. Error Recovery Instructions

```
If you encounter [error type]:
1. Try [alternative approach]
2. If that fails, [fallback action]
3. Report back with [specific details]
```

Example:

```
If you encounter a connection error:
1. Try reconnecting to the MCP server
2. If that fails, check the server logs
3. Report back with the error message and timestamp
```

### 4. Sequential Operations

For complex tasks requiring multiple tools:

```
Please perform these steps in order:
1. [First tool] to [action]
   - Expected outcome: [description]
   - If successful: [next step]
   - If failed: [alternative]

2. [Second tool] to [action]
   - Using output from step 1
   - Verify [specific condition]

3. [Final tool] to [action]
   - Combine results from steps 1 and 2
   - Generate [final output]
```

### 5. Validation Requests

```
After [operation], please verify:
- [Condition 1] is met
- [Condition 2] is satisfied
- No [specific issues] are present
```

Example:

```
After the file changes, please verify:
- All imports are properly resolved
- TypeScript compilation succeeds
- No linter errors are present
```

## Quick Reference

### Common MCP Tool Patterns

1. **File Operations**

   ```
   Please [read/write/modify] the file at [path]:
   - Content: [details]
   - Format: [specification]
   - Validation: [requirements]
   ```

2. **Project Analysis**

   ```
   Analyze this project focusing on:
   - Structure: [aspects]
   - Dependencies: [requirements]
   - Issues: [types]
   ```

3. **Code Generation**
   ```
   Generate [component/feature] with:
   - Requirements: [list]
   - Style: [guidelines]
   - Testing: [approach]
   ```

### Response Handling

Always verify MCP tool responses using this pattern:

```
After each tool execution:
1. Check response status
2. Validate output format
3. Verify against requirements
4. Handle any warnings
5. Document any issues
```

Remember to maintain context between multiple tool calls and always handle errors gracefully. Keep your prompts clear, specific, and structured for optimal results with Claude and MCP tools.


