# Basic Usage Examples

This document provides practical examples of using the Titan Memory MCP Server in different scenarios.

## 1. Basic Memory Processing

### Process New Code

```typescript
// Process new code input
await callTool("process_input", {
  text: `
function fibonacci(n: number): number {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}
  `,
  context: "TypeScript recursive function",
});

// Check memory state
const state = await callTool("get_memory_state", {});
console.log("Memory Stats:", state.memoryStats);
```

## 2. Using with Cursor IDE

### Process Current File

1. Open a file in Cursor
2. Press `Cmd/Ctrl + Shift + P`
3. Type "MCP: Process Input"
4. Select entire file

```typescript
// The server will process the file and update its memory
// You can then check the memory state:
const state = await callTool("get_memory_state", {});
```

### Process Selection

1. Select code in editor
2. Right-click
3. Choose "Process Selection with Titan Memory"

## 3. HTTP API Example

### Using curl

```bash
# Get server status
curl http://localhost:3000/

# Process input
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "process_input",
      "arguments": {
        "text": "console.log(\"Hello World\");"
      }
    },
    "id": 1
  }'
```

### Using Node.js

```typescript
import fetch from "node-fetch";

async function processCode(code: string) {
  const response = await fetch("http://localhost:3000/mcp", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        name: "process_input",
        arguments: {
          text: code,
        },
      },
      id: 1,
    }),
  });

  return await response.json();
}

// Usage
const result = await processCode(`
class Example {
  constructor() {
    this.value = 42;
  }
}
`);
console.log(result);
```

## 4. Integration with Build Process

### Example npm Script

```json
{
  "scripts": {
    "build": "tsc",
    "postbuild": "node scripts/process-build.js"
  }
}
```

```typescript
// scripts/process-build.js
import { readdir, readFile } from "fs/promises";
import { join } from "path";
import fetch from "node-fetch";

async function processBuildOutput() {
  const buildDir = join(process.cwd(), "build");
  const files = await readdir(buildDir);

  for (const file of files) {
    if (file.endsWith(".js")) {
      const content = await readFile(join(buildDir, file), "utf-8");
      await processCode(content);
    }
  }
}

processBuildOutput().catch(console.error);
```

## 5. Using with Testing

### Jest Integration

```typescript
import { TitanMemoryServer } from "@henryhawke/mcp-titan";

describe("Code Processing", () => {
  let server: TitanMemoryServer;

  beforeAll(async () => {
    server = new TitanMemoryServer();
    await server.run();
  });

  afterAll(async () => {
    await server.cleanup();
  });

  test("processes TypeScript code", async () => {
    const result = await server.processInput(`
      interface User {
        id: number;
        name: string;
      }
    `);

    expect(result.memoryUpdated).toBe(true);
    expect(result.surprise).toBeGreaterThan(0);
  });
});
```

## 6. Advanced Usage

### Custom Memory Processing

```typescript
// Process with custom context
await callTool("process_input", {
  text: "const x = 42;",
  context: JSON.stringify({
    language: "typescript",
    project: "example",
    file: "constants.ts",
    dependencies: ["@types/node"],
  }),
});

// Get detailed memory insights
const state = await callTool("get_memory_state", {});
console.log(JSON.stringify(state, null, 2));
```

### Error Handling

```typescript
try {
  await callTool("process_input", {
    text: "invalid code )",
    context: "JavaScript",
  });
} catch (error) {
  console.error("Processing error:", error);
  // Handle error appropriately
}
```
