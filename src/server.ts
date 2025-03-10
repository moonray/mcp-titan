import * as tf from '@tensorflow/tfjs-node';
import {
  Transport,
  CallToolRequest,
  CallToolResult,
  ServerCapabilities,
  McpServer
} from './types.js';
import * as readline from 'readline';
import WebSocket from 'ws';
import { z } from 'zod';

/**
 * WebSocket transport implementation for MCP server
 */
export class WebSocketTransport implements Transport {
  private ws: WebSocket;
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;

  constructor(private url: string) {
    this.ws = new WebSocket(url);
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws.on('open', resolve);
      this.ws.on('error', reject);

      this.ws.on('message', async (data) => {
        if (!this.requestHandler) return;

        try {
          const request = JSON.parse(data.toString()) as CallToolRequest;
          const result = await this.requestHandler(request);
          this.ws.send(JSON.stringify(result));
        } catch (error) {
          this.ws.send(JSON.stringify({
            success: false,
            error: error instanceof Error ? error.message : 'Invalid request',
            content: [{
              type: "error",
              text: error instanceof Error ? error.message : 'Invalid request'
            }]
          }));
        }
      });
    });
  }

  async disconnect(): Promise<void> {
    this.ws.close();
  }

  onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }

  send(message: unknown): void {
    this.ws.send(JSON.stringify(message));
  }
}

/**
 * Standard I/O transport implementation for MCP server
 */
export class StdioServerTransport implements Transport {
  private rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;

  async connect(): Promise<void> {
    this.rl.on('line', async (line) => {
      if (!this.requestHandler) return;

      try {
        const request = JSON.parse(line) as CallToolRequest;
        const result = await this.requestHandler(request);
        console.log(JSON.stringify(result));
      } catch (error) {
        console.log(JSON.stringify({
          success: false,
          error: error instanceof Error ? error.message : 'Invalid request',
          content: [{
            type: "error",
            text: error instanceof Error ? error.message : 'Invalid request'
          }]
        }));
      }
    });
    return Promise.resolve();
  }

  async disconnect(): Promise<void> {
    this.rl.close();
  }

  onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }

  send(message: unknown): void {
    console.log(JSON.stringify(message));
  }
}

/**
 * Enhanced MCP Server Implementation
 * This implementation provides better error handling and standardized responses
 * to make it more compatible with Cursor and other LLM tools.
 */
export class McpServerImpl implements McpServer {
  private tools = new Map<string, Function>();
  private name: string;
  private version: string;
  private description?: string;
  private toolSchemas: Record<string, any> = {};

  constructor(config: { name: string; version: string; description?: string }) {
    this.name = config.name;
    this.version = config.version;
    this.description = config.description;
  }

  /**
   * Register a tool with the server
   * @param name Tool name
   * @param schema Zod schema for parameters validation
   * @param handler Function to handle tool requests
   */
  tool(name: string, schema: z.ZodRawShape | string, handler: Function): void {
    const zodSchema = typeof schema === 'string'
      ? z.object({})
      : z.object(schema);

    // Store schema for tool discovery
    this.toolSchemas[name] = {
      description: typeof schema === 'string' ? schema : `Tool: ${name}`,
      parameters: this.extractSchemaProperties(zodSchema)
    };

    this.tools.set(name, async (params: unknown) => {
      try {
        const validated = zodSchema.parse(params);
        const result = await handler(validated);
        
        // Ensure result has the correct format
        if (!result.content) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify(result)
            }]
          };
        }
        
        return result;
      } catch (error) {
        console.error(`Error executing tool ${name}:`, error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Tool execution failed',
          content: [{
            type: "error",
            text: error instanceof Error ? error.message : 'Tool execution failed'
          }],
          isError: true
        };
      }
    });
  }

  /**
   * Extract schema properties for tool discovery
   */
  private extractSchemaProperties(schema: z.ZodObject<any>): Record<string, any> {
    const shape = schema._def.shape();
    const properties: Record<string, any> = {};
    
    for (const [key, def] of Object.entries(shape)) {
      let type = 'string';
      let description = '';
      
      if (def instanceof z.ZodString) {
        type = 'string';
      } else if (def instanceof z.ZodNumber) {
        type = 'number';
      } else if (def instanceof z.ZodBoolean) {
        type = 'boolean';
      } else if (def instanceof z.ZodArray) {
        type = 'array';
      } else if (def instanceof z.ZodObject) {
        type = 'object';
      }
      
      // Extract description if available
      if (def._def.description) {
        description = def._def.description;
      }
      
      properties[key] = {
        type,
        description,
        required: !def.isOptional()
      };
    }
    
    return properties;
  }

  /**
   * Connect to a transport
   * @param transport Transport to connect to
   */
  async connect(transport: Transport): Promise<void> {
    transport.onRequest(async (request: CallToolRequest) => {
      const handler = this.tools.get(request.name);
      if (!handler) {
        return {
          success: false,
          error: `Tool ${request.name} not found`,
          content: [{
            type: "error" as const,
            text: `Tool ${request.name} not found`
          }],
          isError: true
        };
      }
      
      try {
        return await handler(request.parameters);
      } catch (error) {
        console.error(`Error handling request for tool ${request.name}:`, error);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          content: [{
            type: "error" as const,
            text: error instanceof Error ? error.message : 'Unknown error'
          }],
          isError: true
        };
      }
    });

    await transport.connect();

    // Send server info with tool schemas for discovery
    transport.send?.({
      jsonrpc: "2.0",
      method: "server_info",
      params: {
        name: this.name,
        version: this.version,
        description: this.description || `${this.name} MCP Server v${this.version}`,
        capabilities: {
          tools: true,
          memory: true
        },
        tools: this.toolSchemas
      }
    });
  }
}