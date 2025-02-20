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
            error: error instanceof Error ? error.message : 'Invalid request'
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
          error: error instanceof Error ? error.message : 'Invalid request'
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
 */
export class McpServerImpl implements McpServer {
  private tools = new Map<string, Function>();
  private name: string;
  private version: string;

  constructor(config: { name: string; version: string }) {
    this.name = config.name;
    this.version = config.version;
  }

  tool(name: string, schema: z.ZodRawShape | string, handler: Function): void {
    const zodSchema = typeof schema === 'string'
      ? z.object({})
      : z.object(schema);

    this.tools.set(name, async (params: unknown) => {
      const validated = zodSchema.parse(params);
      return handler(validated);
    });
  }

  async connect(transport: Transport): Promise<void> {
    transport.onRequest(async (request: CallToolRequest) => {
      const handler = this.tools.get(request.name);
      if (!handler) {
        return {
          success: false,
          error: `Tool ${request.name} not found`,
          content: [{
            type: "text" as const,
            text: `Tool ${request.name} not found`
          }]
        };
      }
      try {
        return await handler(request.parameters);
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          content: [{
            type: "text" as const,
            text: error instanceof Error ? error.message : 'Unknown error'
          }]
        };
      }
    });

    await transport.connect();

    // Send server info
    transport.send?.({
      jsonrpc: "2.0",
      method: "server_info",
      params: {
        name: this.name,
        version: this.version,
        capabilities: {
          tools: true,
          memory: true
        }
      }
    });
  }
}