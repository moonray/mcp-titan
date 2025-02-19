import express from 'express';
import bodyParser from 'body-parser';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor } from './types.js';
import { Server, ServerCapabilities, CallToolRequest, CallToolResult, Transport } from './types.js';
import WebSocket from 'ws';
import { Request, Response } from 'express';
import { z } from 'zod';
import readline from 'readline';

interface ToolHandlers {
  storeMemory: {
    subject: string;
    relationship: string;
    object: string;
  };
  recallMemory: {
    query: string;
  };
}

// Schema definitions
export const CallToolRequestSchema = z.object({
  name: z.string(),
  parameters: z.record(z.unknown())
});

const CallToolResultSchema = z.object({
  success: z.boolean(),
  result: z.unknown().optional(),
  error: z.string().optional()
});

export class MCPServer implements Server {
  public name: string;
  public version: string;
  public capabilities: ServerCapabilities;
  private errorHandler: ((error: Error) => void) | null = null;
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;

  constructor(name: string, version: string, capabilities: ServerCapabilities) {
    this.name = name;
    this.version = version;
    this.capabilities = capabilities;
  }

  onError(handler: (error: Error) => void): void {
    this.errorHandler = handler;
  }

  onToolCall(handler: (request: CallToolRequest) => Promise<CallToolResult>, p0: { name: any; description: any; schema: { type: string; properties: any; required: any; }; }, p1: (params: any) => Promise<unknown>): void {
    this.requestHandler = handler;
  }

  async connect(transport: Transport): Promise<void> {
    await transport.connect();
  }

  handleError(error: Error): void {
    if (this.errorHandler) {
      this.errorHandler(error);
    } else {
      console.error('Unhandled server error:', error);
    }
  }

  async handleRequest(request: CallToolRequest): Promise<CallToolResult> {
    if (!this.requestHandler) {
      return {
        success: false,
        error: 'No request handler registered'
      };
    }

    try {
      // Validate request
      const validatedRequest = CallToolRequestSchema.parse(request);
      return await this.requestHandler(validatedRequest);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  public setRequestHandler(handler: (request: CallToolRequest) => Promise<CallToolResult>) {
    this.requestHandler = handler;
  }
}

export class WebSocketTransport implements Transport {
  private ws: WebSocket;
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;

  constructor(url: string) {
    if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
      throw new Error('Invalid WebSocket URL: must start with ws:// or wss://');
    }
    this.ws = new WebSocket(url);
  }

  public async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);

      this.ws.onmessage = async (event) => {
        if (!this.requestHandler) return;

        try {
          const request = JSON.parse(event.data.toString());
          const validatedRequest = CallToolRequestSchema.parse(request);
          const response = await this.requestHandler(validatedRequest);
          this.ws.send(JSON.stringify(response));
        } catch (error) {
          this.ws.send(JSON.stringify({
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error occurred'
          }));
        }
      };
    });
  }

  public async disconnect(): Promise<void> {
    this.ws.close();
    return Promise.resolve();
  }

  public onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }
}

export class StdioTransport implements Transport {
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;
  private readline: readline.Interface;

  constructor() {
    this.readline = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  public async connect(): Promise<void> {
    this.readline.on('line', async (line: string) => {
      if (!this.requestHandler) {
        console.log(JSON.stringify({
          success: false,
          error: 'No handler registered'
        }));
        return;
      }
      try {
        const rawRequest = JSON.parse(line);
        const request = CallToolRequestSchema.parse(rawRequest);
        const response = await this.requestHandler(request);
        console.log(JSON.stringify(response));
      } catch (error) {
        console.log(JSON.stringify({
          success: false,
          error: error instanceof Error ? error.message : 'Invalid request'
        }));
      }
    });

    return Promise.resolve();
  }

  public async disconnect(): Promise<void> {
    this.readline.close();
    return Promise.resolve();
  }

  public onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }
}

export class StdioServerTransportImpl implements Transport {
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;

  public onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }

  async connect(): Promise<void> {
    // Setup stdin/stdout handlers
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', this.handleInput.bind(this));
  }

  async disconnect(): Promise<void> {
    process.stdin.removeAllListeners('data');
  }

  private handleInput(data: string): void {
    if (!this.requestHandler) return;
    try {
      const message = JSON.parse(data);
      // Handle message...
      process.stdout.write(JSON.stringify({ type: 'response', data: message }) + '\n');
    } catch (error) {
      process.stderr.write(`Error handling input: ${error}\n`);
    }
  }


}

export class TitanExpressServer {
  private app: express.Application;
  private server: Server;
  private model: TitanMemoryModel | null = null;
  private memoryVec: tf.Variable | null = null;
  private port: number;

  constructor(port: number = 3000) {
    this.port = port;
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();

    const capabilities: ServerCapabilities = {
      tools: true,
      memory: true
    };

    this.server = new MCPServer(
      'titan-express',
      '0.1.0',
      capabilities
    );

    this.setupHandlers();
  }

  private setupMiddleware() {
    this.app.use(bodyParser.json());
  }

  private setupHandlers() {
    this.server.handleRequest = async (request: CallToolRequest): Promise<CallToolResult> => {
      try {
        switch (request.name) {
          case 'storeMemory': {
            const { subject, relationship, object } = request.parameters as ToolHandlers['storeMemory'];
            // Implementation
            return {
              success: true,
              result: { stored: true }
            };
          }
          case 'recallMemory': {
            const { query } = request.parameters as ToolHandlers['recallMemory'];
            // Implementation
            return {
              success: true,
              result: { results: [] }
            };
          }
          default:
            throw new Error(`Unknown tool: ${request.name}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        return {
          success: false,
          error: `Error: ${errorMessage}`
        };
      }
    };
  }

  private setupRoutes() {
    this.app.get('/status', (req: Request, res: Response) => {
      res.json({
        status: 'ok',
        model: this.model ? 'initialized' : 'not initialized'

      });
    });
  }

  public async start(): Promise<void> {
    // Connect stdio transport
    const stdioTransport = new StdioServerTransportImpl();
    await this.server.connect(stdioTransport);

    // Start HTTP server
    return new Promise((resolve, reject) => {
      const server = this.app.listen(this.port, () => {
        console.log(`Server running on port ${this.port}`);
        resolve();
      });

      server.on('error', (error: Error) => {
        reject(error);
      });
    });
  }

  public async stop(): Promise<void> {
    if (this.memoryVec) {
      this.memoryVec.dispose();
      this.memoryVec = null;
    }
  }
}
