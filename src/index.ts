#!/usr/bin/env node
import '@tensorflow/tfjs-node';  // Import and register the Node.js backend
import * as tf from '@tensorflow/tfjs-node';
import express from 'express';
import bodyParser from 'body-parser';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs/promises';
import WebSocket from 'ws';
import { TitanMemoryModel } from './model.js';
import {
  IMemoryState,
  wrapTensor,
  unwrapTensor,
  ServerCapabilities,
  CallToolRequest,
  CallToolResult,
  Server,
  Transport
} from './types.js';
import { MCPServer, WebSocketTransport, StdioServerTransportImpl } from './server.js';
import { StdioTransport } from './transports.js';

import { z } from 'zod';

// Tool interfaces
interface ToolParameter {
  type: string;
  description?: string;
  items?: {
    type: string;
  };
}

interface ToolParameters {
  type: string;
  properties: Record<string, ToolParameter>;
  required?: string[];
}

interface Tool {
  name: string;
  description: string;
  parameters: ToolParameters;
}

interface TitanMemoryTools {
  init_model: Tool;
  train_step: Tool;
  forward_pass: Tool;
  get_memory_state: Tool;
}

// Error handler interface
interface MCPError {
  message: string;
  code: string;
}

// Memory configuration interface
interface TitanMemoryConfig {
  port?: number;
  memoryPath?: string;
  modelPath?: string;
  weightsPath?: string;
  inputDim?: number;
  outputDim?: number;
}

// Request handler interface
interface ToolCallRequest extends CallToolRequest {
  params: Record<string, unknown>;
}

// Result interface
interface ToolCallResult extends Omit<CallToolResult, 'error'> {
  error?: string;
}

const MCPErrorSchema = z.object({
  message: z.string(),
  code: z.string()
});

export class TitanServer implements Server {
  public capabilities: ServerCapabilities = {
    tools: true,
    memory: true
  };

  private transport: WebSocketTransport | StdioTransport;

  constructor(transport: WebSocketTransport | StdioTransport) {
    this.transport = transport;
  }

  async connect(transport: Transport): Promise<void> {
    await transport.connect();
  }

  async start(): Promise<void> {
    await this.transport.connect();
    this.transport.onRequest(this.handleRequest.bind(this));
  }

  async stop(): Promise<void> {
    await this.transport.disconnect();
  }

  async handleRequest(request: CallToolRequest): Promise<CallToolResult> {
    try {
      const result = await this.processRequest(request);
      return {
        success: true,
        result
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  private async processRequest(request: CallToolRequest): Promise<unknown> {
    switch (request.name) {
      case 'storeMemory':
        return { stored: true };
      case 'recallMemory':
        return { recalled: true };
      default:
        throw new Error(`Unknown tool: ${request.name}`);
    }
  }
}

export class TitanMemoryServer {
  private model!: TitanMemoryModel;
  private server: MCPServer;
  private tools: TitanMemoryTools;
  private memoryState!: IMemoryState;
  private app: express.Application;
  private port: number;
  private memoryPath: string;
  private modelPath: string;
  private weightsPath: string;
  private autoSaveInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private readonly RECONNECT_DELAY = 5000;
  private wsServer: WebSocket.Server | null = null;
  private readonly DEFAULT_WS_PORT = 3000;
  private readonly AUTO_RECONNECT_INTERVAL = 5000;
  private isAutoReconnectEnabled = true;

  constructor(config: TitanMemoryConfig = {}) {
    this.port = config.port || 0;
    this.memoryPath = config.memoryPath || path.join(
      os.platform() === 'win32' ? process.env.APPDATA || os.homedir() : os.homedir(),
      '.mcp-titan'
    );
    this.modelPath = config.modelPath || path.join(this.memoryPath, 'model.json');
    this.weightsPath = config.weightsPath || path.join(this.memoryPath, 'weights');

    this.app = express();
    this.app.use(bodyParser.json());

    this.tools = {
      init_model: this.createInitModelTool(),
      train_step: this.createTrainStepTool(),
      forward_pass: this.createForwardPassTool(),
      get_memory_state: this.createGetMemoryStateTool()
    };

    this.server = new MCPServer('titan-memory', '1.0.0', {
      tools: true,
      memory: true
    });

    this.setupErrorHandler();
    this.setupToolHandler();
    this.autoInitialize().catch(console.error);
  }

  private setupErrorHandler(): void {
    this.server.onError((error: Error) => {
      console.error('Server error:', error);
    });
  }

  private setupToolHandler(): void {
    this.server.onToolCall(async (request: CallToolRequest): Promise<CallToolResult> => {
      try {
        const result = await this.handleToolCall(request as ToolCallRequest);
        return { success: true, result };
      } catch (error) {
        const mcpError = error as Error;
        return {
          success: false,
          error: mcpError.message
        };
      }
    });
  }

  private async setupWebSocket(): Promise<void> {
    const transport = new WebSocketTransport(this.port.toString());
    await this.server.connect(transport);
  }

  private createInitModelTool(): Tool {
    return {
      name: 'init_model',
      description: 'Initialize the Titan Memory model for learning code patterns',
      parameters: {
        type: 'object',
        properties: {
          inputDim: { type: 'number', description: 'Size of input vectors (default: 768)' },
          outputDim: { type: 'number', description: 'Size of memory state (default: 768)' }
        }
      }
    };
  }

  private createTrainStepTool(): Tool {
    return {
      name: 'train_step',
      description: 'Train the model on a sequence of code to improve pattern recognition',
      parameters: {
        type: 'object',
        properties: {
          x_t: { type: 'array', items: { type: 'number' }, description: 'Current code state vector' },
          x_next: { type: 'array', items: { type: 'number' }, description: 'Next code state vector' }
        },
        required: ['x_t', 'x_next']
      }
    };
  }

  private createForwardPassTool(): Tool {
    return {
      name: 'forward_pass',
      description: 'Predict the next likely code pattern based on current input',
      parameters: {
        type: 'object',
        properties: {
          x: { type: 'array', items: { type: 'number' }, description: 'Current code state vector' }
        },
        required: ['x']
      }
    };
  }

  private createGetMemoryStateTool(): Tool {
    return {
      name: 'get_memory_state',
      description: 'Get insights about what patterns the model has learned',
      parameters: {
        type: 'object',
        properties: {}
      }
    };
  }

  private async handleToolCall(request: ToolCallRequest): Promise<unknown> {
    if (!this.model || !this.memoryState) {
      await this.autoInitialize();
    }

    return tf.tidy(() => {
      const args = request.params as Record<string, unknown>;

      switch (request.name) {
        case 'init_model': {
          const inputDim = (args.inputDim as number) || 768;
          const outputDim = (args.outputDim as number) || 768;

          this.model = new TitanMemoryModel({ inputDim, outputDim });
          const zeros = tf.zeros([outputDim]);
          this.memoryState = {
            shortTerm: wrapTensor(zeros),
            longTerm: wrapTensor(zeros.clone()),
            meta: wrapTensor(zeros.clone())
          };
          zeros.dispose();
          return { config: { inputDim, outputDim } };
        }

        case 'train_step': {
          const x_t = args.x_t as number[];
          const x_next = args.x_next as number[];

          if (!x_t || !x_next) {
            throw new Error('Missing required parameters x_t or x_next');
          }

          const x_tT = wrapTensor(tf.tensor1d(x_t));
          const x_nextT = wrapTensor(tf.tensor1d(x_next));
          const cost = this.model.trainStep(x_tT, x_nextT, this.memoryState);
          const { predicted, memoryUpdate } = this.model.forward(x_tT, this.memoryState);

          this.memoryState = memoryUpdate.newState;

          const result = {
            cost: unwrapTensor(cost.loss).dataSync()[0],
            predicted: Array.from(unwrapTensor(predicted).dataSync()),
            surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
          };

          [x_tT, x_nextT, predicted].forEach(t => t.dispose());
          return result;
        }

        case 'forward_pass': {
          const x = args.x as number[];

          if (!x) {
            throw new Error('Missing required parameter x');
          }

          const xT = wrapTensor(tf.tensor1d(x));
          const { predicted, memoryUpdate } = this.model.forward(xT, this.memoryState);

          this.memoryState = memoryUpdate.newState;

          const result = {
            predicted: Array.from(unwrapTensor(predicted).dataSync()),
            memory: Array.from(unwrapTensor(memoryUpdate.newState.shortTerm).dataSync()),
            surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
          };

          [xT, predicted].forEach(t => t.dispose());
          return result;
        }

        case 'get_memory_state': {
          const stats = {
            mean: tf.mean(unwrapTensor(this.memoryState.shortTerm)).dataSync()[0],
            std: tf.moments(unwrapTensor(this.memoryState.shortTerm)).variance.sqrt().dataSync()[0]
          };

          return {
            memoryStats: stats,
            memorySize: this.memoryState.shortTerm.shape[0],
            status: 'active'
          };
        }

        default:
          throw new Error(`Unknown tool: ${request.name}`);
      }
    });
  }

  private async autoInitialize(): Promise<void> {
    try {
      // Create memory directory if it doesn't exist
      await fs.mkdir(this.memoryPath, { recursive: true });

      // Check if we have saved state
      const hasSavedState = await this.loadSavedState();

      if (!hasSavedState) {
        // Initialize new model with default settings
        this.model = new TitanMemoryModel({
          inputDim: 768,
          outputDim: 768
        });

        // Initialize memory state with zeros
        const zeros = tf.zeros([768]);
        this.memoryState = {
          shortTerm: wrapTensor(zeros),
          longTerm: wrapTensor(zeros.clone()),
          meta: wrapTensor(zeros.clone())
        };
        zeros.dispose();

        // Save initial state
        await this.saveMemoryState();
      }

      // Set up auto-save interval
      if (this.autoSaveInterval === null) {
        this.autoSaveInterval = setInterval(() => {
          this.saveMemoryState().catch(console.error);
        }, 5 * 60 * 1000); // Auto-save every 5 minutes
      }
    } catch (error) {
      console.error('Failed to initialize model:', error);
      throw error;
    }
  }

  private async loadSavedState(): Promise<boolean> {
    try {
      // Check if model and weights files exist
      const modelExists = await fs.access(this.modelPath)
        .then(() => true)
        .catch(() => false);
      const weightsExist = await fs.access(this.weightsPath)
        .then(() => true)
        .catch(() => false);

      if (!modelExists || !weightsExist) {
        return false;
      }

      // Load model and weights
      this.model = new TitanMemoryModel();
      await this.model.save(this.modelPath, this.weightsPath);

      // Load memory state
      const memoryStateJson = await fs.readFile(
        path.join(this.memoryPath, 'memory_state.json'),
        'utf8'
      );
      const memoryState = JSON.parse(memoryStateJson);
      this.memoryState = {
        shortTerm: wrapTensor(tf.tensor(memoryState.shortTerm)),
        longTerm: wrapTensor(tf.tensor(memoryState.longTerm)),
        meta: wrapTensor(tf.tensor(memoryState.meta))
      };

      return true;
    } catch (error) {
      console.error('Failed to load saved state:', error);
      return false;
    }
  }

  private async saveMemoryState(): Promise<void> {
    try {
      // Save model and weights
      await this.model.save(this.modelPath, this.weightsPath);

      // Save memory state
      const memoryState = {
        shortTerm: await unwrapTensor(this.memoryState.shortTerm).array(),
        longTerm: await unwrapTensor(this.memoryState.longTerm).array(),
        meta: await unwrapTensor(this.memoryState.meta).array()
      };

      await fs.writeFile(
        path.join(this.memoryPath, 'memory_state.json'),
        JSON.stringify(memoryState, null, 2)
      );
    } catch (error) {
      console.error('Failed to save memory state:', error);
      throw error;
    }
  }

  private async reconnect() {
    try {
      const transport = new StdioServerTransportImpl();
      await this.server.connect(transport);
      this.reconnectAttempts = 0;
      console.error('Successfully reconnected to MCP');
    } catch (error) {
      console.error('Reconnection failed:', error);
    }
  }

  private assertModelInitialized(): asserts this is { model: TitanMemoryModel, memoryState: IMemoryState } {
    if (!this.model || !this.memoryState) {
      throw new Error('Model not initialized');
    }
  }

  public async cleanup(): Promise<void> {
    this.isAutoReconnectEnabled = false;

    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }

    if (this.wsServer) {
      await new Promise<void>((resolve) => {
        this.wsServer?.close(() => resolve());
      });
    }

    await this.saveMemoryState();
  }

  public async run(): Promise<void> {
    await this.setupWebSocket();
    console.log('Titan Memory Server running...');
  }
}

// Command line entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const config = process.argv[2] ? JSON.parse(process.argv[2]) : {};
  const server = new TitanMemoryServer({
    port: config.port || 0,
    memoryPath: config.memoryPath,
    modelPath: config.modelPath,
    weightsPath: config.weightsPath,
    inputDim: config.inputDim,
    outputDim: config.outputDim
  });

  // Handle process signals
  process.on('SIGINT', async () => {
    await server.cleanup();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    await server.cleanup();
    process.exit(0);
  });

  // Start the server
  server.run().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}

// Example usage
async function main() {
  // Create a WebSocket transport
  const wsTransport = new WebSocketTransport('ws://localhost:8080');
  const server = new TitanServer(wsTransport);
  await server.start();

  // Create a stdio transport
  const stdioTransport = new StdioTransport();
  const stdioServer = new TitanServer(stdioTransport);
  await stdioServer.start();
}

if (require.main === module) {
  main().catch(console.error);
}
