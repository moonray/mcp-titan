#!/usr/bin/env node
import '@tensorflow/tfjs-node';  // Import and register the Node.js backend
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, CallToolResultSchema, ServerCapabilities } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import express from 'express';
import bodyParser from 'body-parser';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor, unwrapTensor, ITensor, IMemoryState } from './types.js';
import path from 'path';
import os from 'os';
import fs from 'fs/promises';
import { WebSocketServerTransport } from '@modelcontextprotocol/sdk/server/websocket.js';
import WebSocket from 'ws';

interface TitanMemoryTools {
  init_model: {
    inputDim?: number;
    outputDim?: number;
  };
  train_step: {
    x_t: number[];
    x_next: number[];
  };
  forward_pass: {
    x: number[];
  };
  get_memory_state: Record<string, never>;
}

export class TitanMemoryServer {
  protected server: Server;
  protected model: TitanMemoryModel | null = null;
  protected memoryState: IMemoryState | null = null;
  private app: express.Application;
  private port: number;
  private memoryPath: string;
  private autoSaveInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts: number = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private readonly RECONNECT_DELAY = 5000;
  private wsServer: WebSocket.Server | null = null;
  private readonly DEFAULT_WS_PORT = 3000;
  private readonly AUTO_RECONNECT_INTERVAL = 5000;
  private isAutoReconnectEnabled = true;

  constructor(port: number = 0) {
    this.port = port;
    this.app = express();
    this.app.use(bodyParser.json());

    // Use a more robust path for memory storage
    this.memoryPath = path.join(
      os.platform() === 'win32' ? process.env.APPDATA || os.homedir() : os.homedir(),
      '.mcp-titan'
    );

    const tools = {
      init_model: {
        name: 'init_model',
        description: 'Initialize the Titan Memory model for learning code patterns',
        parameters: {
          type: 'object',
          properties: {
            inputDim: { type: 'number', description: 'Size of input vectors (default: 768)' },
            outputDim: { type: 'number', description: 'Size of memory state (default: 768)' }
          }
        },
        function: async (params: any) => {
          return this.handleToolCall({ params: { name: 'init_model', arguments: params } });
        }
      },
      train_step: {
        name: 'train_step',
        description: 'Train the model on a sequence of code to improve pattern recognition',
        parameters: {
          type: 'object',
          properties: {
            x_t: { type: 'array', items: { type: 'number' }, description: 'Current code state vector' },
            x_next: { type: 'array', items: { type: 'number' }, description: 'Next code state vector' }
          },
          required: ['x_t', 'x_next']
        },
        function: async (params: any) => {
          return this.handleToolCall({ params: { name: 'train_step', arguments: params } });
        }
      },
      forward_pass: {
        name: 'forward_pass',
        description: 'Predict the next likely code pattern based on current input',
        parameters: {
          type: 'object',
          properties: {
            x: { type: 'array', items: { type: 'number' }, description: 'Current code state vector' }
          },
          required: ['x']
        },
        function: async (params: any) => {
          return this.handleToolCall({ params: { name: 'forward_pass', arguments: params } });
        }
      },
      get_memory_state: {
        name: 'get_memory_state',
        description: 'Get insights about what patterns the model has learned',
        parameters: {
          type: 'object',
          properties: {}
        },
        function: async (params: any) => {
          return this.handleToolCall({ params: { name: 'get_memory_state', arguments: params } });
        }
      }
    };

    const capabilities: ServerCapabilities = {
      tools: {
        list: tools,
        call: {
          request: CallToolRequestSchema,
          result: CallToolResultSchema
        },
        listChanged: true
      }
    };

    this.server = new Server({
      name: 'mcp-titan',
      version: '0.1.2',
      description: 'AI-powered code memory that learns from your coding patterns to provide better suggestions and insights',
      capabilities
    });

    // Enhanced error handler with reconnection logic
    this.server.onerror = async (error) => {
      console.error('[MCP Error]', error);
      if (this.isAutoReconnectEnabled && this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS) {
        this.reconnectAttempts++;
        console.error(`Attempting reconnection (${this.reconnectAttempts}/${this.MAX_RECONNECT_ATTEMPTS})...`);
        setTimeout(() => this.reconnect(), this.RECONNECT_DELAY);
      }
    };

    // Register capabilities and handlers
    this.server.registerCapabilities(capabilities);
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        const result = await this.handleToolCall(request);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify(result)
          }]
        };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        throw new Error(`Error handling tool call: ${errorMessage}`);
      }
    });

    // Auto-initialize immediately
    this.autoInitialize().catch(error => {
      console.error('Auto-initialization failed:', error);
    });
  }

  private async setupWebSocket() {
    if (!this.wsServer) {
      const wsPort = this.port || this.DEFAULT_WS_PORT;
      this.wsServer = new WebSocket.Server({ port: wsPort });

      this.wsServer.on('connection', async (ws) => {
        try {
          const transport = new WebSocketServerTransport(ws);
          await this.server.connect(transport);
          console.error(`WebSocket client connected on port ${wsPort}`);
        } catch (error) {
          console.error('WebSocket connection error:', error);
        }
      });

      this.wsServer.on('error', (error) => {
        console.error('WebSocket server error:', error);
        if (this.isAutoReconnectEnabled) {
          setTimeout(() => this.setupWebSocket(), this.AUTO_RECONNECT_INTERVAL);
        }
      });
    }
  }

  private async autoInitialize() {
    try {
      // Ensure memory directory exists
      await fs.mkdir(this.memoryPath, { recursive: true });

      // Initialize model if needed
      if (!this.model) {
        this.model = new TitanMemoryModel({ inputDim: 768, outputDim: 768 });

        // Initialize memory state
        const zeros = tf.zeros([768]);
        this.memoryState = {
          shortTerm: wrapTensor(zeros),
          longTerm: wrapTensor(zeros.clone()),
          meta: wrapTensor(zeros.clone())
        };
        zeros.dispose();

        // Try to load saved state
        await this.loadSavedState();
      }

      // Setup automatic memory saving
      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          await this.saveMemoryState();
        }, 5 * 60 * 1000);
        this.autoSaveInterval.unref();
      }

      // Setup WebSocket server
      await this.setupWebSocket();
    } catch (error) {
      console.error('Auto-initialization failed:', error);
      throw error;
    }
  }

  private async loadSavedState() {
    try {
      const statePath = path.join(this.memoryPath, 'memory.json');
      const exists = await fs.access(statePath).then(() => true).catch(() => false);

      if (exists) {
        const savedState = JSON.parse(await fs.readFile(statePath, 'utf-8'));
        if (savedState && savedState.shortTerm) {
          this.memoryState = {
            shortTerm: wrapTensor(tf.tensor1d(savedState.shortTerm)),
            longTerm: wrapTensor(tf.tensor1d(savedState.longTerm)),
            meta: wrapTensor(tf.tensor1d(savedState.meta))
          };
        }
      }
    } catch (error) {
      console.error('Error loading saved state:', error);
    }
  }

  private async saveMemoryState() {
    if (this.memoryState) {
      try {
        const memoryState = {
          shortTerm: Array.from(this.memoryState.shortTerm.dataSync()),
          longTerm: Array.from(this.memoryState.longTerm.dataSync()),
          meta: Array.from(this.memoryState.meta.dataSync()),
          timestamp: Date.now()
        };
        await fs.writeFile(
          path.join(this.memoryPath, 'memory.json'),
          JSON.stringify(memoryState),
          'utf-8'
        );
      } catch (error) {
        console.error('Error saving memory state:', error);
      }
    }
  }

  private async reconnect() {
    try {
      const transport = new StdioServerTransport();
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

  private async handleToolCall(request: any) {
    try {
      // Ensure model is initialized
      if (!this.model || !this.memoryState) {
        await this.autoInitialize();
      }

      this.assertModelInitialized();

      // Wrap all tensor operations in tidy
      return tf.tidy(() => {
        switch (request.params.name) {
          case 'init_model': {
            const { inputDim = 768, outputDim = 768 } = request.params.arguments as TitanMemoryTools['init_model'];
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
            const { x_t, x_next } = request.params.arguments as TitanMemoryTools['train_step'];
            const x_tT = wrapTensor(tf.tensor1d(x_t));
            const x_nextT = wrapTensor(tf.tensor1d(x_next));
            const cost = this.model.trainStep(x_tT, x_nextT, this.memoryState);
            const { predicted, memoryUpdate } = this.model.forward(x_tT, this.memoryState);

            // Update memory state
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
            const { x } = request.params.arguments as TitanMemoryTools['forward_pass'];
            const xT = wrapTensor(tf.tensor1d(x));
            const { predicted, memoryUpdate } = this.model.forward(xT, this.memoryState);

            // Update memory state
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
              mean: tf.mean(this.memoryState.shortTerm).dataSync()[0],
              std: tf.moments(this.memoryState.shortTerm).variance.sqrt().dataSync()[0]
            };
            return {
              memoryStats: stats,
              memorySize: this.memoryState.shortTerm.shape[0],
              status: 'active'
            };
          }
          default:
            throw new Error(`Unknown tool: ${request.params.name}`);
        }
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      console.error('Tool call error:', errorMessage);
      throw new Error(`Error handling tool call: ${errorMessage}`);
    }
  }

  public async run() {
    try {
      // Ensure initialization
      await this.autoInitialize();

      // Set up stdio transport for MCP communication
      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      // Start express server if port is specified
      if (this.port > 0) {
        await new Promise<void>((resolve) => {
          this.app.listen(this.port, () => {
            console.error(`HTTP server listening on port ${this.port}`);
            resolve();
          });
        });
      }
    } catch (error) {
      console.error('Error starting server:', error);
      throw error;
    }
  }
}

// Command line entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const config = process.argv[2] ? JSON.parse(process.argv[2]) : {};
  const server = new TitanMemoryServer(config.port || 0);

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
