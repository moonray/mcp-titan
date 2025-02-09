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

  constructor(port: number = 0) {
    this.port = port;
    this.app = express();
    this.app.use(bodyParser.json());
    this.memoryPath = path.join(os.homedir(), '.cursor', 'titan-memory');

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

    // Set up error handler
    this.server.onerror = (error) => console.error('[MCP Error]', error);

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

    // Set up memory and cleanup
    this.setupAutomaticMemory();
    process.on('SIGINT', async () => {
      await this.cleanup();
      process.exit(0);
    });
  }

  private async setupAutomaticMemory() {
    try {
      await fs.mkdir(this.memoryPath, { recursive: true });
      if (!this.memoryState) {
        this.memoryState = {
          shortTerm: tf.zeros([768]),
          longTerm: tf.zeros([768]),
          meta: tf.zeros([768])
        };
      }

      // Set up auto-save interval
      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          try {
            if (this.memoryState) {
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
            }
          } catch (error) {
            console.error('Error saving memory state:', error);
          }
        }, 5 * 60 * 1000); // Every 5 minutes

        // Prevent the interval from keeping the process alive
        this.autoSaveInterval.unref();
      }
    } catch (error) {
      console.error('Error setting up automatic memory:', error);
    }
  }

  private async handleToolCall(request: any) {
    switch (request.params.name) {
      case 'init_model': {
        const { inputDim = 768, outputDim = 768 } = request.params.arguments as TitanMemoryTools['init_model'];
        this.model = new TitanMemoryModel({ inputDim, outputDim });

        // Initialize three-tier memory state
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
        if (!this.model || !this.memoryState) {
          throw new Error('Model not initialized');
        }
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
        if (!this.model || !this.memoryState) {
          throw new Error('Model not initialized');
        }
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
        if (!this.memoryState) {
          throw new Error('Memory not initialized');
        }
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
  }

  public async run() {
    try {
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

  private async cleanup(): Promise<void> {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }
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
        console.error('Error saving memory state during cleanup:', error);
      }
    }
  }
}

// Command line entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const config = process.argv[2] ? JSON.parse(process.argv[2]) : {};
  const server = new TitanMemoryServer(config.port || 0);
  server.run().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}
