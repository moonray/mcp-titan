#!/usr/bin/env node
import '@tensorflow/tfjs-node';  // Import and register the Node.js backend
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, CallToolResultSchema, ServerCapabilities } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import express from 'express';
import bodyParser from 'body-parser';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor } from './types.js';
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
  protected memoryVec: tf.Variable | null = null;
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
        description: 'Initialize the Titan Memory model',
        parameters: {
          type: 'object',
          properties: {
            inputDim: { type: 'number', description: 'Input dimension' },
            outputDim: { type: 'number', description: 'Output dimension' }
          }
        }
      },
      train_step: {
        name: 'train_step',
        description: 'Perform a training step',
        parameters: {
          type: 'object',
          properties: {
            x_t: { type: 'array', items: { type: 'number' } },
            x_next: { type: 'array', items: { type: 'number' } }
          },
          required: ['x_t', 'x_next']
        }
      },
      forward_pass: {
        name: 'forward_pass',
        description: 'Run a forward pass through the model',
        parameters: {
          type: 'object',
          properties: {
            x: { type: 'array', items: { type: 'number' } }
          },
          required: ['x']
        }
      },
      get_memory_state: {
        name: 'get_memory_state',
        description: 'Get current memory state',
        parameters: {
          type: 'object',
          properties: {}
        }
      }
    };

    const capabilities: ServerCapabilities = {
      tools: {
        list: tools,
        call: true,
        listChanged: true
      }
    };

    this.server = new Server({
      name: 'titan-memory',
      version: '0.1.0',
      description: 'Automatic memory-augmented learning for Cursor',
      capabilities
    });

    // Set up error handler
    this.server.onerror = (error) => console.error('[MCP Error]', error);

    // Register capabilities before setting up handlers
    this.server.registerCapabilities(capabilities);

    // Set up tool handlers
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        switch (request.params.name) {
          case 'init_model': {
            const { inputDim = 768, outputDim = 768 } = request.params.arguments as TitanMemoryTools['init_model'];
            this.model = new TitanMemoryModel({ inputDim, outputDim });
            return {
              content: [{
                type: 'text',
                text: JSON.stringify({ config: { inputDim, outputDim } })
              }]
            };
          }
          case 'train_step': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }
            const { x_t, x_next } = request.params.arguments as TitanMemoryTools['train_step'];
            const x_tT = wrapTensor(tf.tensor1d(x_t));
            const x_nextT = wrapTensor(tf.tensor1d(x_next));
            const memoryT = wrapTensor(this.memoryVec);
            const cost = this.model.trainStep(x_tT, x_nextT, memoryT);
            const { predicted, newMemory, surprise } = this.model.forward(x_tT, memoryT);
            const result = {
              cost: cost.dataSync()[0],
              predicted: Array.from(predicted.dataSync()),
              surprise: surprise.dataSync()[0]
            };
            [x_tT, x_nextT, memoryT, predicted, newMemory, surprise, cost].forEach(t => t.dispose());
            return {
              content: [{
                type: 'text',
                text: JSON.stringify(result)
              }]
            };
          }
          case 'forward_pass': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Model not initialized');
            }
            const { x } = request.params.arguments as TitanMemoryTools['forward_pass'];
            const xT = wrapTensor(tf.tensor1d(x));
            const memoryT = wrapTensor(this.memoryVec);
            const { predicted, newMemory, surprise } = this.model.forward(xT, memoryT);
            const result = {
              predicted: Array.from(predicted.dataSync()),
              memory: Array.from(newMemory.dataSync()),
              surprise: surprise.dataSync()[0]
            };
            [xT, memoryT, predicted, newMemory, surprise].forEach(t => t.dispose());
            return {
              content: [{
                type: 'text',
                text: JSON.stringify(result)
              }]
            };
          }
          case 'get_memory_state': {
            if (!this.memoryVec) {
              throw new Error('Memory not initialized');
            }
            const stats = {
              mean: tf.mean(this.memoryVec).dataSync()[0],
              std: tf.moments(this.memoryVec).variance.sqrt().dataSync()[0]
            };
            return {
              content: [{
                type: 'text',
                text: JSON.stringify({
                  memoryStats: stats,
                  memorySize: this.memoryVec.shape[0],
                  status: 'active'
                })
              }]
            };
          }
          default:
            throw new Error(`Unknown tool: ${request.params.name}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        throw new Error(`Error: ${errorMessage}`);
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
      if (!this.memoryVec) {
        this.memoryVec = tf.variable(tf.zeros([768]));
      }
    } catch (error) {
      console.error('Error setting up automatic memory:', error);
    }
  }

  public async run() {
    // Connect stdio for Cursor
    const stdioTransport = new StdioServerTransport();
    await this.server.connect(stdioTransport);

    // Start HTTP server with dynamic port
    const server = this.app.listen(this.port, () => {
      this.port = (server.address() as any).port;
      console.log(`Titan Memory MCP server running on port ${this.port}`);
    });

    // Handle server errors
    server.on('error', (error: any) => {
      if (error.code === 'EADDRINUSE') {
        console.error(`Port ${this.port} is already in use, trying another port...`);
        server.listen(0); // Let OS assign random port
      } else {
        console.error('Server error:', error);
      }
    });
  }

  private async cleanup(): Promise<void> {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
    }

    if (this.memoryVec) {
      this.memoryVec.dispose();
      this.memoryVec = null;
    }
  }
}

// Create and run server instance if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  const server = new TitanMemoryServer();
  server.run().catch(console.error);
}
