#!/usr/bin/env node
import '@tensorflow/tfjs-node';  // Import and register the Node.js backend
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, CallToolResultSchema } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import express from 'express';
import bodyParser from 'body-parser';
import {
  ErrorCode
} from '@modelcontextprotocol/sdk/types.js';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor, unwrapTensor } from './types.js';
import path from 'path';
import os from 'os';
import fs from 'fs/promises';

interface TitanMemoryTools {
  process_input: {
    text: string;
    context?: string;
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

  constructor(port: number = 0) {  // Use port 0 to let OS assign random port
    this.port = port;
    this.app = express();
    this.app.use(bodyParser.json());
    this.memoryPath = path.join(os.homedir(), '.cursor', 'titan-memory');

    this.server = new Server({
      name: 'titan-memory',
      version: '0.1.0',
      description: 'Automatic memory-augmented learning for Cursor',
      capabilities: {
        tools: {
          process_input: {
            name: 'process_input',
            description: 'Process input text and update memory state automatically',
            parameters: {
              type: 'object',
              properties: {
                text: {
                  type: 'string',
                  description: 'Input text to process'
                },
                context: {
                  type: 'string',
                  description: 'Optional context information'
                }
              },
              required: ['text']
            }
          },
          get_memory_state: {
            name: 'get_memory_state',
            description: 'Get current memory state and insights',
            parameters: {
              type: 'object',
              properties: {}
            }
          }
        }
      }
    });
    this.setupToolHandlers();
    this.setupAutomaticMemory();

    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.cleanup();
      process.exit(0);
    });
  }

  private async setupAutomaticMemory() {
    try {
      await fs.mkdir(this.memoryPath, { recursive: true });

      // Initialize model with default config if not exists
      if (!this.model) {
        this.model = new TitanMemoryModel({ inputDim: 768, outputDim: 768 });
      }

      // Try to load existing memory
      const memoryFile = path.join(this.memoryPath, 'memory.json');
      try {
        const data = await fs.readFile(memoryFile, 'utf-8');
        const memoryData = JSON.parse(data);
        this.memoryVec = tf.variable(tf.tensor1d(memoryData));
      } catch {
        // Initialize new memory if none exists
        this.memoryVec = tf.variable(tf.zeros([768]));
      }

      // Set up auto-save every 5 minutes
      this.autoSaveInterval = setInterval(async () => {
        await this.saveMemoryState();
      }, 5 * 60 * 1000);
    } catch (error) {
      console.error('Error setting up automatic memory:', error);
    }
  }

  private async saveMemoryState() {
    if (!this.memoryVec) return;

    try {
      const memoryData = Array.from(this.memoryVec.dataSync());
      await fs.writeFile(
        path.join(this.memoryPath, 'memory.json'),
        JSON.stringify(memoryData),
        'utf-8'
      );
    } catch (error) {
      console.error('Error saving memory state:', error);
    }
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        switch (request.params.name) {
          case 'process_input': {
            if (!this.model || !this.memoryVec) {
              throw new Error('Memory system not initialized');
            }

            const { text, context } = request.params.arguments as TitanMemoryTools['process_input'];

            // Convert text to vector using a stable hash function
            const inputVector = tf.tidy(() => {
              const hash = Array.from(text).reduce((h, c) => Math.imul(31, h) + c.charCodeAt(0) | 0, 0);
              return tf.randomNormal([768]).mul(tf.scalar(hash));
            });

            const input = wrapTensor(inputVector);
            const memory = wrapTensor(this.memoryVec);
            const { predicted, newMemory, surprise } = this.model.forward(input, memory);

            // Update memory state
            this.memoryVec.assign(tf.tensor(newMemory.dataSync()));

            // Save memory state after update
            await this.saveMemoryState();

            const result = {
              surprise: surprise.dataSync()[0],
              memoryUpdated: true,
              insight: `Memory updated with new information. Surprise level: ${surprise.dataSync()[0]}`
            };

            [input, memory, predicted, newMemory, surprise, inputVector].forEach(t => t.dispose());

            return {
              content: [{
                type: 'text',
                text: JSON.stringify(result, null, 2)
              }]
            };
          }

          case 'get_memory_state': {
            if (!this.memoryVec) {
              throw new Error('Memory system not initialized');
            }

            const memoryData = Array.from(this.memoryVec.dataSync());
            const stats = {
              mean: tf.mean(this.memoryVec).dataSync()[0],
              std: tf.moments(this.memoryVec).variance.sqrt().dataSync()[0]
            };

            return {
              content: [{
                type: 'text',
                text: JSON.stringify({
                  memoryStats: stats,
                  memorySize: memoryData.length,
                  status: 'active'
                }, null, 2)
              }]
            };
          }

          default: {
            throw new Error(`Unknown tool: ${request.params.name}`);
          }
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        throw new Error(`Error: ${errorMessage}`);
      }
    });
  }

  public async run() {
    // Set up Express routes for MCP
    this.app.get('/', (req, res) => {
      res.json({
        name: 'Titan Memory MCP Server',
        version: '0.1.0',
        description: 'Automatic memory-augmented learning for Cursor',
        status: 'active',
        memoryPath: this.memoryPath
      });
    });

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

    await this.saveMemoryState();

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
