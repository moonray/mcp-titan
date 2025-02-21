// Import polyfills first
import './utils/polyfills.js';

// Henry's Titan Memory Server
import { z } from "zod";
import * as tf from '@tensorflow/tfjs-node';
import type { TensorContainer } from '@tensorflow/tfjs-core';
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { RequestHandlerExtra } from "@modelcontextprotocol/sdk/shared/protocol.js";

import { IMemoryState, wrapTensor, unwrapTensor } from './types.js';
import { TitanMemoryModel } from './model.js';
import { VectorProcessor } from './utils.js';
import * as path from 'path';
import { promises as fs } from 'fs';
import * as crypto from 'crypto';

interface SerializedMemoryState {
  shortTerm: number[];
  longTerm: number[];
  meta: number[];
  timestamps: number[];
  accessCounts: number[];
  surpriseHistory: number[];
}

interface MemoryStats {
  shortTermMean: number;
  shortTermStd: number;
  longTermMean: number;
  longTermStd: number;
  capacity: number;
}
interface ToolResponse {
  [key: string]: unknown;
  content: Array<{
    [key: string]: unknown;
    type: "text";
    text: string;
  }>;
  _meta?: Record<string, unknown>;
  isError?: boolean;
}
export class TitanMemoryServer {
  private server: McpServer;
  private model!: TitanMemoryModel;
  private vectorProcessor: VectorProcessor;
  private memoryState: IMemoryState;
  private isInitialized = false;
  private autoSaveInterval?: NodeJS.Timeout;
  private readonly memoryPath: string;
  private readonly modelPath: string;
  private readonly weightsPath: string;

  constructor(options: { memoryPath?: string } = {}) {
    this.server = new McpServer({
      name: "Titan Memory",
      version: "1.2.0"
    });
    this.vectorProcessor = VectorProcessor.getInstance();
    this.memoryPath = options.memoryPath || path.join(process.cwd(), '.titan_memory');
    this.modelPath = path.join(this.memoryPath, 'model.json');
    this.weightsPath = path.join(this.memoryPath, 'model.weights.bin');
    this.memoryState = this.initializeEmptyState();

    this.registerTools();
  }

  private initializeEmptyState(): IMemoryState {
    return tf.tidy(() => ({
      shortTerm: wrapTensor(tf.zeros([0])),
      longTerm: wrapTensor(tf.zeros([0])),
      meta: wrapTensor(tf.zeros([0])),
      timestamps: wrapTensor(tf.zeros([0])),
      accessCounts: wrapTensor(tf.zeros([0])),
      surpriseHistory: wrapTensor(tf.zeros([0]))
    }));
  }

  private wrapWithMemoryManagement<T extends TensorContainer>(fn: () => T): T {
    return tf.tidy(fn);
  }

  private async wrapWithMemoryManagementAsync<T extends TensorContainer>(fn: () => Promise<T>): Promise<T> {
    tf.engine().startScope();
    try {
      return await fn();
    } finally {
      tf.engine().endScope();
    }
  }

  private encryptTensor(tensor: tf.Tensor): Uint8Array {
    const data = tensor.dataSync();
    const key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    const encrypted = Buffer.concat([cipher.update(Buffer.from(data.buffer)), cipher.final()]);
    return new Uint8Array(Buffer.concat([iv, key, encrypted]));
  }

  private validateMemoryState(state: IMemoryState): boolean {
    return tf.tidy(() => {
      try {
        const validations = [
          state.shortTerm && !unwrapTensor(state.shortTerm).isDisposed,
          state.longTerm && !unwrapTensor(state.longTerm).isDisposed,
          state.meta && !unwrapTensor(state.meta).isDisposed,
          state.timestamps && !unwrapTensor(state.timestamps).isDisposed,
          state.accessCounts && !unwrapTensor(state.accessCounts).isDisposed,
          state.surpriseHistory && !unwrapTensor(state.surpriseHistory).isDisposed
        ];

        return validations.every(Boolean);
      } catch (error) {
        console.warn('Error validating memory state:', error);
        return false;
      }
    });
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.autoInitialize();
      this.isInitialized = true;
    }
  }

  private registerTools(): void {
    // Help tool
    this.server.tool(
      'help',
      {
        tool: z.string().optional(),
        category: z.string().optional(),
        showExamples: z.boolean().optional(),
        verbose: z.boolean().optional()
      },
      async (params, extra) => {
        await this.ensureInitialized();
        return {
          content: [{
            type: "text",
            text: "Available tools:\n" +
              "- help: Get help about available tools\n" +
              "- init_model: Initialize the Titan Memory model\n" +
              "- forward_pass: Perform a forward pass through the model\n" +
              "- train_step: Execute a training step\n" +
              "- get_memory_state: Get current memory state\n" +
              "- save_memory_state: Save memory state to file\n" +
              "- load_memory_state: Load memory state from file"
          }]
        };
      }
    );

    // Init model tool
    this.server.tool(
      'init_model',
      {
        inputDim: z.number().int().positive(),
        memorySlots: z.number().int().positive(),
        transformerLayers: z.number().int().positive()
      },
      async (params, extra) => {
        await this.ensureInitialized();
        const config = {
          inputDim: params.inputDim,
          memorySlots: params.memorySlots,
          transformerLayers: params.transformerLayers
        };
        await this.model.initialize(config);
        return {
          content: [{
            type: "text",
            text: `Model initialized with configuration: ${JSON.stringify(config)}`
          }]
        };
      }
    );

    // Forward pass tool
    this.server.tool(
      'forward_pass',
      {
        x: z.array(z.number())
      },
      async (params, extra) => {
        await this.ensureInitialized();
        const input = tf.tensor2d([params.x]);
        const result = await this.model.forward(wrapTensor(input), this.memoryState);
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              predicted: Array.from(unwrapTensor(result.predicted).dataSync()),
              surprise: Array.from(unwrapTensor(result.memoryUpdate.surprise.immediate).dataSync())
            })
          }]
        };
      }
    );

    // Train step tool
    this.server.tool(
      'train_step',
      {
        x_t: z.array(z.number()),
        x_next: z.array(z.number())
      },
      async (params, extra) => {
        await this.ensureInitialized();
        const x_t = tf.tensor2d([params.x_t]);
        const x_next = tf.tensor2d([params.x_next]);
        const result = await this.model.trainStep(wrapTensor(x_t), wrapTensor(x_next), this.memoryState);
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              loss: Array.from(unwrapTensor(result.loss).dataSync())
            })
          }]
        };
      }
    );

    // Get memory state tool
    this.server.tool(
      'get_memory_state',
      {},
      async (params, extra) => {
        await this.ensureInitialized();
        const state = this.model.getMemorySnapshot();
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              shortTerm: Array.from(state.shortTerm.dataSync()),
              longTerm: Array.from(state.longTerm.dataSync()),
              meta: Array.from(state.meta.dataSync())
            })
          }]
        };
      }
    );

    // Save memory state tool
    this.server.tool(
      'save_memory_state',
      {
        path: z.string()
      },
      async (params, extra) => {
        await this.ensureInitialized();
        await this.model.saveModel(params.path);
        return {
          content: [{
            type: "text",
            text: `Memory state saved to ${params.path}`
          }]
        };
      }
    );

    // Load memory state tool
    this.server.tool(
      'load_memory_state',
      {
        path: z.string()
      },
      async (params, extra) => {
        await this.ensureInitialized();
        await this.model.loadModel(params.path);
        return {
          content: [{
            type: "text",
            text: `Memory state loaded from ${params.path}`
          }]
        };
      }
    );
  }

  private async processInput(input: string | number[] | tf.Tensor | number): Promise<tf.Tensor> {
    return tf.tidy(() => {
      try {
        if (typeof input === 'string') {
          // Convert string to bytes and pad/truncate to exactly 768 elements
          const encoder = new TextEncoder();
          const bytes = encoder.encode(input);
          const paddedArray = new Float32Array(768).fill(0);

          // Copy bytes and normalize to [0,1]
          for (let i = 0; i < Math.min(bytes.length, 768); i++) {
            paddedArray[i] = bytes[i] / 255;
          }

          return tf.tensor1d(paddedArray);
        } else if (Array.isArray(input)) {
          // Pad or truncate array to exactly 768 elements
          const paddedArray = new Float32Array(768).fill(0);
          const inputArray = input.slice(0, 768);
          paddedArray.set(inputArray);
          return tf.tensor1d(paddedArray);
        } else if (input instanceof tf.Tensor) {
          // Ensure tensor is flattened and has exactly 768 elements
          const flattened = input.flatten();
          const inputData = flattened.dataSync();
          const paddedData = new Float32Array(768).fill(0);
          paddedData.set(Array.from(inputData).slice(0, 768));

          // Clean up intermediate tensor
          if (flattened !== input) {
            flattened.dispose();
          }

          return tf.tensor1d(paddedData);
        } else if (typeof input === 'number') {
          // Create a tensor of 768 elements filled with the input number
          return tf.ones([768]).mul(tf.scalar(input));
        }
        throw new Error('Invalid input type');
      } catch (error) {
        console.error('Error processing input:', error);
        throw error;
      }
    });
  }

  private async autoInitialize(): Promise<void> {
    try {
      // Initialize TensorFlow.js backend first
      await tf.ready();
      await tf.setBackend('tensorflow');

      // Verify backend is ready
      const backend = tf.getBackend();
      if (!backend) {
        throw new Error('Failed to initialize TensorFlow.js backend');
      }

      console.log('TensorFlow backend initialized:', backend);

      // Ensure memory directory exists
      await fs.mkdir(this.memoryPath, { recursive: true });

      // Initialize model with backend verification
      this.model = new TitanMemoryModel({
        inputDim: 768,
        memorySlots: 5000,
        transformerLayers: 6
      });

      // Initialize the model (this will set up the backend)
      await this.model.initialize();

      // Try to load saved state if exists
      try {
        const [modelExists, weightsExist, memoryStateExists] = await Promise.all([
          fs.access(this.modelPath).then(() => true).catch(() => false),
          fs.access(this.weightsPath).then(() => true).catch(() => false),
          fs.access(path.join(this.memoryPath, 'memory_state.json')).then(() => true).catch(() => false)
        ]);

        if (modelExists && weightsExist) {
          console.log('Found existing model and weights, loading...');
          await this.model.loadModel(this.modelPath);

          if (memoryStateExists) {
            console.log('Found existing memory state, loading...');
            const memoryStateJson = await fs.readFile(
              path.join(this.memoryPath, 'memory_state.json'),
              'utf8'
            );
            const memoryState = JSON.parse(memoryStateJson) as SerializedMemoryState;

            // Load memory state within a tidy block
            this.memoryState = tf.tidy(() => ({
              shortTerm: wrapTensor(tf.tensor1d(memoryState.shortTerm)),
              longTerm: wrapTensor(tf.tensor1d(memoryState.longTerm)),
              meta: wrapTensor(tf.tensor1d(memoryState.meta)),
              timestamps: wrapTensor(tf.tensor1d(memoryState.timestamps)),
              accessCounts: wrapTensor(tf.tensor1d(memoryState.accessCounts)),
              surpriseHistory: wrapTensor(tf.tensor1d(memoryState.surpriseHistory))
            }));
          } else {
            console.log('No saved memory state found, initializing new state');
            this.memoryState = this.initializeEmptyState();
            await this.saveMemoryState();
          }
        } else {
          console.log('No saved model found, initializing new model and state');
          await this.model.saveModel(this.modelPath);
          this.memoryState = this.initializeEmptyState();
          await this.saveMemoryState();
        }
      } catch (loadError) {
        console.error('Error loading saved state:', loadError);
        console.log('Initializing new model and state');
        this.memoryState = this.initializeEmptyState();
        await this.model.saveModel(this.modelPath);
        await this.saveMemoryState();
      }

      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          try {
            await this.saveMemoryState();
          } catch (error) {
            console.error('Failed to auto-save memory state:', error);
          }
        }, 300000); // Save every 5 minutes
      }

      this.isInitialized = true;
    } catch (error: unknown) {
      console.error('Initialization failed:', error instanceof Error ? error.message : error);
      throw error;
    }
  }

  private async loadSavedState(): Promise<boolean> {
    try {
      const [modelExists, weightsExist] = await Promise.all([
        fs.access(this.modelPath).then(() => true).catch(() => false),
        fs.access(this.weightsPath).then(() => true).catch(() => false)
      ]);

      if (!modelExists || !weightsExist) return false;

      await this.model.loadModel(this.modelPath);
      const memoryStateJson = await fs.readFile(
        path.join(this.memoryPath, 'memory_state.json'),
        'utf8'
      );
      const memoryState = JSON.parse(memoryStateJson) as SerializedMemoryState;

      this.memoryState = this.wrapWithMemoryManagement(() => ({
        shortTerm: wrapTensor(tf.tensor(memoryState.shortTerm)),
        longTerm: wrapTensor(tf.tensor(memoryState.longTerm)),
        meta: wrapTensor(tf.tensor(memoryState.meta)),
        timestamps: wrapTensor(tf.tensor(memoryState.timestamps)),
        accessCounts: wrapTensor(tf.tensor(memoryState.accessCounts)),
        surpriseHistory: wrapTensor(tf.tensor(memoryState.surpriseHistory))
      }));

      return true;
    } catch (error: unknown) {
      console.error('Failed to load saved state:', error instanceof Error ? error.message : error);
      return false;
    }
  }

  private async saveMemoryState(): Promise<void> {
    // Start a new scope for memory management
    tf.engine().startScope();

    try {
      // Save model first
      await this.model.saveModel(this.modelPath);

      // Clone and get memory state data within a tidy block
      const memoryState = tf.tidy(() => {
        // Validate tensors before accessing
        if (!this.validateMemoryState(this.memoryState)) {
          throw new Error('Invalid memory state during save');
        }

        // Clone tensors to prevent disposal issues
        return {
          shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).clone().dataSync()),
          longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).clone().dataSync()),
          meta: Array.from(unwrapTensor(this.memoryState.meta).clone().dataSync()),
          timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).clone().dataSync()),
          accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).clone().dataSync()),
          surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).clone().dataSync())
        };
      });

      // Create encrypted state with proper tensor lifecycle management
      const encryptedState = tf.tidy(() => {
        const tensors = [
          tf.tensor(memoryState.shortTerm),
          tf.tensor(memoryState.longTerm),
          tf.tensor(memoryState.meta),
          tf.tensor(memoryState.timestamps),
          tf.tensor(memoryState.accessCounts),
          tf.tensor(memoryState.surpriseHistory)
        ];

        const encryptedBuffers = tensors.map(tensor => {
          const encrypted = this.encryptTensor(tensor);
          return Buffer.from(encrypted);
        });

        return Buffer.concat(encryptedBuffers);
      });

      await fs.writeFile(this.weightsPath, encryptedState);
    } catch (error) {
      console.error('Failed to save memory state:', error);
      throw error;
    } finally {
      tf.engine().endScope();
    }
  }

  public async run(): Promise<void> {
    try {
      // Initialize TensorFlow.js backend first
      await tf.ready();
      await tf.setBackend('tensorflow');

      // Verify backend is ready
      const backend = tf.getBackend();
      if (!backend) {
        throw new Error('Failed to initialize TensorFlow.js backend');
      }

      console.log('TensorFlow backend initialized:', backend);

      // Initialize server
      await this.autoInitialize();
      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      transport.send({
        jsonrpc: "2.0",
        method: "log",
        params: { message: "Titan Memory Server running on stdio" }
      });
    } catch (error: unknown) {
      const message = `Fatal error: ${error instanceof Error ? error.message : error}`;
      console.error(JSON.stringify({
        jsonrpc: "2.0",
        method: "error",
        params: { message }
      }));
      process.exit(1);
    }
  }
}

// CLI entry with proper error handling
if (import.meta.url === `file://${process.argv[1]}`) {
  new TitanMemoryServer().run().catch(error => {
    console.error(JSON.stringify({
      jsonrpc: "2.0",
      method: "error",
      params: {
        message: `Boot failed: ${error instanceof Error ? error.message : error}`
      }
    }));
    process.exit(1);
  });
}

// Define parameter schemas
const HelpParams = z.object({
  tool: z.string().optional(),
  category: z.string().optional(),
  showExamples: z.boolean().optional(),
  verbose: z.boolean().optional(),
  interactive: z.boolean().optional(),
  context: z.record(z.any()).optional()
});

const InitModelParams = z.object({
  inputDim: z.number().int().positive().optional(),
  memorySlots: z.number().int().positive().optional(),
  transformerLayers: z.number().int().positive().optional()
});