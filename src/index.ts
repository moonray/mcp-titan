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

/**
 * Represents a serialized memory state that can be stored and loaded.
 */
interface SerializedMemoryState {
  shortTerm: number[];
  longTerm: number[];
  meta: number[];
  timestamps: number[];
  accessCounts: number[];
  surpriseHistory: number[];
}

/**
 * Statistics about the memory state.
 */
interface MemoryStats {
  shortTermMean: number;
  shortTermStd: number;
  longTermMean: number;
  longTermStd: number;
  capacity: number;
  surpriseScore: number;
  patternDiversity: number;
}

/**
 * Standard response format for all MCP tools.
 */
interface ToolResponse {
  [key: string]: unknown;
  content: Array<{
    [key: string]: unknown;
    type: "text" | "error" | "data";
    text: string;
    data?: any;
  }>;
  _meta?: Record<string, unknown>;
  isError?: boolean;
}

/**
 * Titan Memory Server - A neural memory system that can learn and predict sequences
 * while maintaining state through a memory vector.
 */
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
      version: "1.2.0",
      description: "A neural memory system for LLMs that can learn and predict sequences while maintaining state"
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
        tool: z.string().optional().describe("Specific tool name to get help for"),
        category: z.string().optional().describe("Category of tools to explore"),
        showExamples: z.boolean().optional().describe("Include usage examples"),
        verbose: z.boolean().optional().describe("Include detailed descriptions")
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
              "- manifold_step: Update memory along a manifold direction\n" +
              "- prune_memory: Remove less relevant memories\n" +
              "- save_checkpoint: Save memory state to file\n" +
              "- load_checkpoint: Load memory state from file\n" +
              "- reset_gradients: Reset accumulated gradients"
          }]
        };
      }
    );

    // Init model tool
    this.server.tool(
      'init_model',
      {
        inputDim: z.number().int().positive().default(768).describe("Input dimension size"),
        hiddenDim: z.number().int().positive().default(512).describe("Hidden dimension size"),
        memoryDim: z.number().int().positive().default(1024).describe("Memory dimension size"),
        transformerLayers: z.number().int().positive().default(6).describe("Number of transformer layers"),
        numHeads: z.number().int().positive().default(8).describe("Number of attention heads"),
        ffDimension: z.number().int().positive().default(2048).describe("Feed-forward dimension"),
        dropoutRate: z.number().min(0).max(0.9).default(0.1).describe("Dropout rate"),
        maxSequenceLength: z.number().int().positive().default(512).describe("Maximum sequence length"),
        memorySlots: z.number().int().positive().default(5000).describe("Number of memory slots"),
        similarityThreshold: z.number().min(0).max(1).default(0.65).describe("Similarity threshold"),
        surpriseDecay: z.number().min(0).max(1).default(0.9).describe("Surprise decay rate"),
        pruningInterval: z.number().int().positive().default(1000).describe("Pruning interval"),
        gradientClip: z.number().positive().default(1.0).describe("Gradient clipping value")
      },
      async (params, extra) => {
        await this.ensureInitialized();
        const config = {
          inputDim: params.inputDim,
          hiddenDim: params.hiddenDim || 512,
          memoryDim: params.memoryDim || 1024,
          transformerLayers: params.transformerLayers,
          numHeads: params.numHeads || 8,
          ffDimension: params.ffDimension || 2048,
          dropoutRate: params.dropoutRate || 0.1,
          maxSequenceLength: params.maxSequenceLength || 512,
          memorySlots: params.memorySlots,
          similarityThreshold: params.similarityThreshold || 0.65,
          surpriseDecay: params.surpriseDecay || 0.9,
          pruningInterval: params.pruningInterval || 1000,
          gradientClip: params.gradientClip || 1.0
        };
        await this.model.initialize(config);
        return {
          content: [{
            type: "data",
            text: `Model initialized with configuration: ${JSON.stringify(config)}`,
            data: config
          }]
        };
      }
    );

    // Forward pass tool
    this.server.tool(
      'forward_pass',
      {
        x: z.union([
          z.array(z.number()),
          z.string()
        ]).describe("Input vector or text"),
        memoryState: z.object({
          shortTerm: z.array(z.number()).optional(),
          longTerm: z.array(z.number()).optional(),
          meta: z.array(z.number()).optional(),
          timestamps: z.array(z.number()).optional(),
          accessCounts: z.array(z.number()).optional(),
          surpriseHistory: z.array(z.number()).optional()
        }).optional().describe("Memory state to use")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        const input = await this.processInput(params.x);

        // Use provided memory state or current state
        let memoryState = this.memoryState;
        if (params.memoryState) {
          // Convert arrays to tensors
          memoryState = this.wrapWithMemoryManagement(() => ({
            shortTerm: wrapTensor(tf.tensor(params.memoryState?.shortTerm || [])),
            longTerm: wrapTensor(tf.tensor(params.memoryState?.longTerm || [])),
            meta: wrapTensor(tf.tensor(params.memoryState?.meta || [])),
            timestamps: wrapTensor(tf.tensor(params.memoryState?.timestamps || [])),
            accessCounts: wrapTensor(tf.tensor(params.memoryState?.accessCounts || [])),
            surpriseHistory: wrapTensor(tf.tensor(params.memoryState?.surpriseHistory || []))
          }));
        }

        const result = await this.model.forward(wrapTensor(input), memoryState);

        // Update memory state
        this.memoryState = result.memoryUpdate.newState;

        return {
          content: [{
            type: "data",
            text: "Forward pass completed successfully",
            data: {
              predicted: Array.from(unwrapTensor(result.predicted).dataSync()),
              memoryUpdate: {
                surprise: {
                  immediate: Array.from(unwrapTensor(result.memoryUpdate.surprise.immediate).dataSync()),
                  accumulated: Array.from(unwrapTensor(result.memoryUpdate.surprise.accumulated).dataSync())
                }
              }
            }
          }]
        };
      }
    );

    // Train step tool
    this.server.tool(
      'train_step',
      {
        x_t: z.union([
          z.array(z.number()),
          z.string()
        ]).describe("Current input vector or text"),
        x_next: z.union([
          z.array(z.number()),
          z.string()
        ]).describe("Next input vector or text")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        const x_t = await this.processInput(params.x_t);
        const x_next = await this.processInput(params.x_next);

        const result = await this.model.trainStep(
          wrapTensor(x_t),
          wrapTensor(x_next),
          this.memoryState
        );

        return {
          content: [{
            type: "data",
            text: "Training step completed successfully",
            data: {
              loss: Array.from(unwrapTensor(result.loss).dataSync()),
              gradients: {
                shortTerm: Array.from(unwrapTensor(result.gradients.shortTerm).dataSync().slice(0, 10)) + "...",
                longTerm: Array.from(unwrapTensor(result.gradients.longTerm).dataSync().slice(0, 10)) + "...",
                meta: Array.from(unwrapTensor(result.gradients.meta).dataSync().slice(0, 10)) + "..."
              }
            }
          }]
        };
      }
    );

    // Get memory state tool
    this.server.tool(
      'get_memory_state',
      {
        type: z.string().optional().describe("Optional memory type filter")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        // Get memory snapshot
        const state = this.model.getMemorySnapshot();

        // Calculate memory statistics
        const stats = this.wrapWithMemoryManagement(() => {
          const shortTermMean = state.shortTerm.mean().dataSync()[0];
          const shortTermStd = tf.moments(state.shortTerm).variance.sqrt().dataSync()[0];
          const longTermMean = state.longTerm.mean().dataSync()[0];
          const longTermStd = tf.moments(state.longTerm).variance.sqrt().dataSync()[0];

          // Calculate surprise score (average of recent surprise history)
          const surpriseHistory = unwrapTensor(this.memoryState.surpriseHistory);
          const surpriseScore = surpriseHistory.size > 0
            ? surpriseHistory.mean().dataSync()[0]
            : 0;

          // Calculate pattern diversity (using standard deviation of meta memory)
          const meta = unwrapTensor(this.memoryState.meta);
          const patternDiversity = meta.size > 0
            ? tf.moments(meta).variance.sqrt().mean().dataSync()[0]
            : 0;

          // Calculate memory capacity
          const memorySlots = this.model.getConfig().memorySlots;
          const usedSlots = unwrapTensor(this.memoryState.timestamps).shape[0];
          const capacity = 1 - (usedSlots / memorySlots);

          return {
            shortTermMean,
            shortTermStd,
            longTermMean,
            longTermStd,
            surpriseScore,
            patternDiversity,
            capacity
          };
        });

        return {
          content: [{
            type: "data",
            text: "Memory state retrieved successfully",
            data: {
              stats,
              capacity: stats.capacity,
              status: stats.capacity > 0.3 ? "active" : "pruning",
              timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
              accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync())
            }
          }]
        };
      }
    );

    // Manifold step tool
    this.server.tool(
      'manifold_step',
      {
        base: z.array(z.number()).describe("Base memory state"),
        velocity: z.array(z.number()).describe("Update direction")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        const base = tf.tensor(params.base);
        const velocity = tf.tensor(params.velocity);

        const result = await this.model.manifoldStep(
          wrapTensor(base),
          wrapTensor(velocity)
        );

        return {
          content: [{
            type: "data",
            text: "Manifold step completed successfully",
            data: {
              newBase: Array.from(unwrapTensor(result).dataSync())
            }
          }]
        };
      }
    );

    // Prune memory tool
    this.server.tool(
      'prune_memory',
      {
        threshold: z.number().min(0).max(1).describe("Pruning threshold (0-1)")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        this.memoryState = this.model.pruneMemory(this.memoryState, params.threshold);

        // Get updated stats after pruning
        const stats = this.wrapWithMemoryManagement(() => {
          const memorySlots = this.model.getConfig().memorySlots;
          const usedSlots = unwrapTensor(this.memoryState.timestamps).shape[0];
          const capacity = 1 - (usedSlots / memorySlots);
          return { capacity };
        });

        return {
          content: [{
            type: "data",
            text: "Memory pruned successfully",
            data: {
              newCapacity: stats.capacity,
              remainingEntries: unwrapTensor(this.memoryState.timestamps).shape[0]
            }
          }]
        };
      }
    );

    // Save checkpoint tool
    this.server.tool(
      'save_checkpoint',
      {
        path: z.string().describe("Checkpoint file path")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        const checkpointPath = params.path;
        await this.saveMemoryState(checkpointPath);

        return {
          content: [{
            type: "text",
            text: `Memory state saved to ${checkpointPath}`
          }]
        };
      }
    );

    // Load checkpoint tool
    this.server.tool(
      'load_checkpoint',
      {
        path: z.string().describe("Checkpoint file path")
      },
      async (params, extra) => {
        await this.ensureInitialized();

        const checkpointPath = params.path;
        const success = await this.loadSavedState(checkpointPath);

        if (!success) {
          return {
            content: [{
              type: "error",
              text: `Failed to load memory state from ${checkpointPath}`
            }],
            isError: true
          };
        }

        return {
          content: [{
            type: "text",
            text: `Memory state loaded from ${checkpointPath}`
          }]
        };
      }
    );

    // Reset gradients tool
    this.server.tool(
      'reset_gradients',
      {},
      async (params, extra) => {
        await this.ensureInitialized();

        // Reset optimizer state
        this.model.resetGradients();

        return {
          content: [{
            type: "text",
            text: "Gradients reset successfully"
          }]
        };
      }
    );
  }

  private async processInput(input: string | number[] | tf.Tensor | number): Promise<tf.Tensor> {
    return this.wrapWithMemoryManagement(() => {
      try {
        // Handle string input
        if (typeof input === 'string') {
          // Use vectorProcessor to encode text
          return this.vectorProcessor.encodeText(input);
        }

        // Handle number input (single value)
        if (typeof input === 'number') {
          return tf.tensor2d([[input]]);
        }

        // Handle array input
        if (Array.isArray(input)) {
          // Check if it's a 1D or 2D array
          if (input.length > 0 && Array.isArray(input[0])) {
            // It's already 2D
            return tf.tensor2d(input as number[][]);
          } else {
            // It's 1D, convert to 2D
            return tf.tensor2d([input]);
          }
        }

        // Handle tensor input
        if (input instanceof tf.Tensor) {
          // Ensure it's 2D
          if (input.rank === 1) {
            return input.expandDims(0);
          } else if (input.rank === 2) {
            return input;
          } else {
            throw new Error(`Unsupported tensor rank: ${input.rank}. Expected 1 or 2.`);
          }
        }

        throw new Error(`Unsupported input type: ${typeof input}`);
      } catch (error) {
        console.error('Error processing input:', error);
        // Return a default tensor in case of error
        return tf.zeros([1, this.model.getConfig().inputDim || 768]);
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

  /**
   * Run the Titan Memory Server
   * This method initializes the server and connects to the transport
   */
  public async run(): Promise<void> {
    try {
      console.log('Starting Titan Memory Server...');

      // Ensure the model is initialized
      await this.ensureInitialized();

      // Set up auto-save interval
      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          try {
            await this.saveMemoryState();
            console.log('Auto-saved memory state');
          } catch (error) {
            console.error('Error during auto-save:', error);
          }
        }, 5 * 60 * 1000); // Auto-save every 5 minutes
      }

      // Connect to the transport
      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      console.log(`Titan Memory Server v${this.server.version} running`);
      console.log('Available tools:');

      // List available tools
      const tools = Object.keys(this.server.tools || {});
      tools.forEach(tool => {
        console.log(`- ${tool}`);
      });

      // Handle process termination
      process.on('SIGINT', async () => {
        console.log('Shutting down Titan Memory Server...');

        // Save memory state before exit
        try {
          await this.saveMemoryState();
          console.log('Memory state saved');
        } catch (error) {
          console.error('Error saving memory state:', error);
        }

        // Clean up resources
        if (this.autoSaveInterval) {
          clearInterval(this.autoSaveInterval);
        }

        // Dispose tensors
        this.model.dispose();

        process.exit(0);
      });
    } catch (error) {
      console.error('Error starting Titan Memory Server:', error);
      throw error;
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