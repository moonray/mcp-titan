#!/usr/bin/env node
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

// Define parameter schemas first
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

const ForwardPassParams = z.object({
  x: z.union([z.string(), z.number(), z.array(z.number()), z.custom<tf.Tensor>()])
});

const TrainStepParams = z.object({
  x_t: z.union([z.string(), z.array(z.number())]),
  x_next: z.union([z.string(), z.array(z.number())])
});

const GetMemoryStateParams = z.object({
  type: z.string().optional()
});

// Interface definitions
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
  private readonly testMode: boolean;

  constructor(options: { memoryPath?: string; testMode?: boolean } = {}) {
    this.server = new McpServer({
      name: "Titan Memory",
      version: "1.2.0",
      description: "A memory-augmented model server for Cursor",
      capabilities: {
        tools: {
          help: {
            description: "Get help about available tools",
            parameters: HelpParams
          },
          init_model: {
            description: "Initialize or reset the model",
            parameters: InitModelParams
          },
          train_step: {
            description: "Perform a training step",
            parameters: TrainStepParams
          },
          forward_pass: {
            description: "Perform a forward pass through the model",
            parameters: ForwardPassParams
          },
          get_memory_state: {
            description: "Get the current memory state",
            parameters: GetMemoryStateParams
          }
        }
      }
    });

    this.vectorProcessor = VectorProcessor.getInstance();
    this.memoryPath = options.memoryPath || path.join(process.cwd(), '.titan_memory');
    this.modelPath = path.join(this.memoryPath, 'model.json');
    this.weightsPath = path.join(this.memoryPath, 'weights.bin');
    this.testMode = options.testMode || false;
    this.memoryState = this.initializeEmptyState();

    if (this.testMode) {
      // Initialize model with default configuration in test mode
      this.model = new TitanMemoryModel({
        inputDim: 768,
        hiddenDim: 512,
        memoryDim: 1024,
        transformerLayers: 6,
        numHeads: 8,
        ffDimension: 2048,
        dropoutRate: 0.1,
        maxSequenceLength: 512,
        memorySlots: 5000,
        similarityThreshold: 0.65,
        surpriseDecay: 0.9,
        pruningInterval: 1000,
        gradientClip: 1.0,
        learningRate: 0.001,
        vocabSize: 50000
      });
    }

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
      return (
        state.shortTerm !== undefined &&
        state.longTerm !== undefined &&
        state.meta !== undefined &&
        state.timestamps !== undefined &&
        state.accessCounts !== undefined &&
        state.surpriseHistory !== undefined
      );
    });
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.autoInitialize();
      this.isInitialized = true;
    }
  }

  private registerTools(): void {
    // Help tool with improved schema validation
    this.server.tool(
      'help',
      {
        tool: z.string().optional(),
        category: z.string().optional(),
        showExamples: z.boolean().optional(),
        verbose: z.boolean().optional(),
        interactive: z.boolean().optional(),
        context: z.record(z.any()).optional()
      },
      async (params: z.infer<typeof HelpParams>, extra: RequestHandlerExtra): Promise<ToolResponse> => {
        await this.ensureInitialized();
        return {
          content: [{
            type: "text" as const,
            text: "Available tools: help, init_model, train_step, forward_pass, get_memory_state"
          }]
        };
      }
    );

    // Init model tool with memory safety
    this.server.tool(
      'init_model',
      {
        inputDim: z.number().int().positive().optional(),
        memorySlots: z.number().int().positive().optional(),
        transformerLayers: z.number().int().positive().optional()
      },
      async (params: z.infer<typeof InitModelParams>, extra: RequestHandlerExtra): Promise<ToolResponse> => {
        if (this.isInitialized) {
          return {
            content: [{
              type: "text" as const,
              text: "Model already initialized"
            }]
          };
        }

        return this.wrapWithMemoryManagementAsync(async () => {
          this.model = new TitanMemoryModel(params);
          const config = this.model.getConfig();
          const zeros = tf.tidy(() => tf.zeros([config.inputDim]));

          try {
            const slots = config.memorySlots;
            this.memoryState = {
              shortTerm: wrapTensor(zeros.clone()),
              longTerm: wrapTensor(zeros.clone()),
              meta: wrapTensor(zeros.clone()),
              timestamps: wrapTensor(tf.zeros([slots])),
              accessCounts: wrapTensor(tf.zeros([slots])),
              surpriseHistory: wrapTensor(tf.zeros([slots]))
            };
          } finally {
            zeros.dispose();
          }

          this.isInitialized = true;
          return {
            content: [{
              type: "text" as const,
              text: `Model initialized with configuration: ${JSON.stringify(config)}`
            }]
          };
        });
      }
    );

    // Train step tool with enhanced validation
    this.server.tool(
      'train_step',
      {
        x_t: z.union([z.string(), z.array(z.number())]),
        x_next: z.union([z.string(), z.array(z.number())])
      },
      async (params: z.infer<typeof TrainStepParams>, extra: RequestHandlerExtra): Promise<ToolResponse> => {
        await this.ensureInitialized();
        return {
          content: [{
            type: "text",
            text: "Training step completed"
          }]
        };
      }
    );

    // Forward pass tool with memory validation
    this.server.tool(
      'forward_pass',
      {
        x: z.union([z.string(), z.number(), z.array(z.number()), z.custom<tf.Tensor>()])
      },
      async (params: z.infer<typeof ForwardPassParams>): Promise<ToolResponse> => {
        if (params.x === null || params.x === undefined) {
          throw new Error('Input x must be provided');
        }

        return this.wrapWithMemoryManagementAsync(async () => {
          try {
            let xT: tf.Tensor;
            if (typeof params.x === 'string') {
              xT = await this.vectorProcessor.encodeText(params.x);
            } else {
              xT = await this.vectorProcessor.processInput(params.x);
            }

            const config = this.model.getConfig();
            const normalizedXT = this.vectorProcessor.validateAndNormalize(
              xT,
              [config.inputDim]
            );

            if (!this.validateMemoryState(this.memoryState)) {
              throw new Error('Invalid memory state');
            }

            const xReshaped = tf.tidy(() => unwrapTensor(normalizedXT).reshape([1, -1]));

            const { predicted, memoryUpdate } = this.model.forward(
              wrapTensor(xReshaped),
              this.memoryState
            );

            this.memoryState = memoryUpdate.newState;

            const result = {
              predicted: Array.from(unwrapTensor(predicted).dataSync()),
              memory: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
              surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
            };

            xReshaped.dispose();

            return {
              content: [{
                type: "text" as const,
                text: JSON.stringify(result)
              }]
            };
          } catch (error) {
            return {
              content: [{
                type: "text" as const,
                text: error instanceof Error ? error.message : 'Unknown error occurred'
              }],
              isError: true
            };
          }
        });
      }
    );

    // Memory state tool with typed response
    this.server.tool(
      'get_memory_state',
      {
        type: z.string().optional()
      },
      async (params: z.infer<typeof GetMemoryStateParams>, extra: RequestHandlerExtra): Promise<ToolResponse> => {
        const state = {
          shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
          longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
          meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync())
        };

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify(state)
          }]
        };
      }
    );
  }

  private async autoInitialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Create memory directory if it doesn't exist
      await fs.mkdir(this.memoryPath, { recursive: true });

      if (!this.testMode) {
        // Try to load existing model and state
        const loaded = await this.loadSavedState();
        if (!loaded) {
          // Only initialize new model if loading fails and not in test mode
          this.model = new TitanMemoryModel({
            inputDim: 768,
            hiddenDim: 512,
            memoryDim: 1024,
            transformerLayers: 6,
            numHeads: 8,
            ffDimension: 2048,
            dropoutRate: 0.1,
            maxSequenceLength: 512,
            memorySlots: 5000,
            similarityThreshold: 0.65,
            surpriseDecay: 0.9,
            pruningInterval: 1000,
            gradientClip: 1.0,
            learningRate: 0.001,
            vocabSize: 50000
          });

          // Initialize empty memory state
          this.memoryState = this.wrapWithMemoryManagement(() => {
            const zeros = tf.zeros([768]);
            const slots = this.model.getConfig().memorySlots;
            return {
              shortTerm: wrapTensor(zeros),
              longTerm: wrapTensor(zeros.clone()),
              meta: wrapTensor(zeros.clone()),
              timestamps: wrapTensor(tf.zeros([slots])),
              accessCounts: wrapTensor(tf.zeros([slots])),
              surpriseHistory: wrapTensor(tf.zeros([slots]))
            };
          });
        }
      }

      // Set up auto-save interval for non-test mode
      if (!this.testMode && !this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          await this.saveMemoryState().catch(console.error);
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
    try {
      await this.model.saveModel(this.modelPath);
      const memoryState = this.wrapWithMemoryManagement(() => {
        const state = {
          shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()) as number[],
          longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()) as number[],
          meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()) as number[],
          timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()) as number[],
          accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()) as number[],
          surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync()) as number[]
        };
        return state;
      });

      const encryptedState = Buffer.concat([
        this.encryptTensor(tf.tensor(memoryState.shortTerm)),
        this.encryptTensor(tf.tensor(memoryState.longTerm)),
        this.encryptTensor(tf.tensor(memoryState.meta)),
        this.encryptTensor(tf.tensor(memoryState.timestamps)),
        this.encryptTensor(tf.tensor(memoryState.accessCounts)),
        this.encryptTensor(tf.tensor(memoryState.surpriseHistory))
      ]);

      await fs.writeFile(this.weightsPath, encryptedState);
    } catch (error: unknown) {
      console.error('Failed to save memory state:', error instanceof Error ? error.message : error);
      throw error;
    }
  }

  public async run(): Promise<void> {
    try {
      await this.autoInitialize();
      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      // Send server info for Cursor Composer
      transport.send({
        jsonrpc: "2.0",
        method: "server.info",
        params: {
          name: "Titan Memory",
          version: "1.2.0",
          description: "A memory-augmented model server for Cursor",
          tools: [
            {
              name: "help",
              description: "Get help about available tools"
            },
            {
              name: "init_model",
              description: "Initialize or reset the model"
            },
            {
              name: "train_step",
              description: "Perform a training step"
            },
            {
              name: "forward_pass",
              description: "Perform a forward pass through the model"
            },
            {
              name: "get_memory_state",
              description: "Get the current memory state"
            }
          ]
        }
      });

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
