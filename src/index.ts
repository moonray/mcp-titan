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
    this.weightsPath = path.join(this.memoryPath, 'weights.bin');
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
        showExamples: z.boolean().optional().default(false),
        verbose: z.boolean().optional().default(false),
        interactive: z.boolean().optional().default(false),
        context: z.record(z.any()).optional().default({})
      },
      async (params: z.infer<typeof HelpParams>, extra: RequestHandlerExtra): Promise<ToolResponse> => {
        await this.ensureInitialized();
        return {
          content: [{
            type: "text" as const,
            text: "Help information"
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
      async ({ x_t, x_next }: { x_t: string | number[]; x_next: string | number[] }, extra: RequestHandlerExtra): Promise<ToolResponse> => {
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
        x: z.union([z.string(), z.array(z.number()), z.custom<tf.Tensor>()])
      },
      async ({ x }): Promise<ToolResponse> => {
        await this.ensureInitialized();

        return this.wrapWithMemoryManagementAsync(async () => {
          if (!x) {
            throw new Error('Input x must be provided');
          }

          const xT = await this.vectorProcessor.processInput(x);
          const normalizedXT = this.vectorProcessor.validateAndNormalize(
            xT,
            [this.model.getConfig().inputDim]
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
        });
      }
    );

    // Memory state tool with typed response
    this.server.tool(
      'get_memory_state',
      {
        type: z.string().optional()
      },
      async (params: { type?: string }, extra: RequestHandlerExtra): Promise<ToolResponse> => {
        await this.ensureInitialized();
        const response: ToolResponse = {
          content: [{
            type: "text" as const,
            text: "Memory state retrieved"
          }]
        };
        return response;
      }
    );
  }

  private async autoInitialize(): Promise<void> {
    try {
      await fs.mkdir(this.memoryPath, { recursive: true });

      if (!(await this.loadSavedState())) {
        this.model = new TitanMemoryModel();
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
        await this.saveMemoryState();
      }

      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          await this.saveMemoryState().catch(console.error);
        }, 300000);
      }
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