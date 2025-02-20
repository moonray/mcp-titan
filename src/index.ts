#!/usr/bin/env node
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { TitanMemoryModel } from './model.js';
import {
  IMemoryState,
  wrapTensor,
  unwrapTensor,
  ServerCapabilities,
  CallToolRequest,
  CallToolResult
} from './types.js';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs/promises';

export class TitanMemoryServer {
  private model!: TitanMemoryModel;
  private server: McpServer;
  private memoryState!: IMemoryState;
  private memoryPath: string;
  private modelPath: string;
  private weightsPath: string;
  private autoSaveInterval: NodeJS.Timeout | null = null;
  private isInitialized: boolean = false;

  constructor(config: { memoryPath?: string; modelPath?: string; weightsPath?: string } = {}) {
    this.memoryPath = config.memoryPath || path.join(
      os.platform() === 'win32' ? process.env.APPDATA || os.homedir() : os.homedir(),
      '.mcp-titan'
    );
    this.modelPath = config.modelPath || path.join(this.memoryPath, 'model.json');
    this.weightsPath = config.weightsPath || path.join(this.memoryPath, 'weights');

    // Initialize MCP server with proper name and version
    this.server = new McpServer({
      name: "Titan Memory",
      version: "1.0.0"
    });

    this.registerTools();
  }

  private async ensureInitialized() {
    if (!this.isInitialized) {
      await this.autoInitialize();
      this.isInitialized = true;
    }
  }

  private registerTools() {
    // Register the init_model tool
    this.server.tool(
      'init_model',
      {
        inputDim: z.number().optional(),
        hiddenDim: z.number().optional(),
        memoryDim: z.number().optional(),
        transformerLayers: z.number().optional(),
        numHeads: z.number().optional(),
        ffDimension: z.number().optional(),
        dropoutRate: z.number().optional(),
        maxSequenceLength: z.number().optional(),
        memorySlots: z.number().optional(),
        similarityThreshold: z.number().optional(),
        surpriseDecay: z.number().optional(),
        pruningInterval: z.number().optional(),
        gradientClip: z.number().optional()
      },
      async (params) => {
        this.model = new TitanMemoryModel(params);
        const zeros = tf.zeros([params.inputDim || 768]);
        const slots = this.model.getConfig().memorySlots;
        this.memoryState = {
          shortTerm: wrapTensor(zeros),
          longTerm: wrapTensor(zeros.clone()),
          meta: wrapTensor(zeros.clone()),
          timestamps: wrapTensor(tf.zeros([slots])),
          accessCounts: wrapTensor(tf.zeros([slots])),
          surpriseHistory: wrapTensor(tf.zeros([slots]))
        };
        zeros.dispose();
        this.isInitialized = true;

        return {
          content: [{
            type: "text",
            text: `Model initialized with params: ${JSON.stringify(params)}`
          }]
        };
      }
    );

    // Register train_step tool
    this.server.tool(
      'train_step',
      {
        x_t: z.array(z.number()),
        x_next: z.array(z.number())
      },
      async ({ x_t, x_next }) => {
        await this.ensureInitialized();

        const x_tT = wrapTensor(tf.tensor1d(x_t));
        const x_nextT = wrapTensor(tf.tensor1d(x_next));
        const { loss, gradients } = this.model.trainStep(x_tT, x_nextT, this.memoryState);
        const { predicted, memoryUpdate } = this.model.forward(x_tT, this.memoryState);

        this.memoryState = memoryUpdate.newState;

        const result = {
          loss: unwrapTensor(loss).dataSync()[0],
          predicted: Array.from(unwrapTensor(predicted).dataSync()),
          surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
        };

        [x_tT, x_nextT, predicted].forEach(t => t.dispose());

        return {
          content: [{
            type: "text",
            text: JSON.stringify(result)
          }]
        };
      }
    );

    // Register forward_pass tool
    this.server.tool(
      'forward_pass',
      {
        x: z.array(z.number())
      },
      async ({ x }) => {
        await this.ensureInitialized();

        const xT = wrapTensor(tf.tensor1d(x));
        const { predicted, memoryUpdate } = this.model.forward(xT, this.memoryState);

        this.memoryState = memoryUpdate.newState;

        const result = {
          predicted: Array.from(unwrapTensor(predicted).dataSync()),
          memory: Array.from(unwrapTensor(memoryUpdate.newState.shortTerm).dataSync()),
          surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
        };

        xT.dispose();
        predicted.dispose();

        return {
          content: [{
            type: "text",
            text: JSON.stringify(result)
          }]
        };
      }
    );

    // Register get_memory_state tool
    this.server.tool(
      'get_memory_state',
      {
        type: z.string().optional()
      },
      async (_args, _extra) => {
        await this.ensureInitialized();

        return tf.tidy(() => {
          const snapshot = this.model.getMemorySnapshot();
          const stats = {
            shortTermMean: tf.mean(unwrapTensor(snapshot.shortTerm)).dataSync()[0],
            shortTermStd: tf.sqrt(tf.moments(unwrapTensor(snapshot.shortTerm)).variance).dataSync()[0],
            longTermMean: tf.mean(unwrapTensor(snapshot.longTerm)).dataSync()[0],
            longTermStd: tf.sqrt(tf.moments(unwrapTensor(snapshot.longTerm)).variance).dataSync()[0]
          };

          return {
            content: [{
              type: "text" as const,
              text: JSON.stringify(stats)
            }]
          };
        });
      }
    );
  }

  private async autoInitialize(): Promise<void> {
    try {
      await fs.mkdir(this.memoryPath, { recursive: true });

      if (!(await this.loadSavedState())) {
        this.model = new TitanMemoryModel();
        const zeros = tf.zeros([768]);
        const slots = this.model.getConfig().memorySlots;
        this.memoryState = {
          shortTerm: wrapTensor(zeros),
          longTerm: wrapTensor(zeros.clone()),
          meta: wrapTensor(zeros.clone()),
          timestamps: wrapTensor(tf.zeros([slots])),
          accessCounts: wrapTensor(tf.zeros([slots])),
          surpriseHistory: wrapTensor(tf.zeros([slots]))
        };
        zeros.dispose();
        await this.saveMemoryState();
      }

      if (!this.autoSaveInterval) {
        this.autoSaveInterval = setInterval(async () => {
          await this.saveMemoryState().catch(console.error);
        }, 300000); // 5 minutes
      }
    } catch (error) {
      console.error('Initialization failed:', error);
      throw error;
    }
  }

  private async loadSavedState(): Promise<boolean> {
    try {
      const modelExists = await fs.access(this.modelPath)
        .then(() => true)
        .catch(() => false);
      const weightsExist = await fs.access(this.weightsPath)
        .then(() => true)
        .catch(() => false);

      if (!modelExists || !weightsExist) {
        return false;
      }

      await this.model.loadModel(this.modelPath);
      const memoryStateJson = await fs.readFile(
        path.join(this.memoryPath, 'memory_state.json'),
        'utf8'
      );
      const memoryState = JSON.parse(memoryStateJson);
      this.memoryState = {
        shortTerm: wrapTensor(tf.tensor(memoryState.shortTerm)),
        longTerm: wrapTensor(tf.tensor(memoryState.longTerm)),
        meta: wrapTensor(tf.tensor(memoryState.meta)),
        timestamps: wrapTensor(tf.tensor(memoryState.timestamps)),
        accessCounts: wrapTensor(tf.tensor(memoryState.accessCounts)),
        surpriseHistory: wrapTensor(tf.tensor(memoryState.surpriseHistory))
      };

      return true;
    } catch (error) {
      console.error('Failed to load saved state:', error);
      return false;
    }
  }

  private async saveMemoryState(): Promise<void> {
    try {
      await this.model.saveModel(this.modelPath);
      const memoryState = {
        shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
        longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
        meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()),
        timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
        accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()),
        surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync())
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

  public async run(): Promise<void> {
    try {
      await this.autoInitialize();

      const transport = new StdioServerTransport();
      await this.server.connect(transport);

      transport.send({
        jsonrpc: "2.0",
        method: "log",
        params: {
          message: "Titan Memory Server running on stdio"
        }
      });
    } catch (error) {
      const errorMessage = {
        jsonrpc: "2.0",
        method: "error",
        params: {
          message: `Fatal error running server: ${error instanceof Error ? error.message : String(error)}`
        }
      };
      console.error(JSON.stringify(errorMessage));
      process.exit(1);
    }
  }
}

// Command line entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const server = new TitanMemoryServer();
  server.run().catch((error) => {
    const errorMessage = {
      jsonrpc: "2.0",
      method: "error",
      params: {
        message: `Fatal error running server: ${error instanceof Error ? error.message : String(error)}`
      }
    };
    console.error(JSON.stringify(errorMessage));
    process.exit(1);
  });
}