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
import { MemoryManager, VectorProcessor, AutomaticMemoryMaintenance } from './utils.js';
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
  private memoryManager: MemoryManager;
  private vectorProcessor: VectorProcessor;
  private maintenance: AutomaticMemoryMaintenance;

  constructor(config: { memoryPath?: string; modelPath?: string; weightsPath?: string } = {}) {
    this.memoryPath = config.memoryPath || path.join(
      os.platform() === 'win32' ? process.env.APPDATA || os.homedir() : os.homedir(),
      '.mcp-titan'
    );
    this.modelPath = config.modelPath || path.join(this.memoryPath, 'model.json');
    this.weightsPath = config.weightsPath || path.join(this.memoryPath, 'weights');

    // Initialize utilities
    this.memoryManager = MemoryManager.getInstance();
    this.vectorProcessor = VectorProcessor.getInstance();
    this.maintenance = AutomaticMemoryMaintenance.getInstance();

    // Initialize MCP server with proper name and version
    this.server = new McpServer({
      name: "Titan Memory",
      version: "1.2.0"
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
    // Register the help tool
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
      async (params) => {
        await this.ensureInitialized();

        interface ToolDefinition {
          name: string;
          description: string;
          parameters: Record<string, {
            type: string;
            description: string;
            required: boolean;
            default?: number;
          }>;
          examples: string[];
        }

        interface ToolRegistry {
          [key: string]: ToolDefinition;
        }

        interface CategoryRegistry {
          [key: string]: string[];
        }

        const allTools: ToolRegistry = {
          'init_model': {
            name: 'init_model',
            description: 'Initialize the Titan Memory model for learning code patterns. If already initialized, returns early with a message.',
            parameters: {
              inputDim: { type: 'number', description: 'Size of input vectors', required: false, default: 768 },
              memorySlots: { type: 'number', description: 'Number of memory slots', required: false, default: 5000 },
              transformerLayers: { type: 'number', description: 'Number of transformer layers', required: false, default: 6 }
            },
            examples: [
              'await callTool("init_model", {})',
              'await callTool("init_model", { inputDim: 768, memorySlots: 10000, transformerLayers: 4 })'
            ]
          },
          'train_step': {
            name: 'train_step',
            description: 'Train the model on sequential inputs',
            parameters: {
              x_t: { type: 'array|string', description: 'Current input', required: true },
              x_next: { type: 'array|string', description: 'Next input', required: true }
            },
            examples: ['await callTool("train_step", { x_t: "function hello() {", x_next: "console.log(\'world\');" })']
          },
          'forward_pass': {
            name: 'forward_pass',
            description: 'Process input through the model',
            parameters: {
              x: { type: 'array|string', description: 'Input to process', required: true }
            },
            examples: ['await callTool("forward_pass", { x: "const x = 5;" })']
          },
          'get_memory_state': {
            name: 'get_memory_state',
            description: 'Get current memory statistics and state',
            parameters: {
              type: { type: 'string', description: 'Optional memory type filter', required: false }
            },
            examples: ['await callTool("get_memory_state", {})']
          }
        };

        if (params.tool && params.tool in allTools) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify(allTools[params.tool], null, 2)
            }]
          };
        }

        const categories: CategoryRegistry = {
          'memory': ['init_model', 'forward_pass', 'train_step'],
          'maintenance': ['get_memory_state'],
        };

        if (params.category && params.category in categories) {
          const toolsInCategory = categories[params.category];
          return {
            content: [{
              type: "text",
              text: JSON.stringify(
                toolsInCategory.map(t => allTools[t]),
                null,
                2
              )
            }]
          };
        }

        // Return all tools if no specific request
        return {
          content: [{
            type: "text",
            text: JSON.stringify(allTools, null, 2)
          }]
        };
      }
    );

    // Register the init_model tool
    this.server.tool(
      'init_model',
      {
        inputDim: z.number().optional(),
        memorySlots: z.number().optional(),
        transformerLayers: z.number().optional()
      },
      async (params) => {
        // If already initialized, return early
        if (this.isInitialized) {
          return {
            content: [{
              type: "text",
              text: "Model already initialized"
            }]
          };
        }

        return this.memoryManager.wrapWithMemoryManagementAsync(async () => {
          this.model = new TitanMemoryModel(params);
          const config = this.model.getConfig();
          const zeros = tf.zeros([config.inputDim]);
          const slots = config.memorySlots;

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
              text: `Model initialized with configuration: ${JSON.stringify(config)}`
            }]
          };
        });
      }
    );

    // Register train_step tool
    this.server.tool(
      'train_step',
      {
        x_t: z.array(z.number()).or(z.string()),
        x_next: z.array(z.number()).or(z.string())
      },
      async ({ x_t, x_next }) => {
        await this.ensureInitialized();

        return this.memoryManager.wrapWithMemoryManagementAsync(async () => {
          // Process inputs through vectorProcessor
          let x_tT = await this.vectorProcessor.encodeText(x_t.toString());
          let x_nextT = await this.vectorProcessor.encodeText(x_next.toString());

          // Reshape to add batch dimension [1, features]
          const x_tReshaped = unwrapTensor(x_tT).reshape([1, -1]);
          const x_nextReshaped = unwrapTensor(x_nextT).reshape([1, -1]);

          const { loss, gradients } = this.model.trainStep(
            wrapTensor(x_tReshaped),
            wrapTensor(x_nextReshaped),
            this.memoryState
          );
          const { predicted, memoryUpdate } = this.model.forward(wrapTensor(x_tReshaped), this.memoryState);

          this.memoryState = memoryUpdate.newState;

          const result = {
            loss: unwrapTensor(loss).dataSync()[0],
            predicted: Array.from(unwrapTensor(predicted).dataSync()),
            surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
          };

          return {
            content: [{
              type: "text",
              text: JSON.stringify(result)
            }]
          };
        });
      }
    );

    // Register forward_pass tool
    this.server.tool(
      'forward_pass',
      {
        x: z.array(z.number()).or(z.string())
      },
      async ({ x }) => {
        await this.ensureInitialized();

        return this.memoryManager.wrapWithMemoryManagementAsync(async () => {
          // Process input through vectorProcessor
          let xT = await this.vectorProcessor.encodeText(x.toString());

          // Reshape to add batch dimension [1, features]
          const xReshaped = unwrapTensor(xT).reshape([1, -1]);

          // Ensure tensor is properly wrapped before passing to model
          const { predicted, memoryUpdate } = this.model.forward(
            wrapTensor(xReshaped),
            this.memoryState
          );

          this.memoryState = memoryUpdate.newState;

          const result = {
            predicted: Array.from(unwrapTensor(predicted).dataSync()),
            memory: Array.from(unwrapTensor(memoryUpdate.newState.shortTerm).dataSync()),
            surprise: unwrapTensor(memoryUpdate.surprise.immediate).dataSync()[0]
          };

          return {
            content: [{
              type: "text",
              text: JSON.stringify(result)
            }]
          };
        });
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

        return this.memoryManager.wrapWithMemoryManagement(() => {
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
        this.memoryState = this.memoryManager.wrapWithMemoryManagement(() => {
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

      this.memoryState = this.memoryManager.wrapWithMemoryManagement(() => ({
        shortTerm: wrapTensor(tf.tensor(memoryState.shortTerm)),
        longTerm: wrapTensor(tf.tensor(memoryState.longTerm)),
        meta: wrapTensor(tf.tensor(memoryState.meta)),
        timestamps: wrapTensor(tf.tensor(memoryState.timestamps)),
        accessCounts: wrapTensor(tf.tensor(memoryState.accessCounts)),
        surpriseHistory: wrapTensor(tf.tensor(memoryState.surpriseHistory))
      }));

      return true;
    } catch (error) {
      console.error('Failed to load saved state:', error);
      return false;
    }
  }

  private async saveMemoryState(): Promise<void> {
    try {
      await this.model.saveModel(this.modelPath);
      const memoryState = this.memoryManager.wrapWithMemoryManagement(() => {
        const state = {
          shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
          longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
          meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()),
          timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
          accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()),
          surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync())
        };
        // Encrypt the state before saving
        const encryptedState = Buffer.concat([
          this.memoryManager.encryptTensor(tf.tensor(state.shortTerm)),
          this.memoryManager.encryptTensor(tf.tensor(state.longTerm)),
          this.memoryManager.encryptTensor(tf.tensor(state.meta)),
          this.memoryManager.encryptTensor(tf.tensor(state.timestamps)),
          this.memoryManager.encryptTensor(tf.tensor(state.accessCounts)),
          this.memoryManager.encryptTensor(tf.tensor(state.surpriseHistory))
        ]);
        return encryptedState;
      });

      await fs.writeFile(
        path.join(this.memoryPath, 'memory_state.json'),
        JSON.stringify({ encrypted: memoryState.toString('base64') }, null, 2)
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