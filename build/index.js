#!/usr/bin/env node
import '@tensorflow/tfjs-node'; // Import and register the Node.js backend
import * as tf from '@tensorflow/tfjs-node';
import express from 'express';
import bodyParser from 'body-parser';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs/promises';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { TitanMemoryModel } from './model.js';
import { wrapTensor, unwrapTensor } from './types.js';
export class TitanMemoryServer {
    constructor(config = {}) {
        this.autoSaveInterval = null;
        this.port = config.port || 0;
        this.memoryPath = config.memoryPath || path.join(os.platform() === 'win32' ? process.env.APPDATA || os.homedir() : os.homedir(), '.mcp-titan');
        this.modelPath = config.modelPath || path.join(this.memoryPath, 'model.json');
        this.weightsPath = config.weightsPath || path.join(this.memoryPath, 'weights');
        this.app = express();
        this.app.use(bodyParser.json());
        // Initialize MCP server with proper name and version
        this.server = new McpServer({
            name: "Titan Memory",
            version: "1.0.0"
        });
        this.registerTools();
    }
    registerTools() {
        // Register the init_model tool
        this.server.tool('init_model', {
            inputDim: z.number().optional(),
            outputDim: z.number().optional()
        }, async ({ inputDim = 768, outputDim = 768 }) => {
            this.model = new TitanMemoryModel({ inputDim, outputDim });
            const zeros = tf.zeros([outputDim]);
            this.memoryState = {
                shortTerm: wrapTensor(zeros),
                longTerm: wrapTensor(zeros.clone()),
                meta: wrapTensor(zeros.clone())
            };
            zeros.dispose();
            return {
                content: [{
                        type: "text",
                        text: `Model initialized with inputDim=${inputDim}, outputDim=${outputDim}`
                    }]
            };
        });
        // Register train_step tool
        this.server.tool('train_step', {
            x_t: z.array(z.number()),
            x_next: z.array(z.number())
        }, async ({ x_t, x_next }) => {
            if (!this.model || !this.memoryState) {
                throw new Error('Model not initialized');
            }
            const x_tT = wrapTensor(tf.tensor1d(x_t));
            const x_nextT = wrapTensor(tf.tensor1d(x_next));
            const cost = this.model.trainStep(x_tT, x_nextT, this.memoryState);
            const { predicted, memoryUpdate } = this.model.forward(x_tT, this.memoryState);
            this.memoryState = memoryUpdate.newState;
            const result = {
                cost: unwrapTensor(cost.loss).dataSync()[0],
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
        });
    }
    async run() {
        try {
            await this.autoInitialize();
            const transport = new StdioServerTransport();
            await this.server.connect(transport);
            // Use proper MCP message format for logging
            transport.send({
                jsonrpc: "2.0",
                method: "log",
                params: {
                    message: "Titan Memory Server running on stdio"
                }
            });
        }
        catch (error) {
            // Format error as proper MCP message
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
    async autoInitialize() {
        try {
            await fs.mkdir(this.memoryPath, { recursive: true });
            const initializeState = () => {
                const shortTerm = tf.zeros([768]);
                return {
                    shortTerm: wrapTensor(shortTerm),
                    longTerm: wrapTensor(tf.zerosLike(shortTerm)),
                    meta: wrapTensor(tf.zerosLike(shortTerm))
                };
            };
            if (!(await this.loadSavedState())) {
                this.model = new TitanMemoryModel({ inputDim: 768, outputDim: 768 });
                this.memoryState = initializeState();
                await this.saveMemoryState();
            }
            if (!this.autoSaveInterval) {
                this.autoSaveInterval = setInterval(async () => {
                    await this.saveMemoryState().catch(console.error);
                }, 300000); // 5 minutes
            }
        }
        catch (error) {
            const initializeState = () => {
                const shortTerm = tf.zeros([768]);
                return {
                    shortTerm: wrapTensor(shortTerm),
                    longTerm: wrapTensor(tf.zerosLike(shortTerm)),
                    meta: wrapTensor(tf.zerosLike(shortTerm))
                };
            };
            console.error('Initialization failed, using default model:', error);
            this.model = new TitanMemoryModel({ inputDim: 768, outputDim: 768 });
            this.memoryState = initializeState();
            await this.saveMemoryState();
            console.error('Failed to initialize model:', error);
            throw error;
        }
    }
    async loadSavedState() {
        try {
            // Check if model and weights files exist
            const modelExists = await fs.access(this.modelPath)
                .then(() => true)
                .catch(() => false);
            const weightsExist = await fs.access(this.weightsPath)
                .then(() => true)
                .catch(() => false);
            if (!modelExists || !weightsExist) {
                return false;
            }
            // Load model and weights
            this.model = new TitanMemoryModel();
            await this.model.save(this.modelPath, this.weightsPath);
            // Load memory state
            const memoryStateJson = await fs.readFile(path.join(this.memoryPath, 'memory_state.json'), 'utf8');
            const memoryState = JSON.parse(memoryStateJson);
            this.memoryState = {
                shortTerm: wrapTensor(tf.tensor(memoryState.shortTerm)),
                longTerm: wrapTensor(tf.tensor(memoryState.longTerm)),
                meta: wrapTensor(tf.tensor(memoryState.meta))
            };
            return true;
        }
        catch (error) {
            console.error('Failed to load saved state:', error);
            return false;
        }
    }
    async saveMemoryState() {
        try {
            // Use tf.keep() to prevent premature disposal
            const shortTerm = tf.keep(tf.clone(unwrapTensor(this.memoryState.shortTerm)));
            const longTerm = tf.keep(tf.clone(unwrapTensor(this.memoryState.longTerm)));
            const meta = tf.keep(tf.clone(unwrapTensor(this.memoryState.meta)));
            const memoryState = {
                shortTerm: await shortTerm.array(),
                longTerm: await longTerm.array(),
                meta: await meta.array()
            };
            await fs.writeFile(path.join(this.memoryPath, 'memory_state.json'), JSON.stringify(memoryState, null, 2));
            // Dispose only after serialization completes
            tf.tidy(() => {
                [shortTerm, longTerm, meta].forEach(t => t.dispose());
            });
        }
        catch (error) {
            console.error('Failed to save memory state:', error);
            throw error;
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
//# sourceMappingURL=index.js.map