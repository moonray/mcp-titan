#!/usr/bin/env node
import '@tensorflow/tfjs-node'; // Import and register the Node.js backend
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, CallToolResultSchema } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import express from 'express';
import bodyParser from 'body-parser';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor, unwrapTensor } from './types.js';
import path from 'path';
import os from 'os';
import fs from 'fs/promises';
export class TitanMemoryServer {
    constructor(port = 0) {
        this.model = null;
        this.memoryState = null;
        this.autoSaveInterval = null;
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
                }
            },
            get_memory_state: {
                name: 'get_memory_state',
                description: 'Get insights about what patterns the model has learned',
                parameters: {
                    type: 'object',
                    properties: {}
                }
            }
        };
        const capabilities = {
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
            }
            catch (error) {
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
    async setupAutomaticMemory() {
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
                            await fs.writeFile(path.join(this.memoryPath, 'memory.json'), JSON.stringify(memoryState), 'utf-8');
                        }
                    }
                    catch (error) {
                        console.error('Error saving memory state:', error);
                    }
                }, 5 * 60 * 1000); // Every 5 minutes
                // Prevent the interval from keeping the process alive
                this.autoSaveInterval.unref();
            }
        }
        catch (error) {
            console.error('Error setting up automatic memory:', error);
        }
    }
    async handleToolCall(request) {
        switch (request.params.name) {
            case 'init_model': {
                const { inputDim = 768, outputDim = 768 } = request.params.arguments;
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
                const { x_t, x_next } = request.params.arguments;
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
                const { x } = request.params.arguments;
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
    async run() {
        try {
            // Connect stdio for Cursor
            const stdioTransport = new StdioServerTransport();
            await this.server.connect(stdioTransport);
            // Start HTTP server with dynamic port
            await new Promise((resolve, reject) => {
                const server = this.app.listen(this.port, () => {
                    this.port = server.address().port;
                    console.log(`Titan Memory MCP server running on port ${this.port}`);
                    resolve();
                });
                server.on('error', (error) => {
                    if (error.code === 'EADDRINUSE') {
                        console.error(`Port ${this.port} is already in use, trying another port...`);
                        server.listen(0); // Let OS assign random port
                    }
                    else {
                        console.error('Server error:', error);
                        reject(error);
                    }
                });
                // Prevent the server from keeping the process alive
                server.unref();
            });
            // Set up cleanup handlers
            process.on('SIGINT', async () => {
                await this.cleanup();
                process.exit(0);
            });
            process.on('SIGTERM', async () => {
                await this.cleanup();
                process.exit(0);
            });
            // Handle uncaught errors
            process.on('uncaughtException', async (error) => {
                console.error('Uncaught exception:', error);
                await this.cleanup();
                process.exit(1);
            });
            process.on('unhandledRejection', async (error) => {
                console.error('Unhandled rejection:', error);
                await this.cleanup();
                process.exit(1);
            });
        }
        catch (error) {
            console.error('Error starting server:', error);
            throw error;
        }
    }
    async cleanup() {
        try {
            if (this.autoSaveInterval) {
                clearInterval(this.autoSaveInterval);
            }
            if (this.memoryState) {
                this.memoryState.shortTerm.dispose();
                this.memoryState.longTerm.dispose();
                this.memoryState.meta.dispose();
                this.memoryState = null;
            }
            if (this.model) {
                this.model = null;
            }
            // Close HTTP server if it's running
            if (this.app) {
                await new Promise((resolve) => {
                    const server = this.app.listen().close(() => resolve());
                    server.unref();
                });
            }
            // Close MCP server connection
            if (this.server) {
                await this.server.close();
            }
        }
        catch (error) {
            console.error('Error during cleanup:', error);
        }
    }
}
// Create and run server instance if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
    const server = new TitanMemoryServer();
    server.run().catch(console.error);
}
//# sourceMappingURL=index.js.map