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
        this.memoryVec = null;
        this.autoSaveInterval = null;
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
            name: 'titan-memory',
            version: '0.1.0',
            description: 'Automatic memory-augmented learning for Cursor',
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
            if (!this.memoryVec) {
                this.memoryVec = tf.variable(tf.zeros([768]));
            }
            // Set up auto-save interval
            if (!this.autoSaveInterval) {
                this.autoSaveInterval = setInterval(async () => {
                    try {
                        if (this.memoryVec) {
                            const memoryState = {
                                vector: Array.from(this.memoryVec.dataSync()),
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
                this.memoryVec = tf.variable(tf.zeros([outputDim]));
                return { config: { inputDim, outputDim } };
            }
            case 'train_step': {
                if (!this.model || !this.memoryVec) {
                    throw new Error('Model not initialized');
                }
                const { x_t, x_next } = request.params.arguments;
                const x_tT = wrapTensor(tf.tensor1d(x_t));
                const x_nextT = wrapTensor(tf.tensor1d(x_next));
                const memoryT = wrapTensor(this.memoryVec);
                const cost = this.model.trainStep(x_tT, x_nextT, memoryT);
                const { predicted, newMemory, surprise } = this.model.forward(x_tT, memoryT);
                // Update memory vector with new state
                this.memoryVec.assign(unwrapTensor(newMemory));
                const result = {
                    cost: unwrapTensor(cost).dataSync()[0],
                    predicted: Array.from(unwrapTensor(predicted).dataSync()),
                    surprise: unwrapTensor(surprise).dataSync()[0]
                };
                [x_tT, x_nextT, memoryT, predicted, newMemory, surprise, cost].forEach(t => t.dispose());
                return result;
            }
            case 'forward_pass': {
                if (!this.model || !this.memoryVec) {
                    throw new Error('Model not initialized');
                }
                const { x } = request.params.arguments;
                const xT = wrapTensor(tf.tensor1d(x));
                const memoryT = wrapTensor(this.memoryVec);
                const { predicted, newMemory, surprise } = this.model.forward(xT, memoryT);
                // Update memory vector with new state
                this.memoryVec.assign(unwrapTensor(newMemory));
                const result = {
                    predicted: Array.from(unwrapTensor(predicted).dataSync()),
                    memory: Array.from(unwrapTensor(newMemory).dataSync()),
                    surprise: unwrapTensor(surprise).dataSync()[0]
                };
                [xT, memoryT, predicted, newMemory, surprise].forEach(t => t.dispose());
                return result;
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
                    memoryStats: stats,
                    memorySize: this.memoryVec.shape[0],
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
            if (this.memoryVec) {
                this.memoryVec.dispose();
                this.memoryVec = null;
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