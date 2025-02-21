#!/usr/bin/env node
import { z } from "zod";
import * as tf from '@tensorflow/tfjs-node';
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { wrapTensor, unwrapTensor } from './types.js';
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
    x: z.union([z.string(), z.number(), z.array(z.number()), z.custom()])
});
const TrainStepParams = z.object({
    x_t: z.union([z.string(), z.array(z.number())]),
    x_next: z.union([z.string(), z.array(z.number())])
});
const GetMemoryStateParams = z.object({
    type: z.string().optional()
});
export class TitanMemoryServer {
    server;
    model;
    vectorProcessor;
    memoryState;
    isInitialized = false;
    autoSaveInterval;
    memoryPath;
    modelPath;
    weightsPath;
    testMode;
    toolSchemas;
    constructor(options = {}) {
        console.log("Initializing TitanMemoryServer...");
        this.testMode = options.testMode ?? false;
        this.memoryPath = options.memoryPath ?? path.join(process.cwd(), 'memory.json');
        this.modelPath = path.join(process.cwd(), 'model.json');
        this.weightsPath = path.join(process.cwd(), 'weights.bin');
        this.toolSchemas = {
            help: {
                description: "Get help about available tools",
                parameters: {
                    type: "object",
                    properties: {
                        tool: { type: "string", description: "Tool to get help for" },
                        category: { type: "string", description: "Category of tools" },
                        showExamples: { type: "boolean", description: "Show examples" },
                        verbose: { type: "boolean", description: "Show detailed info" }
                    }
                }
            },
            init_model: {
                description: "Initialize the memory model",
                parameters: {
                    type: "object",
                    properties: {
                        inputDim: { type: "number", description: "Input dimension" },
                        memorySlots: { type: "number", description: "Memory slot count" },
                        transformerLayers: { type: "number", description: "Transformer layers" }
                    }
                }
            },
            forward_pass: {
                description: "Process input through model",
                parameters: {
                    type: "object",
                    properties: {
                        x: {
                            oneOf: [
                                { type: "string", description: "Text input" },
                                {
                                    type: "array",
                                    items: { type: "number" },
                                    description: "Vector input"
                                }
                            ]
                        }
                    },
                    required: ["x"]
                }
            },
            train_step: {
                description: "Train on sequential inputs",
                parameters: {
                    type: "object",
                    properties: {
                        x_t: {
                            oneOf: [
                                { type: "string", description: "Current text" },
                                {
                                    type: "array",
                                    items: { type: "number" },
                                    description: "Current vector"
                                }
                            ]
                        },
                        x_next: {
                            oneOf: [
                                { type: "string", description: "Next text" },
                                {
                                    type: "array",
                                    items: { type: "number" },
                                    description: "Next vector"
                                }
                            ]
                        }
                    },
                    required: ["x_t", "x_next"]
                }
            },
            get_memory_state: {
                description: "Get memory state",
                parameters: {
                    type: "object",
                    properties: {
                        type: {
                            type: "string",
                            enum: ["short_term", "long_term", "meta", "all"],
                            description: "Memory type"
                        }
                    }
                }
            }
        };
        // Initialize components
        this.vectorProcessor = VectorProcessor.getInstance();
        this.memoryState = this.initializeMemoryState();
        console.log("Components initialized");
        // Initialize server and register tools
        this.initializeServer();
        console.log("Server initialized with tools");
    }
    initializeServer() {
        // Create server with tools schema
        this.server = new McpServer({
            name: "Titan Memory",
            version: "1.3.1",
            description: "Memory-augmented model server",
            transport: "stdio",
            tools: this.toolSchemas
        });
        // Register tool handlers
        this.server.tool("help", HelpParams.shape, async (params, extra) => ({
            success: true,
            content: [{ type: "text", text: "Available tools: help, init_model, forward_pass, train_step, get_memory_state" }]
        }));
        this.server.tool("init_model", InitModelParams.shape, async (params, extra) => {
            await this.ensureInitialized();
            return { success: true, content: [{ type: "text", text: "Model initialized" }] };
        });
        this.server.tool("forward_pass", ForwardPassParams.shape, async (params, extra) => {
            await this.ensureInitialized();
            const input = await this.processInput(params.x);
            const result = await this.model.forward(input, this.memoryState);
            return { success: true, content: [{ type: "text", text: "Forward pass complete" }], data: result };
        });
        this.server.tool("train_step", TrainStepParams.shape, async (params, extra) => {
            await this.ensureInitialized();
            const x_t = await this.processInput(params.x_t);
            const x_next = await this.processInput(params.x_next);
            const result = await this.model.trainStep(x_t, x_next, this.memoryState);
            return { success: true, content: [{ type: "text", text: "Training complete" }], data: result };
        });
        this.server.tool("get_memory_state", GetMemoryStateParams.shape, async (params, extra) => {
            await this.ensureInitialized();
            const state = this.model.getMemorySnapshot();
            return { success: true, content: [{ type: "text", text: "Memory state retrieved" }], data: state };
        });
    }
    async processInput(input) {
        if (typeof input === 'string') {
            return this.vectorProcessor.encodeText(input);
        }
        else if (Array.isArray(input)) {
            return tf.tensor(input);
        }
        else if (input instanceof tf.Tensor) {
            return input;
        }
        else if (typeof input === 'number') {
            return tf.scalar(input);
        }
        throw new Error('Invalid input type');
    }
    initializeMemoryState() {
        return tf.tidy(() => ({
            shortTerm: wrapTensor(tf.zeros([0])),
            longTerm: wrapTensor(tf.zeros([0])),
            meta: wrapTensor(tf.zeros([0])),
            timestamps: wrapTensor(tf.zeros([0])),
            accessCounts: wrapTensor(tf.zeros([0])),
            surpriseHistory: wrapTensor(tf.zeros([0]))
        }));
    }
    wrapWithMemoryManagement(fn) {
        return tf.tidy(fn);
    }
    async wrapWithMemoryManagementAsync(fn) {
        tf.engine().startScope();
        try {
            return await fn();
        }
        finally {
            tf.engine().endScope();
        }
    }
    encryptTensor(tensor) {
        const data = tensor.dataSync();
        const key = crypto.randomBytes(32);
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
        const encrypted = Buffer.concat([cipher.update(Buffer.from(data.buffer)), cipher.final()]);
        return new Uint8Array(Buffer.concat([iv, key, encrypted]));
    }
    validateMemoryState(state) {
        return tf.tidy(() => {
            return (state.shortTerm !== undefined &&
                state.longTerm !== undefined &&
                state.meta !== undefined &&
                state.timestamps !== undefined &&
                state.accessCounts !== undefined &&
                state.surpriseHistory !== undefined);
        });
    }
    async ensureInitialized() {
        if (!this.isInitialized) {
            await this.autoInitialize();
            this.isInitialized = true;
        }
    }
    async autoInitialize() {
        if (this.isInitialized)
            return;
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
        }
        catch (error) {
            console.error('Initialization failed:', error instanceof Error ? error.message : error);
            throw error;
        }
    }
    async loadSavedState() {
        try {
            const [modelExists, weightsExist] = await Promise.all([
                fs.access(this.modelPath).then(() => true).catch(() => false),
                fs.access(this.weightsPath).then(() => true).catch(() => false)
            ]);
            if (!modelExists || !weightsExist)
                return false;
            await this.model.loadModel(this.modelPath);
            const memoryStateJson = await fs.readFile(path.join(this.memoryPath, 'memory_state.json'), 'utf8');
            const memoryState = JSON.parse(memoryStateJson);
            this.memoryState = this.wrapWithMemoryManagement(() => ({
                shortTerm: wrapTensor(tf.tensor(memoryState.shortTerm)),
                longTerm: wrapTensor(tf.tensor(memoryState.longTerm)),
                meta: wrapTensor(tf.tensor(memoryState.meta)),
                timestamps: wrapTensor(tf.tensor(memoryState.timestamps)),
                accessCounts: wrapTensor(tf.tensor(memoryState.accessCounts)),
                surpriseHistory: wrapTensor(tf.tensor(memoryState.surpriseHistory))
            }));
            return true;
        }
        catch (error) {
            console.error('Failed to load saved state:', error instanceof Error ? error.message : error);
            return false;
        }
    }
    async saveMemoryState() {
        try {
            await this.model.saveModel(this.modelPath);
            const memoryState = this.wrapWithMemoryManagement(() => {
                const state = {
                    shortTerm: Array.from(unwrapTensor(this.memoryState.shortTerm).dataSync()),
                    longTerm: Array.from(unwrapTensor(this.memoryState.longTerm).dataSync()),
                    meta: Array.from(unwrapTensor(this.memoryState.meta).dataSync()),
                    timestamps: Array.from(unwrapTensor(this.memoryState.timestamps).dataSync()),
                    accessCounts: Array.from(unwrapTensor(this.memoryState.accessCounts).dataSync()),
                    surpriseHistory: Array.from(unwrapTensor(this.memoryState.surpriseHistory).dataSync())
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
        }
        catch (error) {
            console.error('Failed to save memory state:', error instanceof Error ? error.message : error);
            throw error;
        }
    }
    async run() {
        try {
            console.log("Starting TitanMemoryServer...");
            await this.autoInitialize();
            const transport = new StdioServerTransport();
            console.log("Transport created");
            // Send server info before connecting
            transport.send({
                jsonrpc: "2.0",
                method: "server.info",
                params: {
                    name: "Titan Memory",
                    version: "1.3.1",
                    description: "Memory-augmented model server",
                    transport: "stdio",
                    tools: this.toolSchemas
                }
            });
            console.log("Server info sent");
            await this.server.connect(transport);
            console.log("Server connected");
            if (!this.testMode) {
                this.startAutoSave();
                console.log("Auto-save started");
            }
        }
        catch (error) {
            console.error("Server startup error:", error);
            process.exit(1);
        }
    }
    startAutoSave() {
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
        }
        this.autoSaveInterval = setInterval(async () => {
            await this.saveMemoryState().catch(console.error);
        }, 300000); // Save every 5 minutes
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
