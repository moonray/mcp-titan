#!/usr/bin/env node
import '@tensorflow/tfjs-node';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
export declare class TitanMemoryServer {
    protected server: Server;
    protected model: TitanMemoryModel | null;
    protected memoryVec: tf.Variable | null;
    private app;
    private port;
    private memoryPath;
    private autoSaveInterval;
    constructor(port?: number);
    private setupAutomaticMemory;
    private handleToolCall;
    run(): Promise<void>;
    private cleanup;
}
