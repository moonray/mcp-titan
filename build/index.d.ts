#!/usr/bin/env node
import '@tensorflow/tfjs-node';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { TitanMemoryModel } from './model.js';
import { IMemoryState } from './types.js';
export declare class TitanMemoryServer {
    protected server: Server;
    protected model: TitanMemoryModel | null;
    protected memoryState: IMemoryState | null;
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
