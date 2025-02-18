#!/usr/bin/env node
import '@tensorflow/tfjs-node';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { TitanMemoryModel } from './model.js';
import { IMemoryState } from './types.js';
interface TitanMemoryConfig {
    port?: number;
    memoryPath?: string;
    modelPath?: string;
    weightsPath?: string;
    inputDim?: number;
    outputDim?: number;
}
export declare class TitanMemoryServer {
    protected server: Server;
    protected model: TitanMemoryModel | null;
    protected memoryState: IMemoryState | null;
    private app;
    private port;
    private memoryPath;
    private modelPath;
    private weightsPath;
    private autoSaveInterval;
    private reconnectAttempts;
    private readonly MAX_RECONNECT_ATTEMPTS;
    private readonly RECONNECT_DELAY;
    private wsServer;
    private readonly DEFAULT_WS_PORT;
    private readonly AUTO_RECONNECT_INTERVAL;
    private isAutoReconnectEnabled;
    constructor(config?: TitanMemoryConfig);
    private setupWebSocket;
    private autoInitialize;
    private loadSavedState;
    private saveMemoryState;
    private reconnect;
    private assertModelInitialized;
    cleanup(): Promise<void>;
    private handleToolCall;
    run(): Promise<void>;
}
export {};
