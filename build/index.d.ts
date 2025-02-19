#!/usr/bin/env node
import '@tensorflow/tfjs-node';
export declare class TitanMemoryServer {
    private model;
    private server;
    private memoryState;
    private app;
    private port;
    private memoryPath;
    private modelPath;
    private weightsPath;
    private autoSaveInterval;
    constructor(config?: {
        port?: number;
        memoryPath?: string;
        modelPath?: string;
        weightsPath?: string;
    });
    private registerTools;
    run(): Promise<void>;
    private autoInitialize;
    private loadSavedState;
    private saveMemoryState;
}
