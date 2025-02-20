#!/usr/bin/env node
import '@tensorflow/tfjs-node';
export declare class TitanMemoryServer {
    private model;
    private server;
    private memoryState;
    private memoryPath;
    private modelPath;
    private weightsPath;
    private autoSaveInterval;
    private isInitialized;
    constructor(config?: {
        memoryPath?: string;
        modelPath?: string;
        weightsPath?: string;
    });
    private ensureInitialized;
    private registerTools;
    private autoInitialize;
    private loadSavedState;
    private saveMemoryState;
    run(): Promise<void>;
}
