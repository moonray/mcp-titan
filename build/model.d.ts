/**
 * @fileoverview Titans Memory Model Implementation
 */
import { ITensor, IMemoryModel, IMemoryState, ISurpriseMetrics, IMemoryUpdateResult, IModelGradients } from './types.js';
export interface TitanMemoryConfig {
    inputDim?: number;
    hiddenDim?: number;
    outputDim?: number;
    learningRate?: number;
    useManifold?: boolean;
    momentumFactor?: number;
    attentionHeads?: number;
    keyDim?: number;
    valueDim?: number;
    surpriseThreshold?: number;
    metaLearningRate?: number;
    maxMemorySize?: number;
}
export declare class TitanMemoryModel implements IMemoryModel {
    private inputDim;
    private hiddenDim;
    private memoryDim;
    private learningRate;
    private useManifold;
    private momentumFactor;
    private attentionHeads;
    private keyDim;
    private valueDim;
    private surpriseThreshold;
    private metaLearningRate;
    private maxMemorySize;
    private combinedContextSize;
    private surpriseInputSize;
    private pruningInputSize;
    private queryProjection;
    private keyProjection;
    private valueProjection;
    private outputProjection;
    private shortTermEncoder;
    private longTermEncoder;
    private metaEncoder;
    private shortTermDecoder;
    private longTermDecoder;
    private metaDecoder;
    private surpriseNetwork;
    private pruningNetwork;
    private optimizer;
    private metaOptimizer;
    constructor(config?: TitanMemoryConfig);
    updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor;
    manifoldStep(base: ITensor, velocity: ITensor): ITensor;
    save(modelPath: string, weightsPath: string): Promise<void>;
    private computeAttention;
    private computeSurprise;
    forward(x: ITensor, memoryState: IMemoryState): {
        predicted: ITensor;
        memoryUpdate: IMemoryUpdateResult;
    };
    trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
        loss: ITensor;
        gradients: IModelGradients;
    };
    pruneMemory(memoryState: IMemoryState): IMemoryState;
    saveModel(path: string): Promise<void>;
    loadModel(path: string): Promise<void>;
    getConfig(): TitanMemoryConfig;
}
