/**
 * @fileovertitle Titan Memory Model 2.0 - Neural Memory Architecture with Transformer-XL Inspired Mechanisms
 */
import * as tf from '@tensorflow/tfjs-node';
import { ITensor, IMemoryState, ISurpriseMetrics, IMemoryUpdateResult, IModelGradients, IMemoryModel } from './types.js';
import { z } from 'zod';
declare const ModelConfigSchema: z.ZodObject<{
    inputDim: z.ZodDefault<z.ZodNumber>;
    hiddenDim: z.ZodDefault<z.ZodNumber>;
    memoryDim: z.ZodDefault<z.ZodNumber>;
    transformerLayers: z.ZodDefault<z.ZodNumber>;
    numHeads: z.ZodDefault<z.ZodNumber>;
    ffDimension: z.ZodDefault<z.ZodNumber>;
    dropoutRate: z.ZodDefault<z.ZodNumber>;
    maxSequenceLength: z.ZodDefault<z.ZodNumber>;
    memorySlots: z.ZodDefault<z.ZodNumber>;
    similarityThreshold: z.ZodDefault<z.ZodNumber>;
    surpriseDecay: z.ZodDefault<z.ZodNumber>;
    pruningInterval: z.ZodDefault<z.ZodNumber>;
    gradientClip: z.ZodDefault<z.ZodNumber>;
    learningRate: z.ZodDefault<z.ZodNumber>;
    vocabSize: z.ZodDefault<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    inputDim: number;
    hiddenDim: number;
    memoryDim: number;
    transformerLayers: number;
    numHeads: number;
    ffDimension: number;
    dropoutRate: number;
    maxSequenceLength: number;
    memorySlots: number;
    similarityThreshold: number;
    surpriseDecay: number;
    pruningInterval: number;
    gradientClip: number;
    learningRate: number;
    vocabSize: number;
}, {
    inputDim?: number | undefined;
    hiddenDim?: number | undefined;
    memoryDim?: number | undefined;
    transformerLayers?: number | undefined;
    numHeads?: number | undefined;
    ffDimension?: number | undefined;
    dropoutRate?: number | undefined;
    maxSequenceLength?: number | undefined;
    memorySlots?: number | undefined;
    similarityThreshold?: number | undefined;
    surpriseDecay?: number | undefined;
    pruningInterval?: number | undefined;
    gradientClip?: number | undefined;
    learningRate?: number | undefined;
    vocabSize?: number | undefined;
}>;
type TitanMemoryConfig = z.infer<typeof ModelConfigSchema>;
export declare class TitanMemoryModel implements IMemoryModel {
    private config;
    private transformerStack;
    private memoryProjector;
    private similarityNetwork;
    private optimizer;
    private stepCount;
    private vocabulary;
    private reverseVocabulary;
    private memoryState;
    constructor(config?: Partial<TitanMemoryConfig>);
    private initializeVocabulary;
    encodeText(text: string): Promise<tf.Tensor1D>;
    private tokenize;
    private padSequence;
    private applyPositionalEncoding;
    private initializeComponents;
    private initializeMemoryState;
    storeMemory(text: string): Promise<void>;
    private calculateSimilarity;
    private addMemoryEntry;
    private updateAccessStats;
    private checkPruning;
    pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState;
    private computeMemoryRelevance;
    recallMemory(query: string, topK?: number): Promise<tf.Tensor2D[]>;
    forward(x: ITensor, memoryState: IMemoryState): {
        predicted: ITensor;
        memoryUpdate: IMemoryUpdateResult;
    };
    private computeMemoryAttention;
    private computeSurprise;
    trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
        loss: ITensor;
        gradients: IModelGradients;
    };
    updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor;
    manifoldStep(base: ITensor, velocity: ITensor): ITensor;
    getConfig(): TitanMemoryConfig;
    saveModel(path: string): Promise<void>;
    save(modelPath: string, weightsPath: string): Promise<void>;
    private getWeightData;
    loadModel(path: string): Promise<void>;
    getMemorySnapshot(): Record<string, tf.Tensor>;
    dispose(): void;
}
export {};
