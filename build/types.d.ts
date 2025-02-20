/**
 * @fileoverview Core type definitions for Titan Memory Architecture
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */
import * as tf from '@tensorflow/tfjs-node';
import { z } from 'zod';
export type ITensor = tf.Tensor;
export type TensorContainer = {
    [key: string]: tf.Tensor | TensorContainer;
};
/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export declare const wrapTensor: (t: tf.Tensor) => tf.Tensor<tf.Rank>;
/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Tensor to unwrap
 * @returns Underlying TensorFlow.js tensor
 */
export declare const unwrapTensor: (t: ITensor) => ITensor;
/**
 * Interface defining the core tensor operations available in the system.
 * Provides a subset of TensorFlow.js operations needed for the Titans implementation.
 */
export interface ITensorOps {
    tensor(data: number[], shape?: number[]): ITensor;
    tensor1d(data: number[]): ITensor;
    scalar(value: number): ITensor;
    zeros(shape: number[]): ITensor;
    randomNormal(shape: number[]): ITensor;
    variable(tensor: ITensor): ITensor;
    tidy<T extends tf.TensorContainer>(fn: () => T): T;
    train: {
        adam: (learningRate: number) => {
            minimize: (lossFn: () => tf.Scalar) => ITensor;
        };
    };
    concat(tensors: ITensor[], axis?: number): ITensor;
    matMul(a: ITensor, b: ITensor): ITensor;
    sub(a: ITensor, b: ITensor): ITensor;
    add(a: ITensor, b: ITensor): ITensor;
    mul(a: ITensor, b: ITensor): ITensor;
    div(a: ITensor, b: ITensor): ITensor;
    relu(x: ITensor): ITensor;
    sigmoid(x: ITensor): ITensor;
    tanh(x: ITensor): ITensor;
    mean(x: ITensor, axis?: number): ITensor;
    sum(x: ITensor, axis?: number): ITensor;
    sqrt(x: ITensor): ITensor;
    exp(x: ITensor): ITensor;
    log(x: ITensor): ITensor;
    dispose(): void;
    memory(): {
        numTensors: number;
        numDataBuffers: number;
        numBytes: number;
    };
}
export declare const TitanMemoryConfigSchema: z.ZodObject<{
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
}>;
export type TitanMemoryConfig = z.infer<typeof TitanMemoryConfigSchema>;
/**
 * Represents an attention block in the Titans architecture.
 * Contains the key-value pairs and attention scores used in the memory mechanism.
 */
export interface IAttentionBlock {
    keys: ITensor;
    values: ITensor;
    scores: ITensor;
}
/**
 * Tracks both immediate and accumulated surprise metrics.
 * Used to guide memory updates and test-time learning.
 */
export interface ISurpriseMetrics {
    immediate: ITensor;
    accumulated: ITensor;
}
/**
 * Represents the memory state in the Titans architecture.
 * Combines short-term, long-term, and meta-memory components with temporal dynamics.
 */
export interface IMemoryState {
    shortTerm: ITensor;
    longTerm: ITensor;
    meta: ITensor;
    timestamps: ITensor;
    accessCounts: ITensor;
    surpriseHistory: ITensor;
}
/**
 * Result of a memory update operation, containing the new memory state
 * and associated attention and surprise metrics.
 */
export interface IMemoryUpdateResult {
    newState: IMemoryState;
    attention: IAttentionBlock;
    surprise: ISurpriseMetrics;
}
/**
 * Gradients computed for each memory component during training.
 */
export interface IModelGradients {
    shortTerm: ITensor;
    longTerm: ITensor;
    meta: ITensor;
}
/**
 * Core interface for the Titans memory model.
 */
export interface IMemoryModel {
    forward(x: ITensor, memoryState: IMemoryState): {
        predicted: ITensor;
        memoryUpdate: IMemoryUpdateResult;
    };
    trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
        loss: ITensor;
        gradients: IModelGradients;
    };
    updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor;
    pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState;
    manifoldStep(base: ITensor, velocity: ITensor): ITensor;
    saveModel(path: string): Promise<void>;
    loadModel(path: string): Promise<void>;
    getConfig(): any;
    save(modelPath: string, weightsPath: string): Promise<void>;
    getMemorySnapshot(): Record<string, tf.Tensor>;
}
export interface ServerCapabilities {
    neuralMemory: boolean;
    onlineLearning: boolean;
    surpriseDetection: boolean;
}
export interface CallToolRequest {
    name: string;
    parameters: Record<string, unknown>;
}
export interface CallToolResult {
    success: boolean;
    result?: unknown;
    error?: string;
}
export interface Transport {
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
    send?(message: unknown): void;
}
export interface McpServer {
    tool(name: string, schema: z.ZodRawShape | string, handler: Function): void;
    connect(transport: Transport): Promise<void>;
}
export declare const StoreMemoryInput: z.ZodObject<{
    subject: z.ZodString;
    relationship: z.ZodString;
    object: z.ZodString;
}, "strip", z.ZodTypeAny, {
    object: string;
    subject: string;
    relationship: string;
}, {
    object: string;
    subject: string;
    relationship: string;
}>;
export declare const RecallMemoryInput: z.ZodObject<{
    query: z.ZodString;
}, "strip", z.ZodTypeAny, {
    query: string;
}, {
    query: string;
}>;
