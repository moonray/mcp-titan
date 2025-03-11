/**
 * @fileoverview Core type definitions for Titan Memory Architecture
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */

import * as tf from '@tensorflow/tfjs-node';
import { z } from 'zod';

// Core Tensor Operations
export type ITensor = tf.Tensor;
export type TensorContainer = { [key: string]: tf.Tensor | TensorContainer };

/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export const wrapTensor = (t: tf.Tensor): ITensor => t;

/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Tensor to unwrap
 * @returns Underlying TensorFlow.js tensor
 */
export const unwrapTensor = (t: ITensor): tf.Tensor => t;

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
  memory(): { numTensors: number; numDataBuffers: number; numBytes: number };
}

// Memory Configuration Schema
export const TitanMemoryConfigSchema = z.object({
  inputDim: z.number().int().positive().default(768),
  hiddenDim: z.number().int().positive().default(512),
  memoryDim: z.number().int().positive().default(1024),
  transformerLayers: z.number().int().positive().max(12).default(6),
  numHeads: z.number().int().positive().default(8),
  ffDimension: z.number().int().positive().default(2048),
  dropoutRate: z.number().min(0).max(0.9).default(0.1),
  maxSequenceLength: z.number().int().positive().default(512),
  memorySlots: z.number().int().positive().default(5000),
  similarityThreshold: z.number().min(0).max(1).default(0.65),
  surpriseDecay: z.number().min(0).max(1).default(0.9),
  pruningInterval: z.number().int().positive().default(1000),
  gradientClip: z.number().positive().default(1.0),
});

export type TitanMemoryConfig = z.infer<typeof TitanMemoryConfigSchema>;

/**
 * Interface for memory state in the Titans architecture.
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
 * Interface for attention block in transformer architecture.
 */
export interface IAttentionBlock {
  keys: ITensor;
  values: ITensor;
  scores: ITensor;
}

/**
 * Interface for surprise metrics in memory updates.
 */
export interface ISurpriseMetrics {
  immediate: ITensor;
  accumulated: ITensor;
}

/**
 * Interface for memory update results.
 */
export interface IMemoryUpdateResult {
  newState: IMemoryState;
  attention: IAttentionBlock;
  surprise: ISurpriseMetrics;
}

/**
 * Interface for model gradients.
 */
export interface IModelGradients {
  shortTerm: ITensor;
  longTerm: ITensor;
  meta: ITensor;
}

/**
 * Interface for memory manager operations.
 */
export interface IMemoryManager {
  validateVectorShape(tensor: tf.Tensor, expectedShape: number[]): boolean;
  encryptTensor(tensor: tf.Tensor): Buffer;
  decryptTensor(encrypted: Buffer, shape: number[]): tf.Tensor;
  wrapWithMemoryManagement<T extends tf.TensorContainer>(fn: () => T): T;
  wrapWithMemoryManagementAsync<T>(fn: () => Promise<T>): Promise<T>;
  dispose(): void;
}

/**
 * Interface for vector processing operations.
 */
export interface IVectorProcessor {
  processInput(input: number | number[] | string | tf.Tensor): tf.Tensor;
  validateAndNormalize(tensor: tf.Tensor, expectedShape: number[]): tf.Tensor;
  encodeText(text: string, maxLength?: number): Promise<tf.Tensor>;
}

/**
 * Interface for automatic memory maintenance operations.
 */
export interface IMemoryMaintenance {
  dispose(): void;
}

/**
 * Interface for the memory model.
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
  dispose(): void;
  resetGradients(): void;
  initialize(config?: any): Promise<void>;

  getMemoryState(): IMemoryState;
  resetMemory(): void;
  getMemoryState(): any;
  encodeText(text: string): Promise<tf.Tensor1D>;
  recallMemory(query: string, topK?: number): Promise<tf.Tensor2D[]>;
  storeMemory(text: string): Promise<void>;
  distillMemories?(similarMemories: tf.Tensor2D[]): tf.Tensor2D;
  quantizeMemory?(): IQuantizedMemoryState;
  dequantizeMemory?(quantizedState: IQuantizedMemoryState): IMemoryState;
  contrastiveLoss?(anchor: tf.Tensor2D, positive: tf.Tensor2D, negative: tf.Tensor2D, margin?: number): tf.Scalar;
  trainWithContrastiveLearning?(anchorText: string, positiveText: string, negativeText: string): Promise<number>;
  pruneMemoryByInformationGain?(threshold?: number): void;
  storeMemoryWithType?(text: string, isEpisodic?: boolean): Promise<void>;
  recallMemoryByType?(query: string, type?: 'episodic' | 'semantic' | 'both', topK?: number): Promise<tf.Tensor2D[]>;
  recallAndDistill?(query: string, topK?: number): Promise<tf.Tensor2D>;
}

// Memory Operation Schemas
export const StoreMemoryInput = z.object({
  subject: z.string(),
  relationship: z.string(),
  object: z.string()
});

export const RecallMemoryInput = z.object({
  query: z.string()
});

export interface IHierarchicalMemoryState extends IMemoryState {
  workingMemory: tf.Tensor2D;
  shortTermMemory: tf.Tensor2D;
  longTermMemory: tf.Tensor2D;
  workingAccessCounts: tf.Tensor1D;
  shortTermAccessCounts: tf.Tensor1D;
  longTermAccessCounts: tf.Tensor1D;
}

export interface IExtendedMemoryState extends IMemoryState {
  episodicMemory: tf.Tensor2D;
  semanticMemory: tf.Tensor2D;
  episodicTimestamps: tf.Tensor1D;
  semanticConfidence: tf.Tensor1D;
}

export interface IQuantizedMemoryState {
  shortTermQuantized: Int8Array;
  longTermQuantized: Int8Array;
  metaQuantized: Int8Array;
  shortTermShape: number[];
  longTermShape: number[];
  metaShape: number[];
  quantizer: any;
  timestamps: number[];
  accessCounts: number[];
  surpriseHistory: number[];
}

export interface ITelemetryData {
  timestamp: number;
  operation: string;
  durationMs: number;
  memoryUsage: {
    numTensors: number;
    numBytes: number;
    unreliable: boolean;
  };
  metrics?: Record<string, number>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
}

// Add custom error classes
export class TensorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TensorError';
  }
}

export class MemoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MemoryError';
  }
}
