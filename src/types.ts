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
export const wrapTensor = (t: tf.Tensor) => t;

/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Tensor to unwrap
 * @returns Underlying TensorFlow.js tensor
 */
export const unwrapTensor = (t: ITensor) => t;

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
  processInput(input: number[] | string | tf.Tensor): tf.Tensor;
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
}

/**
 * Interface for server capabilities.
 */
export interface ServerCapabilities {
  name: string;
  version: string;
  description?: string;
  transport: string;
  tools: Record<string, {
    description: string;
    parameters: Record<string, unknown>;
  }>;
}

/**
 * Interface for tool call requests.
 */
export interface CallToolRequest {
  name: string;
  parameters: Record<string, unknown>;
}

/**
 * Interface for tool call results.
 */
export interface CallToolResult {
  content: Array<{
    type: string;
    text: string;
  }>;
}

/**
 * Interface for transport layer.
 */
export interface Transport {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
  send?(message: unknown): void;
}

/**
 * Interface for MCP server.
 */
export interface McpServer {
  tool(name: string, schema: z.ZodRawShape | string, handler: Function): void;
  connect(transport: Transport): Promise<void>;
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