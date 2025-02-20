/**
 * @fileoverview Core type definitions for Titan Memory Architecture
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */
import { z } from 'zod';
/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export const wrapTensor = (t) => t;
/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Tensor to unwrap
 * @returns Underlying TensorFlow.js tensor
 */
export const unwrapTensor = (t) => t;
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
// Memory Operation Schemas
export const StoreMemoryInput = z.object({
    subject: z.string(),
    relationship: z.string(),
    object: z.string()
});
export const RecallMemoryInput = z.object({
    query: z.string()
});
