/**
 * @fileoverview Core type definitions for the Titans memory architecture.
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */
import { z } from "zod";
/**
 * Wrapper class for TensorFlow.js tensors.
 * Provides a consistent interface for tensor operations while managing underlying TF.js tensors.
 */
export class TensorWrapper {
    constructor(tensor) {
        this.tensor = tensor;
    }
    static fromTensor(tensor) {
        return new TensorWrapper(tensor);
    }
    get shape() {
        return this.tensor.shape;
    }
    dataSync() {
        return this.tensor.dataSync();
    }
    dispose() {
        this.tensor.dispose();
    }
    toJSON() {
        return {
            dataSync: Array.from(this.dataSync()),
            shape: this.shape
        };
    }
}
/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export function wrapTensor(tensor) {
    return {
        data: tensor,
        dispose: () => tensor.dispose(),
        dataSync: () => tensor.dataSync(),
        shape: tensor.shape
    };
}
/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Wrapped tensor
 * @returns Underlying TensorFlow.js tensor
 * @throws Error if tensor is not a TensorWrapper
 */
export function unwrapTensor(tensor) {
    return tensor.data;
}
/**
 * Zod schema for memory storage input validation.
 */
export const StoreMemoryInput = z.object({
    subject: z.string(),
    relationship: z.string(),
    object: z.string()
});
/**
 * Zod schema for memory recall input validation.
 */
export const RecallMemoryInput = z.object({
    query: z.string()
});
//# sourceMappingURL=types.js.map