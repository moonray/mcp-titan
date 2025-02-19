/**
 * @fileoverview Core type definitions for the Titans memory architecture.
 * This file defines the interfaces and types used throughout the implementation
 * of the Titans memory model, including tensor operations, memory states, and
 * model interactions.
 */
import { z } from "zod";
/**
 * Creates a wrapped tensor from a TensorFlow.js tensor.
 * @param tensor TensorFlow.js tensor to wrap
 * @returns Wrapped tensor
 */
export function wrapTensor(tensor) {
    return tensor;
}
/**
 * Unwraps a tensor to get the underlying TensorFlow.js tensor.
 * @param tensor Tensor to unwrap
 * @returns Underlying TensorFlow.js tensor
 */
export function unwrapTensor(tensor) {
    return tensor;
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