/**
 * @fileoverview Implementation of the Titans memory architecture.
 *
 * This file contains the core implementation of the Titans memory model,
 * which introduces a novel approach to neural memory that learns to memorize
 * at test time. The architecture features:
 *
 * 1. Three-tier Memory System:
 *    - Short-term memory for immediate context
 *    - Long-term memory for persistent information
 *    - Meta-memory that learns how to memorize during inference
 *
 * 2. Advanced Attention Mechanism:
 *    - Multi-head attention for flexible memory access
 *    - Key-value associative memory blocks
 *    - Attention-based context integration
 *
 * 3. Surprise-based Memory Management:
 *    - Dual tracking of immediate and accumulated surprise
 *    - Dynamic memory updates based on surprise metrics
 *    - Importance-based memory pruning
 *
 * 4. Test-time Learning:
 *    - Meta-memory updates during inference
 *    - Surprise-guided memory management
 *    - Efficient memory utilization with size constraints
 */
import { ITensor, IMemoryModel, IMemoryState, ISurpriseMetrics, IMemoryUpdateResult, IModelGradients } from './types.js';
/**
 * Configuration options for the Titans memory model.
 */
export interface TitanMemoryConfig {
    /** Input dimension */
    inputDim?: number;
    /** Hidden layer dimension */
    hiddenDim?: number;
    /** Output/memory dimension */
    outputDim?: number;
    /** Base learning rate for model updates */
    learningRate?: number;
    /** Whether to use Riemannian manifold optimization */
    useManifold?: boolean;
    /** Momentum factor for optimization */
    momentumFactor?: number;
    /** Number of attention heads */
    attentionHeads?: number;
    /** Dimension of attention keys */
    keyDim?: number;
    /** Dimension of attention values */
    valueDim?: number;
    /** Threshold for surprise-based updates */
    surpriseThreshold?: number;
    /** Learning rate for meta-memory updates */
    metaLearningRate?: number;
    /** Maximum memory size to prevent unbounded growth */
    maxMemorySize?: number;
}
/**
 * Implementation of the Titans memory model.
 *
 * This class provides a complete implementation of the Titans architecture,
 * featuring multi-head attention, three-tier memory, and test-time learning
 * capabilities.
 */
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
    /**
     * Creates a new instance of the Titans memory model.
     * Initializes all components including attention mechanisms, memory modules,
     * and optimization parameters.
     *
     * @param config Configuration options for the model
     */
    constructor(config?: TitanMemoryConfig);
    /**
     * Computes multi-head attention over the memory states.
     *
     * @param query Query tensor for attention computation
     * @param keys Key tensors to attend over
     * @param values Value tensors to be weighted
     * @returns Attention block containing keys, values, and computed scores
     */
    private computeAttention;
    /**
     * Computes immediate and accumulated surprise metrics.
     * These metrics guide memory updates and test-time learning.
     *
     * @param predicted Predicted tensor
     * @param actual Actual/target tensor
     * @param history Historical context for surprise computation
     * @returns Immediate and accumulated surprise metrics
     */
    private computeSurprise;
    /**
     * Performs a forward pass through the model.
     *
     * This method:
     * 1. Encodes input and memory states
     * 2. Computes attention over memories
     * 3. Generates predictions
     * 4. Updates memory states
     * 5. Computes surprise metrics
     *
     * @param x Input tensor
     * @param memoryState Current memory state
     * @returns Predicted output and memory updates
     */
    forward(x: ITensor, memoryState: IMemoryState): {
        predicted: ITensor;
        memoryUpdate: IMemoryUpdateResult;
    };
    /**
     * Performs a training step.
     *
     * Updates model parameters using:
     * 1. Prediction loss
     * 2. Surprise regularization
     * 3. Separate gradients for each memory component
     *
     * @param x_t Current input
     * @param x_next Next input (target)
     * @param memoryState Current memory state
     * @returns Loss and computed gradients
     */
    trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
        loss: ITensor;
        gradients: IModelGradients;
    };
    /**
     * Updates meta-memory based on surprise metrics.
     *
     * This method implements the core test-time learning mechanism,
     * allowing the model to adapt its memorization strategy during inference.
     *
     * @param surprise Current surprise metrics
     * @param context Context tensor
     * @returns Updated meta-memory
     */
    updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor;
    /**
     * Prunes memory based on importance scores.
     *
     * Uses a learned pruning network to identify and remove less important
     * memories, maintaining efficiency and preventing memory overflow.
     *
     * @param memoryState Current memory state
     * @param threshold Pruning threshold
     * @returns Pruned memory state
     */
    pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState;
    /**
     * Performs a manifold optimization step.
     *
     * Implements Riemannian gradient descent when useManifold is true,
     * otherwise performs standard Euclidean updates.
     *
     * @param base Base point on manifold
     * @param velocity Update direction
     * @returns Updated point on manifold
     */
    manifoldStep(base: ITensor, velocity: ITensor): ITensor;
    /**
     * Saves model weights to disk.
     *
     * Serializes all trainable parameters including:
     * - Attention components
     * - Memory encoders/decoders
     * - Surprise and pruning networks
     *
     * @param path File path
     */
    saveModel(path: string): Promise<void>;
    /**
     * Loads model weights from disk.
     *
     * Deserializes and assigns all trainable parameters.
     *
     * @param path File path
     */
    loadModel(path: string): Promise<void>;
    /**
     * Returns current model configuration.
     *
     * @returns Configuration object with all current parameter values
     */
    getConfig(): TitanMemoryConfig;
}
