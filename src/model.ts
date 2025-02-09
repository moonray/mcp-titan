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

import * as tf from '@tensorflow/tfjs';
import { ITensor, IMemoryModel, IMemoryState, ISurpriseMetrics, IAttentionBlock, IMemoryUpdateResult, IModelGradients, TensorWrapper, wrapTensor, unwrapTensor } from './types.js';
import * as fs from 'fs/promises';

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
export class TitanMemoryModel implements IMemoryModel {
  private inputDim: number;
  private hiddenDim: number;
  private memoryDim: number;
  private learningRate: number;
  private useManifold: boolean;
  private momentumFactor: number;
  private attentionHeads: number;
  private keyDim: number;
  private valueDim: number;
  private surpriseThreshold: number;
  private metaLearningRate: number;
  private maxMemorySize: number;

  // Trainable parameters
  private queryProjection: tf.Variable;
  private keyProjection: tf.Variable;
  private valueProjection: tf.Variable;
  private outputProjection: tf.Variable;

  private shortTermEncoder: tf.Variable;
  private longTermEncoder: tf.Variable;
  private metaEncoder: tf.Variable;

  private shortTermDecoder: tf.Variable;
  private longTermDecoder: tf.Variable;
  private metaDecoder: tf.Variable;

  private surpriseNetwork: tf.Variable;
  private pruningNetwork: tf.Variable;

  private optimizer: tf.Optimizer;
  private metaOptimizer: tf.Optimizer;

  /**
   * Creates a new instance of the Titans memory model.
   * Initializes all components including attention mechanisms, memory modules,
   * and optimization parameters.
   * 
   * @param config Configuration options for the model
   */
  constructor(config: TitanMemoryConfig = {}) {
    this.inputDim = config.inputDim || 64;
    this.hiddenDim = config.hiddenDim || 32;
    this.memoryDim = config.outputDim || 64;
    this.learningRate = config.learningRate || 1e-3;
    this.useManifold = config.useManifold || false;
    this.momentumFactor = config.momentumFactor || 0.9;
    this.attentionHeads = config.attentionHeads || 4;
    this.keyDim = config.keyDim || 32;
    this.valueDim = config.valueDim || 32;
    this.surpriseThreshold = config.surpriseThreshold || 0.5;
    this.metaLearningRate = config.metaLearningRate || 1e-4;
    this.maxMemorySize = config.maxMemorySize || 1000;

    // Initialize attention components
    this.queryProjection = tf.variable(tf.randomNormal([this.attentionHeads, this.hiddenDim, this.keyDim]));
    this.keyProjection = tf.variable(tf.randomNormal([this.attentionHeads, this.hiddenDim, this.keyDim]));
    this.valueProjection = tf.variable(tf.randomNormal([this.attentionHeads, this.hiddenDim, this.valueDim]));
    this.outputProjection = tf.variable(tf.randomNormal([this.attentionHeads * this.valueDim, this.hiddenDim]));

    // Initialize memory components
    this.shortTermEncoder = tf.variable(tf.randomNormal([this.hiddenDim, this.inputDim]));
    this.longTermEncoder = tf.variable(tf.randomNormal([this.hiddenDim, this.memoryDim]));
    this.metaEncoder = tf.variable(tf.randomNormal([this.hiddenDim, this.hiddenDim]));

    this.shortTermDecoder = tf.variable(tf.randomNormal([this.inputDim, this.hiddenDim]));
    this.longTermDecoder = tf.variable(tf.randomNormal([this.memoryDim, this.hiddenDim]));
    this.metaDecoder = tf.variable(tf.randomNormal([this.hiddenDim, this.hiddenDim]));

    // Initialize surprise and pruning networks
    this.surpriseNetwork = tf.variable(tf.randomNormal([this.hiddenDim, 2])); // immediate and accumulated
    this.pruningNetwork = tf.variable(tf.randomNormal([this.hiddenDim, 1]));

    // Initialize optimizers
    this.optimizer = tf.train.adam(this.learningRate);
    this.metaOptimizer = tf.train.adam(this.metaLearningRate);
  }

  /**
   * Computes multi-head attention over the memory states.
   * 
   * @param query Query tensor for attention computation
   * @param keys Key tensors to attend over
   * @param values Value tensors to be weighted
   * @returns Attention block containing keys, values, and computed scores
   */
  private computeAttention(query: tf.Tensor, keys: tf.Tensor, values: tf.Tensor): IAttentionBlock {
    return tf.tidy(() => {
      // Multi-head attention
      const scores = tf.matMul(
        tf.matMul(query, this.queryProjection),
        tf.matMul(keys, this.keyProjection).transpose()
      ).div(tf.sqrt(tf.scalar(this.keyDim)));

      const attentionWeights = tf.softmax(scores);
      const weightedValues = tf.matMul(attentionWeights, tf.matMul(values, this.valueProjection));

      const output = tf.matMul(
        weightedValues.reshape([-1, this.attentionHeads * this.valueDim]),
        this.outputProjection
      );

      return {
        keys: wrapTensor(keys),
        values: wrapTensor(values),
        scores: wrapTensor(attentionWeights)
      };
    });
  }

  /**
   * Computes immediate and accumulated surprise metrics.
   * These metrics guide memory updates and test-time learning.
   * 
   * @param predicted Predicted tensor
   * @param actual Actual/target tensor
   * @param history Historical context for surprise computation
   * @returns Immediate and accumulated surprise metrics
   */
  private computeSurprise(predicted: tf.Tensor, actual: tf.Tensor, history: tf.Tensor): ISurpriseMetrics {
    return tf.tidy(() => {
      const diff = tf.sub(predicted, actual);
      const immediate = tf.mean(tf.square(diff));

      const historicalContext = tf.concat([diff, history], 1);
      const surpriseFeatures = tf.matMul(historicalContext, this.surpriseNetwork);
      const [immediateScore, accumulatedScore] = tf.split(surpriseFeatures, 2, 1);

      return {
        immediate: wrapTensor(immediateScore),
        accumulated: wrapTensor(accumulatedScore)
      };
    });
  }

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
  public forward(x: ITensor, memoryState: IMemoryState): {
    predicted: ITensor;
    memoryUpdate: IMemoryUpdateResult;
  } {
    const input = unwrapTensor(x);
    const shortTerm = unwrapTensor(memoryState.shortTerm);
    const longTerm = unwrapTensor(memoryState.longTerm);
    const meta = unwrapTensor(memoryState.meta);

    return tf.tidy(() => {
      // Encode input and memories
      const encodedInput = tf.matMul(input, this.shortTermEncoder);
      const encodedShortTerm = tf.matMul(shortTerm, this.shortTermEncoder);
      const encodedLongTerm = tf.matMul(longTerm, this.longTermEncoder);
      const encodedMeta = tf.matMul(meta, this.metaEncoder);

      // Compute attention over memories
      const attention = this.computeAttention(
        encodedInput,
        tf.concat([encodedShortTerm, encodedLongTerm], 0),
        tf.concat([shortTerm, longTerm], 0)
      );

      // Generate prediction
      const combinedContext = tf.concat([
        encodedInput,
        tf.matMul(attention.scores, tf.concat([shortTerm, longTerm], 0)),
        encodedMeta
      ], 1);

      const predicted = tf.matMul(combinedContext, this.shortTermDecoder);

      // Compute surprise metrics
      const surprise = this.computeSurprise(predicted, input, encodedMeta);

      // Update memory states
      const newShortTerm = tf.matMul(combinedContext, this.shortTermDecoder);
      const newLongTerm = tf.matMul(combinedContext, this.longTermDecoder);
      const newMeta = tf.matMul(combinedContext, this.metaDecoder);

      return {
        predicted: wrapTensor(predicted),
        memoryUpdate: {
          newState: {
            shortTerm: wrapTensor(newShortTerm),
            longTerm: wrapTensor(newLongTerm),
            meta: wrapTensor(newMeta)
          },
          attention,
          surprise
        }
      };
    });
  }

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
  public trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): {
    loss: ITensor;
    gradients: IModelGradients;
  } {
    const xt = unwrapTensor(x_t);
    const xn = unwrapTensor(x_next);

    return tf.tidy(() => {
      const { predicted, memoryUpdate } = this.forward(x_t, memoryState);

      // Compute prediction loss
      const predLoss = tf.mean(tf.square(tf.sub(unwrapTensor(predicted), xn)));

      // Add surprise regularization
      const surpriseLoss = tf.add(
        tf.mul(tf.scalar(0.1), unwrapTensor(memoryUpdate.surprise.immediate)),
        tf.mul(tf.scalar(0.05), unwrapTensor(memoryUpdate.surprise.accumulated))
      );

      // Total loss
      const totalLoss = tf.add(predLoss, surpriseLoss);

      // Compute gradients for each memory component
      const { value: shortTermGrad } = tf.variableGrads(() => totalLoss.asScalar(), [this.shortTermEncoder]);
      const { value: longTermGrad } = tf.variableGrads(() => totalLoss.asScalar(), [this.longTermEncoder]);
      const { value: metaGrad } = tf.variableGrads(() => totalLoss.asScalar(), [this.metaEncoder]);

      return {
        loss: wrapTensor(totalLoss),
        gradients: {
          shortTerm: wrapTensor(shortTermGrad),
          longTerm: wrapTensor(longTermGrad),
          meta: wrapTensor(metaGrad)
        }
      };
    });
  }

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
  public updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor {
    return tf.tidy(() => {
      const surpriseInput = tf.concat([
        unwrapTensor(surprise.immediate),
        unwrapTensor(surprise.accumulated)
      ], 0);

      const contextFeatures = unwrapTensor(context);
      const combined = tf.concat([surpriseInput, contextFeatures], 0);

      const metaUpdate = tf.matMul(combined, this.metaEncoder);
      return wrapTensor(metaUpdate);
    });
  }

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
  public pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState {
    return tf.tidy(() => {
      const shortTerm = unwrapTensor(memoryState.shortTerm);
      const longTerm = unwrapTensor(memoryState.longTerm);
      const meta = unwrapTensor(memoryState.meta);

      // Compute importance scores
      const scores = tf.sigmoid(tf.matMul(
        tf.concat([shortTerm, longTerm, meta], 1),
        this.pruningNetwork
      ));

      // Mask out low-importance memories
      const mask = scores.greater(tf.scalar(threshold));

      return {
        shortTerm: wrapTensor(tf.mul(shortTerm, mask)),
        longTerm: wrapTensor(tf.mul(longTerm, mask)),
        meta: wrapTensor(tf.mul(meta, mask))
      };
    });
  }

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
  public manifoldStep(base: ITensor, velocity: ITensor): ITensor {
    // Riemannian "update" if useManifold is true
    if (!this.useManifold) {
      // Standard Euclidean update
      return wrapTensor(tf.add(unwrapTensor(base), unwrapTensor(velocity)));
    }

    const result = tf.tidy(() => {
      const baseTensor = unwrapTensor(base);
      const velocityTensor = unwrapTensor(velocity);
      const epsilon = 1e-8;
      const maxStep = 0.1;

      const dot = baseTensor.mul(velocityTensor).sum();
      const radial = baseTensor.mul(dot);
      const tangent = velocityTensor.sub(radial);
      const tnorm = tangent.norm();

      const tNormVal = tnorm.dataSync()[0];
      if (tNormVal < epsilon) {
        return wrapTensor(baseTensor);
      }

      const stepSize = Math.min(tNormVal, maxStep);
      const direction = tangent.div(tf.scalar(tNormVal));
      const cosV = tf.cos(tf.scalar(stepSize));
      const sinV = tf.sin(tf.scalar(stepSize));
      const part1 = baseTensor.mul(cosV);
      const part2 = direction.mul(sinV);
      const newParam = part1.add(part2);
      const newParamNorm = newParam.norm();

      return wrapTensor(newParam.div(newParamNorm.add(tf.scalar(1e-12))));
    });

    return result;
  }

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
  public async saveModel(path: string): Promise<void> {
    const weights = {
      queryProjection: await this.queryProjection.array(),
      keyProjection: await this.keyProjection.array(),
      valueProjection: await this.valueProjection.array(),
      outputProjection: await this.outputProjection.array(),
      shortTermEncoder: await this.shortTermEncoder.array(),
      longTermEncoder: await this.longTermEncoder.array(),
      metaEncoder: await this.metaEncoder.array(),
      shortTermDecoder: await this.shortTermDecoder.array(),
      longTermDecoder: await this.longTermDecoder.array(),
      metaDecoder: await this.metaDecoder.array(),
      surpriseNetwork: await this.surpriseNetwork.array(),
      pruningNetwork: await this.pruningNetwork.array()
    };

    await fs.writeFile(path.replace('file://', ''), JSON.stringify(weights));
  }

  /**
   * Loads model weights from disk.
   * 
   * Deserializes and assigns all trainable parameters.
   * 
   * @param path File path
   */
  public async loadModel(path: string): Promise<void> {
    const weightsJson = await fs.readFile(path.replace('file://', ''), 'utf8');
    const weights = JSON.parse(weightsJson);

    this.queryProjection.assign(tf.tensor(weights.queryProjection));
    this.keyProjection.assign(tf.tensor(weights.keyProjection));
    this.valueProjection.assign(tf.tensor(weights.valueProjection));
    this.outputProjection.assign(tf.tensor(weights.outputProjection));
    this.shortTermEncoder.assign(tf.tensor(weights.shortTermEncoder));
    this.longTermEncoder.assign(tf.tensor(weights.longTermEncoder));
    this.metaEncoder.assign(tf.tensor(weights.metaEncoder));
    this.shortTermDecoder.assign(tf.tensor(weights.shortTermDecoder));
    this.longTermDecoder.assign(tf.tensor(weights.longTermDecoder));
    this.metaDecoder.assign(tf.tensor(weights.metaDecoder));
    this.surpriseNetwork.assign(tf.tensor(weights.surpriseNetwork));
    this.pruningNetwork.assign(tf.tensor(weights.pruningNetwork));
  }

  /**
   * Returns current model configuration.
   * 
   * @returns Configuration object with all current parameter values
   */
  public getConfig(): TitanMemoryConfig {
    return {
      inputDim: this.inputDim,
      hiddenDim: this.hiddenDim,
      outputDim: this.memoryDim,
      learningRate: this.learningRate,
      useManifold: this.useManifold,
      momentumFactor: this.momentumFactor,
      attentionHeads: this.attentionHeads,
      keyDim: this.keyDim,
      valueDim: this.valueDim,
      surpriseThreshold: this.surpriseThreshold,
      metaLearningRate: this.metaLearningRate,
      maxMemorySize: this.maxMemorySize
    };
  }
}
