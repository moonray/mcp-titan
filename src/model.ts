/**
 * @fileoverview Titans Memory Model Implementation
 */

import * as tf from '@tensorflow/tfjs';
import { ITensor, IMemoryModel, IMemoryState, ISurpriseMetrics, IAttentionBlock, IMemoryUpdateResult, IModelGradients, TensorContainer, unwrapTensor } from './types.js';
import * as fs from 'fs/promises';

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

export class TitanMemoryModel implements IMemoryModel {
  // Configuration properties
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

  // Derived dimensions
  private combinedContextSize: number;
  private surpriseInputSize: number;
  private pruningInputSize: number;

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

  // Optimization
  private optimizer: tf.Optimizer;
  private metaOptimizer: tf.Optimizer;

  constructor(config: TitanMemoryConfig = {}) {
    // Initialize configuration with defaults
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

    // Calculate derived dimensions
    this.combinedContextSize = this.hiddenDim * 3;
    this.surpriseInputSize = this.inputDim + this.hiddenDim;
    this.pruningInputSize = this.inputDim + this.memoryDim + this.hiddenDim;

    // Initialize parameters with Glorot initialization
    const glorot = (shape: number[]): tf.Tensor => {
      const [fanIn, fanOut] = shape;
      const scale = Math.sqrt(2.0 / (fanIn + fanOut));
      return tf.randomNormal(shape, 0, scale);
    };

    // Attention projections
    this.queryProjection = tf.variable(glorot([this.hiddenDim, this.attentionHeads * this.keyDim]), true, 'queryProjection');
    this.keyProjection = tf.variable(glorot([this.hiddenDim, this.attentionHeads * this.keyDim]), true, 'keyProjection');
    this.valueProjection = tf.variable(glorot([this.hiddenDim, this.attentionHeads * this.valueDim]), true, 'valueProjection');
    this.outputProjection = tf.variable(glorot([this.attentionHeads * this.valueDim, this.hiddenDim]), true, 'outputProjection');

    // Memory encoders
    this.shortTermEncoder = tf.variable(glorot([this.hiddenDim, this.inputDim]), true, 'shortTermEncoder');
    this.longTermEncoder = tf.variable(glorot([this.hiddenDim, this.memoryDim]), true, 'longTermEncoder');
    this.metaEncoder = tf.variable(glorot([this.hiddenDim, this.hiddenDim]), true, 'metaEncoder');

    // Memory decoders
    this.shortTermDecoder = tf.variable(glorot([this.combinedContextSize, this.inputDim]), true, 'shortTermDecoder');
    this.longTermDecoder = tf.variable(glorot([this.combinedContextSize, this.memoryDim]), true, 'longTermDecoder');
    this.metaDecoder = tf.variable(glorot([this.combinedContextSize, this.hiddenDim]), true, 'metaDecoder');

    // Surprise and pruning networks
    this.surpriseNetwork = tf.variable(glorot([this.surpriseInputSize, 2]), true, 'surpriseNetwork');
    this.pruningNetwork = tf.variable(glorot([this.pruningInputSize, 1]), true, 'pruningNetwork');

    // Optimizers
    this.optimizer = tf.train.adam(this.learningRate);
    this.metaOptimizer = tf.train.adam(this.metaLearningRate);
  }
  updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor {
    throw new Error('Method not implemented.');
  }
  manifoldStep(base: ITensor, velocity: ITensor): ITensor {
    throw new Error('Method not implemented.');
  }
  async save(modelPath: string, weightsPath: string): Promise<void> {
    // Ensure the model path has the correct extension
    const path = modelPath.endsWith('.json') ? modelPath : `${modelPath}.json`;

    // Use the existing saveModel implementation
    await this.saveModel(path);
  }

  private computeAttention(query: tf.Tensor, keys: tf.Tensor, values: tf.Tensor): IAttentionBlock {
    return tf.tidy(() => {
      // Split projections into heads
      const splitHeads = (tensor: tf.Tensor, dim: number) =>
        tensor.reshape([-1, this.attentionHeads, dim]).transpose([1, 0, 2]);

      const q = splitHeads(tf.matMul(query, this.queryProjection), this.keyDim);
      const k = splitHeads(tf.matMul(keys, this.keyProjection), this.keyDim);
      const v = splitHeads(tf.matMul(values, this.valueProjection), this.valueDim);

      // Compute attention scores
      const scores = tf.matMul(q, k.transpose([0, 2, 1]))
        .div(tf.sqrt(tf.scalar(this.keyDim, 'float32')));
      const attention = tf.softmax(scores, -1);

      // Combine heads and project
      const combined = tf.matMul(attention, v)
        .transpose([1, 0, 2])
        .reshape([-1, this.attentionHeads * this.valueDim]);
      const output = tf.matMul(combined, this.outputProjection);

      return { keys, values, scores: attention };
    });
  }

  private computeSurprise(predicted: tf.Tensor, actual: tf.Tensor, history: tf.Tensor): ISurpriseMetrics {
    return tf.tidy(() => {
      const diff = tf.sub(predicted, actual);
      const context = tf.concat([diff, history], 1);
      const surprise = tf.matMul(context, this.surpriseNetwork);
      const [immediate, accumulated] = tf.split(surprise, 2, 1);
      return { immediate, accumulated };
    });
  }

  public forward(x: ITensor, memoryState: IMemoryState): { predicted: ITensor; memoryUpdate: IMemoryUpdateResult } {
    return tf.tidy(() => {
      const input = unwrapTensor(x);
      const { shortTerm, longTerm, meta } = memoryState;

      // Encode inputs
      const encodedInput = tf.matMul(input, this.shortTermEncoder);
      const encodedShort = tf.matMul(shortTerm, this.shortTermEncoder);
      const encodedLong = tf.matMul(longTerm, this.longTermEncoder);
      const encodedMeta = tf.matMul(meta, this.metaEncoder);

      // Compute attention
      const attention = this.computeAttention(
        encodedInput,
        tf.concat([encodedShort, encodedLong], 0),
        tf.concat([shortTerm, longTerm], 0)
      );

      // Combine context
      const context = tf.concat([
        encodedInput,
        tf.matMul(attention.scores, attention.values),
        encodedMeta
      ], 1);

      // Generate predictions
      const predicted = tf.matMul(context, this.shortTermDecoder);

      // Compute surprise metrics
      const surprise = this.computeSurprise(predicted, input, encodedMeta);

      // Update memory states
      const newShortTerm = tf.matMul(context, this.shortTermDecoder);
      const newLongTerm = tf.matMul(context, this.longTermDecoder);
      const newMeta = tf.matMul(context, this.metaDecoder);

      return {
        predicted,
        memoryUpdate: {
          newState: {
            shortTerm: newShortTerm,
            longTerm: newLongTerm,
            meta: newMeta
          },
          attention,
          surprise
        }
      };
    });
  }

  public trainStep(
    x_t: ITensor,
    x_next: ITensor,
    memoryState: IMemoryState
  ): { loss: ITensor; gradients: IModelGradients } {
    return tf.tidy(() => {
      const xt = unwrapTensor(x_t);
      const xn = unwrapTensor(x_next);

      const { predicted, memoryUpdate } = this.forward(xt, memoryState);

      // Compute losses with explicit scalar casting
      const predLoss = tf.losses.meanSquaredError(xn, predicted).asScalar();
      const surpriseLoss = tf.add(
        tf.mul(tf.mean(tf.square(memoryUpdate.surprise.immediate)), 0.1).asScalar(),
        tf.mul(tf.mean(tf.square(memoryUpdate.surprise.accumulated)), 0.05).asScalar()
      );
      const totalLoss = tf.add(predLoss, surpriseLoss).asScalar();

      // Get gradients for all variables
      const { grads } = tf.variableGrads(() => totalLoss);

      // Convert to IModelGradients structure
      const gradients: IModelGradients = {
        shortTerm: grads['shortTermEncoder'] || tf.zeros(this.shortTermEncoder.shape),
        longTerm: grads['longTermEncoder'] || tf.zeros(this.longTermEncoder.shape),
        meta: grads['metaEncoder'] || tf.zeros(this.metaEncoder.shape)
      };

      // Apply gradients using proper format
      const gradArray = Object.entries(grads).map(([varName, grad]) => ({
        name: varName,
        tensor: grad
      }));

      this.optimizer.applyGradients(gradArray);

      return {
        loss: totalLoss,
        gradients
      };
    });
  }

  public pruneMemory(memoryState: IMemoryState): IMemoryState {
    return tf.tidy(() => {
      const { shortTerm, longTerm, meta } = memoryState;
      const combined = tf.concat([shortTerm, longTerm, meta], 1);
      const scores = tf.sigmoid(tf.matMul(combined, this.pruningNetwork)).flatten();

      // Maintain max memory size
      const { indices } = tf.topk(scores, this.maxMemorySize);
      return {
        shortTerm: tf.gather(shortTerm, indices),
        longTerm: tf.gather(longTerm, indices),
        meta: tf.gather(meta, indices)
      };
    });
  }

  public async saveModel(path: string): Promise<void> {
    // Get tensor data synchronously within tidy
    const tensors = tf.tidy(() => ({
      queryProjection: this.queryProjection,
      keyProjection: this.keyProjection,
      valueProjection: this.valueProjection,
      outputProjection: this.outputProjection,
      shortTermEncoder: this.shortTermEncoder,
      longTermEncoder: this.longTermEncoder,
      metaEncoder: this.metaEncoder,
      shortTermDecoder: this.shortTermDecoder,
      longTermDecoder: this.longTermDecoder,
      metaDecoder: this.metaDecoder,
      surpriseNetwork: this.surpriseNetwork,
      pruningNetwork: this.pruningNetwork
    }));

    // Convert tensors to arrays outside tidy
    const weights = {
      queryProjection: await tensors.queryProjection.array(),
      keyProjection: await tensors.keyProjection.array(),
      valueProjection: await tensors.valueProjection.array(),
      outputProjection: await tensors.outputProjection.array(),
      shortTermEncoder: await tensors.shortTermEncoder.array(),
      longTermEncoder: await tensors.longTermEncoder.array(),
      metaEncoder: await tensors.metaEncoder.array(),
      shortTermDecoder: await tensors.shortTermDecoder.array(),
      longTermDecoder: await tensors.longTermDecoder.array(),
      metaDecoder: await tensors.metaDecoder.array(),
      surpriseNetwork: await tensors.surpriseNetwork.array(),
      pruningNetwork: await tensors.pruningNetwork.array()
    };

    // Clean up temporary tensors
    Object.values(tensors).forEach(tensor => tensor.dispose());

    const modelData = {
      config: this.getConfig(),
      weights: weights,
      format_version: '1.0.0'
    };

    await fs.writeFile(path, JSON.stringify(modelData, null, 2));
  }

  public async loadModel(path: string): Promise<void> {
    const modelJson = await fs.readFile(path, 'utf8');
    const modelData = JSON.parse(modelJson);

    // Apply configuration
    const config = modelData.config as TitanMemoryConfig;
    Object.assign(this, {
      inputDim: config.inputDim || this.inputDim,
      hiddenDim: config.hiddenDim || this.hiddenDim,
      memoryDim: config.outputDim || this.memoryDim,
      learningRate: config.learningRate || this.learningRate,
      useManifold: config.useManifold || this.useManifold,
      momentumFactor: config.momentumFactor || this.momentumFactor,
      attentionHeads: config.attentionHeads || this.attentionHeads,
      keyDim: config.keyDim || this.keyDim,
      valueDim: config.valueDim || this.valueDim,
      surpriseThreshold: config.surpriseThreshold || this.surpriseThreshold,
      metaLearningRate: config.metaLearningRate || this.metaLearningRate,
      maxMemorySize: config.maxMemorySize || this.maxMemorySize
    });

    const assignWeights = (varName: string, tensor: tf.Tensor) => {
      const variable = (this as any)[varName] as tf.Variable;
      if (!variable) throw new Error(`Unknown variable: ${varName}`);
      if (!tensor.shape.every((v, i) => v === variable.shape[i])) {
        throw new Error(`Shape mismatch for ${varName}`);
      }
      variable.assign(tensor);
    };

    // Load weights
    Object.entries(modelData.weights).forEach(([name, data]) => {
      assignWeights(name, tf.tensor(data as tf.TensorLike));
    });
  }

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