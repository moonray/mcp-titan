/**
 * @fileovertitle Titan Memory Model 2.0 - Neural Memory Architecture with Transformer-XL Inspired Mechanisms
 */

import * as tf from '@tensorflow/tfjs-node';
import { ITensor, IMemoryState, ISurpriseMetrics, IAttentionBlock, IMemoryUpdateResult, IModelGradients, TensorContainer, unwrapTensor, wrapTensor, IMemoryModel, ITelemetryData, TensorError, MemoryError, IHierarchicalMemoryState, IExtendedMemoryState, IQuantizedMemoryState } from './types.js';
import * as fs from 'fs/promises';
import { z } from 'zod';
import { checkNullOrUndefined, validateTensor, validateTensorShape, SafeTensorOps } from './utils.js';

// Add telemetry implementation
class ModelTelemetry {
  private static instance: ModelTelemetry;
  private telemetryData: ITelemetryData[] = [];
  private maxEntries: number = 1000;
  private enabled: boolean = true;

  private constructor() { }

  public static getInstance(): ModelTelemetry {
    if (!ModelTelemetry.instance) {
      ModelTelemetry.instance = new ModelTelemetry();
    }
    return ModelTelemetry.instance;
  }

  public recordOperation(operation: string, metrics?: Record<string, number>): () => void {
    if (!this.enabled) return () => { };

    const startTime = performance.now();
    const startMemory = tf.memory();

    return () => {
      const endTime = performance.now();
      const endMemory = tf.memory();

      const telemetryEntry: ITelemetryData = {
        timestamp: Date.now(),
        operation,
        durationMs: endTime - startTime,
        memoryUsage: {
          numTensors: endMemory.numTensors,
          numBytes: endMemory.numBytes,
          unreliable: endMemory.unreliable
        },
        metrics
      };

      this.telemetryData.push(telemetryEntry);

      // Trim if needed
      if (this.telemetryData.length > this.maxEntries) {
        this.telemetryData = this.telemetryData.slice(-this.maxEntries);
      }
    };
  }

  public recordError(operation: string, error: Error): void {
    if (!this.enabled) return;

    const telemetryEntry: ITelemetryData = {
      timestamp: Date.now(),
      operation,
      durationMs: 0,
      memoryUsage: tf.memory(),
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack
      }
    };

    this.telemetryData.push(telemetryEntry);

    // Trim if needed
    if (this.telemetryData.length > this.maxEntries) {
      this.telemetryData = this.telemetryData.slice(-this.maxEntries);
    }
  }

  public getMetrics(): ITelemetryData[] {
    return [...this.telemetryData];
  }

  public getAverageMetrics(operation: string, lastN: number = 10): Record<string, number> {
    const relevantEntries = this.telemetryData
      .filter(entry => entry.operation === operation)
      .slice(-lastN);

    if (relevantEntries.length === 0) {
      return {};
    }

    const avgDuration = relevantEntries.reduce((sum, entry) => sum + entry.durationMs, 0) / relevantEntries.length;
    const avgTensors = relevantEntries.reduce((sum, entry) => sum + entry.memoryUsage.numTensors, 0) / relevantEntries.length;
    const avgBytes = relevantEntries.reduce((sum, entry) => sum + entry.memoryUsage.numBytes, 0) / relevantEntries.length;

    const result: Record<string, number> = {
      avgDurationMs: avgDuration,
      avgTensors: avgTensors,
      avgBytes: avgBytes
    };

    // Add custom metrics if they exist
    if (relevantEntries[0].metrics) {
      Object.keys(relevantEntries[0].metrics!).forEach(metricKey => {
        result[`avg${metricKey}`] = relevantEntries.reduce(
          (sum, entry) => sum + (entry.metrics?.[metricKey] || 0),
          0
        ) / relevantEntries.length;
      });
    }

    return result;
  }

  public enable(): void {
    this.enabled = true;
  }

  public disable(): void {
    this.enabled = false;
  }

  public clear(): void {
    this.telemetryData = [];
  }
}

// Add polyfill for isNullOrUndefined
const isNullOrUndefined = (value: any): value is null | undefined => value === null || value === undefined;

// Patch TensorFlow.js Node backend
const originalCreateTensorsTypeOpAttr = (tf as any).backend().createTensorsTypeOpAttr;
if (originalCreateTensorsTypeOpAttr) {
  (tf as any).backend().createTensorsTypeOpAttr = function (...args: any[]) {
    // Replace any usage of isNullOrUndefined with our polyfill
    const patchedArgs = args.map(arg => {
      if (typeof arg === 'function' && arg.name === 'isNullOrUndefined') {
        return isNullOrUndefined;
      }
      return arg;
    });
    return originalCreateTensorsTypeOpAttr.apply(this, patchedArgs);
  };
}

// Enhanced configuration schema
const ModelConfigSchema = z.object({
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
  learningRate: z.number().positive().default(0.001),
  vocabSize: z.number().int().positive().default(50000),
  decayRate: z.number().min(0).max(1).default(0.9),
  useRotaryEmbeddings: z.boolean().default(false),
  useMultiQueryAttention: z.boolean().default(false),
  useHierarchicalMemory: z.boolean().default(false),
  useSubwordTokenization: z.boolean().default(false),
  useApproximateNearestNeighbors: z.boolean().default(false),
  useGatedLinearUnits: z.boolean().default(false),
  useSwiGLU: z.boolean().default(false),
  useMemoryDistillation: z.boolean().default(false),
  enableQuantization: z.boolean().default(false),
  enableContrastiveLearning: z.boolean().default(false),
  enableAdaptiveComputationTime: z.boolean().default(false),
  enableInformationGainPruning: z.boolean().default(false),
  enableEpisodicSemanticDistinction: z.boolean().default(false),
  enableJITCompilation: z.boolean().default(false),
  enableSparseAttention: z.boolean().default(false),
  sparsityRate: z.number().min(0).max(0.99).default(0.8),
  enableTelemetry: z.boolean().default(true),
  actConfig: z.object({
    maxPonderSteps: z.number().int().positive().default(10),
    ponderCost: z.number().min(0).max(1).default(0.01)
  }).optional().default({}),
  contrastiveWeight: z.number().min(0).max(1).default(0.1)
});

type TitanMemoryConfig = z.infer<typeof ModelConfigSchema>;

interface WeightInfo {
  shape: number[];
  dtype: string;
}

// Add this near the top of the file, after imports but before class definitions
/**
 * Safe logging function that won't interfere with MCP communication
 * @param message The message to log
 */
function safeLog(message: string): void {
  // Check if we're in an MCP context
  const isMcpContext = process.env.MCP_CONTEXT === 'true';

  if (!isMcpContext) {
    console.log(message);
  }
  // In MCP context, we don't log to console to avoid interfering with JSON communication
}

export class TitanMemoryModel implements IMemoryModel {
  private config: TitanMemoryConfig = ModelConfigSchema.parse({});
  private transformerStack: tf.LayersModel[] = [];
  private memoryProjector!: tf.LayersModel;
  private similarityNetwork!: tf.LayersModel;
  private optimizer!: tf.Optimizer;
  private stepCount = 0;
  private vocabulary: Map<string, number> = new Map();
  private reverseVocabulary: Map<number, string> = new Map();

  // Enhanced memory state with temporal dynamics
  private memoryState: IMemoryState = {
    shortTerm: tf.zeros([0]),
    longTerm: tf.zeros([0]),
    meta: tf.zeros([0]),
    timestamps: tf.zeros([0]),
    accessCounts: tf.zeros([0]),
    surpriseHistory: tf.zeros([0])
  };

  // Add hierarchical memory properties
  private hierarchicalLevels: number = 3;
  private hierarchicalMemory: IHierarchicalMemoryState | null = null;

  // Add quantization properties
  private quantizedMemory: IQuantizedMemoryState | null = null;
  private quantizationBits: number = 8;
  private quantizationRanges: { min: number; max: number }[] = [];

  // Add contrastive learning properties
  private contrastiveBuffer: tf.Tensor[] = [];
  private contrastiveBufferSize: number = 128;
  private contrastiveTemperature: number = 0.07;

  // Add encoder and decoder properties
  private encoder: tf.LayersModel;
  private decoder: tf.LayersModel;
  private tokenizer: any = null;
  private vocabSize: number = 10000;

  // Add error handling wrapper
  private withErrorHandling<T>(operation: string, fn: () => T): T {
    const telemetry = ModelTelemetry.getInstance();
    const endTelemetry = telemetry.recordOperation(operation);

    try {
      return fn();
    } catch (error) {
      console.error(`Error in operation ${operation}:`, error);

      // Log to telemetry
      telemetry.recordError(operation, error);

      // Attempt recovery based on error type
      if (error instanceof TensorError) {
        this.resetGradients();
        console.log(`Recovered from tensor error in ${operation} by resetting gradients`);
      } else if (error instanceof MemoryError) {
        this.initializeMemoryState();
        console.log(`Recovered from memory error in ${operation} by reinitializing memory state`);
      }

      throw error;
    } finally {
      endTelemetry();
    }
  }

  constructor(config?: Partial<TitanMemoryConfig>) {
    // Initialize with empty config first
    this.config = ModelConfigSchema.parse(config || {});
    // Don't initialize components yet - wait for backend
  }

  private async initializeBackend(): Promise<void> {
    try {
      // Ensure TensorFlow.js is properly initialized
      await tf.ready();

      // Set the backend explicitly
      await tf.setBackend('tensorflow');

      // Double check backend is set and ready
      const backend = tf.getBackend();
      if (!backend) {
        throw new Error('TensorFlow backend not initialized');
      }

      // Initialize components after backend is ready
      this.initializeComponents();
      this.initializeMemoryState();

      console.log('TensorFlow backend initialized:', backend);
    } catch (error) {
      console.error('Error initializing TensorFlow backend:', error);
      throw error;
    }
  }

  private initializeVocabulary(): void {
    // Initialize with special tokens
    this.vocabulary.clear();
    this.reverseVocabulary.clear();

    this.vocabulary.set('[PAD]', 0);
    this.vocabulary.set('[UNK]', 1);
    this.vocabulary.set('[CLS]', 2);
    this.vocabulary.set('[SEP]', 3);

    // Add basic characters and common tokens
    const basicChars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_\'"`()[]{}:;/\\+=<>'.split('');
    basicChars.forEach((char, i) => {
      this.vocabulary.set(char, i + 4);
    });

    // Create reverse mapping
    this.vocabulary.forEach((value, key) => {
      this.reverseVocabulary.set(value, key);
    });
  }

  /**
   * Creates encoder model for processing inputs
   */
  private createEncoder(): tf.LayersModel {
    return this.withErrorHandling('createEncoder', () => {
      const inputShape = [this.config.inputDim];
      const embeddingSize = this.config.embeddingSize;

      // Create sequential model
      const model = tf.sequential();

      // Add layers
      model.add(tf.layers.dense({
        inputShape,
        units: embeddingSize * 2,
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
      }));

      // Add dropout for regularization
      model.add(tf.layers.dropout({ rate: 0.2 }));

      // Add final embedding layer
      model.add(tf.layers.dense({
        units: embeddingSize,
        activation: 'tanh',
        kernelInitializer: 'glorotNormal'
      }));

      return model;
    });
  }

  /**
   * Creates decoder model for generating outputs
   */
  private createDecoder(): tf.LayersModel {
    return this.withErrorHandling('createDecoder', () => {
      // Input is concatenated embedding and memory
      const inputShape = [this.config.embeddingSize * 2];
      const outputDim = this.config.inputDim;

      // Create sequential model
      const model = tf.sequential();

      // Add layers
      model.add(tf.layers.dense({
        inputShape,
        units: this.config.embeddingSize * 2,
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
      }));

      // Add dropout for regularization
      model.add(tf.layers.dropout({ rate: 0.2 }));

      // Add final output layer
      model.add(tf.layers.dense({
        units: outputDim,
        activation: 'linear',
        kernelInitializer: 'glorotNormal'
      }));

      return model;
    });
  }

  /**
   * Encodes text input to tensor
   * @param text The text to encode
   * @returns The encoded tensor
   */
  private async encodeText(text: string): Promise<tf.Tensor> {
    return this.withErrorHandling('encodeText', async () => {
      if (!this.tokenizer) {
        throw new Error('Tokenizer not initialized');
      }

      // Tokenize the text
      const tokens = this.tokenizer.encode(text);

      // Convert to one-hot encoding
      const oneHot = tf.oneHot(
        tf.tensor1d(tokens, 'int32'),
        this.vocabSize
      );

      // Average the embeddings if sequence is longer than 1
      if (oneHot.shape[0] > 1) {
        return tf.mean(oneHot, 0);
      }

      return oneHot.reshape([this.vocabSize]);
    });
  }

  /**
   * Initializes the model
   * @param config Configuration options
   */
  public async initialize(config?: Partial<TitanMemoryConfig>): Promise<void> {
    return this.withErrorHandling('initialize', async () => {
      // Update config with provided options
      if (config) {
        this.config = { ...this.config, ...config };
      }

      // Create encoder and decoder
      this.encoder = this.createEncoder();
      this.decoder = this.createDecoder();

      // Initialize optimizer
      const learningRate = this.config.learningRate || 0.001;
      this.optimizer = tf.train.adam(learningRate);

      // Initialize memory state
      this.initializeMemoryState();

      // Initialize tokenizer if text processing is needed
      if (this.config.enableTextProcessing) {
        // This is a placeholder - in a real implementation,
        // you would load a proper tokenizer here
        this.tokenizer = {
          encode: (text: string) => {
            return Array.from(text).map(c => c.charCodeAt(0) % this.vocabSize);
          },
          decode: (tokens: number[]) => {
            return tokens.map(t => String.fromCharCode(t)).join('');
          }
        };
      }

      console.log('Model initialized successfully');
    });
  }

  /**
   * Retrieves memory based on query
   */
  private retrieveFromMemory(query: ITensorWrapper): ITensorWrapper {
    return this.withErrorHandling('retrieveFromMemory', () => {
      // Calculate similarity between query and all memories
      const similarities = tf.matMul(
        this.memoryState.shortTerm,
        unwrapTensor(query).reshape([unwrapTensor(query).shape[0], 1]),
        false,
        true
      );

      // Apply softmax to get attention weights
      const attentionWeights = tf.softmax(similarities);

      // Update access counts
      const newAccessCounts = this.memoryState.accessCounts.add(attentionWeights);
      tf.dispose(this.memoryState.accessCounts);
      this.memoryState.accessCounts = newAccessCounts;

      // Weight memories by attention
      const weightedMemory = tf.matMul(
        attentionWeights,
        this.memoryState.shortTerm,
        true,
        false
      );

      // Clean up
      tf.dispose(similarities);
      tf.dispose(attentionWeights);

      return wrapTensor(weightedMemory);
    });
  }

  private initializeComponents(): void {
    // Initialize transformer stack
    this.transformerStack = [];
    for (let i = 0; i < this.config.transformerLayers; i++) {
      const layer = tf.sequential({
        layers: [
          tf.layers.dense({
            units: this.config.hiddenDim,
            inputShape: [this.config.inputDim],
            activation: 'linear',
            useBias: true,
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
          }),
          tf.layers.layerNormalization(),
          tf.layers.dense({
            units: this.config.ffDimension,
            activation: 'elu',
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
          }),
          tf.layers.dropout({ rate: this.config.dropoutRate }),
          tf.layers.dense({
            units: this.config.hiddenDim,
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
          }),
          tf.layers.layerNormalization()
        ]
      });
      this.transformerStack.push(layer);
    }

    // Initialize memory projector
    this.memoryProjector = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.memoryDim,
          inputShape: [this.config.hiddenDim],
          activation: 'tanh',
          kernelInitializer: 'glorotNormal',
          biasInitializer: 'zeros'
        }),
        tf.layers.layerNormalization()
      ]
    });

    // Initialize similarity network
    this.similarityNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.hiddenDim,
          inputShape: [this.config.memoryDim],
          activation: 'relu',
          kernelInitializer: 'glorotNormal',
          biasInitializer: 'zeros'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid',
          kernelInitializer: 'glorotNormal',
          biasInitializer: 'zeros'
        })
      ]
    });

    // Initialize optimizer
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  private initializeMemoryState(): void {
    tf.tidy(() => {
      const memorySlots = this.config.memorySlots;
      const embeddingSize = this.config.memoryDim;

      // Initialize standard memory components
      this.memoryState = {
        shortTerm: tf.zeros([memorySlots, embeddingSize]),
        longTerm: tf.zeros([Math.floor(memorySlots / 2), embeddingSize]),
        meta: tf.zeros([memorySlots, 5]), // metadata features per memory slot
        timestamps: tf.zeros([memorySlots]),
        accessCounts: tf.zeros([memorySlots]),
        surpriseHistory: tf.zeros([100]) // track last 100 surprise scores
      };

      // Initialize hierarchical memory if enabled
      if (this.config.useHierarchicalMemory) {
        this.initializeHierarchicalMemory();
      }

      // Initialize quantization if enabled
      if (this.config.enableQuantization) {
        this.initializeQuantization();
      }

      console.log(`Memory initialized with ${memorySlots} slots and ${embeddingSize} dimensions`);
    });
  }

  private validateMemoryState(state: IMemoryState): boolean {
    return tf.tidy(() => {
      try {
        const validations = [
          state.shortTerm && !state.shortTerm.isDisposed,
          state.longTerm && !state.longTerm.isDisposed,
          state.meta && !state.meta.isDisposed,
          state.timestamps && !state.timestamps.isDisposed,
          state.accessCounts && !state.accessCounts.isDisposed,
          state.surpriseHistory && !state.surpriseHistory.isDisposed
        ];

        return validations.every(Boolean);
      } catch (error) {
        console.warn('Error validating memory state:', error);
        return false;
      }
    });
  }

  public async storeMemory(text: string): Promise<void> {
    const embedding = await this.encodeText(text);
    const similarity = this.calculateSimilarity(embedding);

    const { values, indices } = tf.topk(similarity, 1);
    if (values.dataSync()[0] < this.config.similarityThreshold) {
      this.addMemoryEntry(embedding);
    }

    this.updateAccessStats(indices);
    this.checkPruning();
  }

  private calculateSimilarity(embedding: tf.Tensor1D): tf.Tensor1D {
    return tf.tidy(() => {
      const expanded = embedding.reshape([1, -1]);
      return tf.matMul(this.memoryState.shortTerm, expanded)
        .div(tf.norm(this.memoryState.shortTerm, 2, 1).mul(tf.norm(expanded)))
        .squeeze() as tf.Tensor1D;
    });
  }

  private addMemoryEntry(embedding: tf.Tensor1D): void {
    tf.tidy(() => {
      const newMemory = tf.concat([
        this.memoryState.shortTerm,
        embedding.reshape([1, -1])
      ], 0).slice(0, this.config.memorySlots);

      this.memoryState.shortTerm.dispose();
      this.memoryState.shortTerm = newMemory as tf.Tensor2D;
    });
  }

  private updateAccessStats(indices: tf.Tensor1D): void {
    tf.tidy(() => {
      const updates = tf.onesLike(indices);
      this.memoryState.accessCounts = tf.add(
        this.memoryState.accessCounts,
        tf.scatterND(indices.reshape([-1, 1]), updates, [this.config.memorySlots])
      );
    });
  }

  private checkPruning(): void {
    this.stepCount++;
    if (this.stepCount % this.config.pruningInterval === 0) {
      this.pruneMemory(this.memoryState, this.config.similarityThreshold);
    }
  }

  public pruneMemory(memoryState: IMemoryState, threshold: number): IMemoryState {
    return tf.tidy(() => {
      const relevance = this.computeMemoryRelevance();
      const { indices } = tf.topk(relevance, this.config.memorySlots);

      return {
        shortTerm: tf.gather(memoryState.shortTerm, indices) as tf.Tensor2D,
        longTerm: tf.gather(memoryState.longTerm, indices) as tf.Tensor2D,
        meta: tf.gather(memoryState.meta, indices) as tf.Tensor2D,
        timestamps: tf.gather(memoryState.timestamps, indices) as tf.Tensor1D,
        accessCounts: tf.gather(memoryState.accessCounts, indices) as tf.Tensor1D,
        surpriseHistory: tf.gather(memoryState.surpriseHistory, indices) as tf.Tensor1D
      };
    });
  }

  private computeMemoryRelevance(): tf.Tensor1D {
    return tf.tidy(() => {
      const recency = tf.sub(tf.scalar(Date.now()), this.memoryState.timestamps);
      const frequency = tf.log(tf.add(this.memoryState.accessCounts, 1));
      const surprise = tf.mul(
        this.memoryState.surpriseHistory,
        this.config.surpriseDecay
      );

      return tf.addN([recency, frequency, surprise]) as tf.Tensor1D;
    });
  }

  public async recallMemory(query: string, topK = 5): Promise<tf.Tensor2D[]> {
    const queryEmbedding = await this.encodeText(query);
    const similarities = this.calculateSimilarity(queryEmbedding);

    const { indices } = tf.topk(similarities, topK);
    return indices.arraySync().map(i =>
      this.memoryState.shortTerm.slice([i, 0], [1, -1]) as tf.Tensor2D
    );
  }

  /**
   * Forward pass with hierarchical memory support
   */
  public forward(input: ITensorWrapper, state?: IMemoryState): {
    predicted: ITensorWrapper;
    memoryUpdate: {
      newState: IMemoryState;
      surprise: ITensorWrapper;
    };
  } {
    return tf.tidy(() => {
      // Use provided state or current state
      const memoryState = state || this.memoryState;

      // Process input through the model
      const inputTensor = unwrapTensor(input);
      const encodedInput = this.encoder(inputTensor);

      // Retrieve from memory - use hierarchical if enabled
      const memoryResult = this.config.useHierarchicalMemory
        ? this.retrieveFromHierarchicalMemory(wrapTensor(encodedInput))
        : this.retrieveFromMemory(wrapTensor(encodedInput));

      // Combine input with memory
      const combined = tf.concat([encodedInput, unwrapTensor(memoryResult)], 1);

      // Process through decoder
      const decoded = this.decoder(combined);

      // Calculate surprise (prediction error)
      const surprise = tf.sub(decoded, inputTensor);
      const surpriseMagnitude = tf.norm(surprise);

      // Update memory
      const newMemoryState = this.updateMemory(
        wrapTensor(encodedInput),
        wrapTensor(surpriseMagnitude),
        memoryState
      );

      // Update hierarchical memory if enabled
      if (this.config.useHierarchicalMemory) {
        this.updateHierarchicalMemory(
          wrapTensor(encodedInput),
          wrapTensor(surpriseMagnitude)
        );
      }

      // Increment step counter
      this.stepCount++;

      return {
        predicted: wrapTensor(decoded),
        memoryUpdate: {
          newState: newMemoryState,
          surprise: wrapTensor(surpriseMagnitude)
        }
      };
    });
  }

  private computeMemoryAttention(query: tf.Tensor2D): IAttentionBlock {
    return tf.tidy(() => {
      const weights = this.similarityNetwork.getWeights();
      const keys = SafeTensorOps.matMul(this.memoryState.shortTerm, weights[0] as tf.Tensor2D);
      const values = SafeTensorOps.matMul(this.memoryState.shortTerm, weights[1] as tf.Tensor2D);

      const scores = tf.softmax(SafeTensorOps.matMul(query, keys.transpose()));
      const attended = SafeTensorOps.matMul(scores, values);

      return {
        keys,
        values: attended,
        scores
      };
    });
  }

  private computeSurprise(input: tf.Tensor2D, expected: tf.Tensor2D): ISurpriseMetrics {
    return tf.tidy(() => {
      const error = SafeTensorOps.sub(input, expected);
      const immediate = tf.mean(tf.square(error), 1);
      const decayTensor = tf.scalar(this.config.surpriseDecay);
      const accumulated = SafeTensorOps.add(
        SafeTensorOps.mul(this.memoryState.surpriseHistory, decayTensor),
        immediate
      );

      return { immediate, accumulated };
    });
  }

  /**
   * Implements contrastive learning to improve embedding space
   * @param anchor The anchor embedding
   * @param positive The positive example (similar to anchor)
   * @returns The contrastive loss
   */
  private contrastiveLearning(anchor: ITensorWrapper, positive: ITensorWrapper): ITensorWrapper {
    if (!this.config.enableContrastiveLearning) {
      return wrapTensor(tf.scalar(0.0));
    }

    return this.withErrorHandling('contrastiveLearning', () => {
      // Normalize embeddings to unit length
      const anchorNorm = tf.div(
        unwrapTensor(anchor),
        tf.norm(unwrapTensor(anchor))
      );

      const positiveNorm = tf.div(
        unwrapTensor(positive),
        tf.norm(unwrapTensor(positive))
      );

      // Add to contrastive buffer if not full
      if (this.contrastiveBuffer.length < this.contrastiveBufferSize) {
        this.contrastiveBuffer.push(anchorNorm.clone());
      } else {
        // Replace random item in buffer
        const replaceIndex = Math.floor(Math.random() * this.contrastiveBufferSize);
        tf.dispose(this.contrastiveBuffer[replaceIndex]);
        this.contrastiveBuffer[replaceIndex] = anchorNorm.clone();
      }

      // Need at least 8 samples for meaningful contrastive learning
      if (this.contrastiveBuffer.length < 8) {
        return wrapTensor(tf.scalar(0.0));
      }

      // Compute similarity between anchor and positive example
      const positiveSimilarity = tf.sum(tf.mul(anchorNorm, positiveNorm));

      // Compute similarities with negative examples from buffer
      const negativeSimilarities = this.contrastiveBuffer.map(negative => {
        return tf.sum(tf.mul(anchorNorm, negative));
      });

      // Concatenate positive and negative similarities
      const allSimilarities = tf.concat([
        positiveSimilarity.reshape([1]),
        tf.stack(negativeSimilarities)
      ]);

      // Scale by temperature
      const scaledSimilarities = tf.div(
        allSimilarities,
        tf.scalar(this.contrastiveTemperature)
      );

      // Compute softmax
      const softmaxSimilarities = tf.softmax(scaledSimilarities);

      // Contrastive loss is negative log likelihood of positive example
      const loss = tf.neg(tf.log(softmaxSimilarities.gather([0])));

      // Clean up
      tf.dispose([
        anchorNorm,
        positiveNorm,
        positiveSimilarity,
        allSimilarities,
        scaledSimilarities,
        softmaxSimilarities
      ]);

      return wrapTensor(loss);
    });
  }

  /**
   * Enhanced training step with contrastive learning
   */
  public trainStep(
    currentInput: ITensorWrapper,
    nextInput: ITensorWrapper,
    state: IMemoryState
  ): {
    loss: ITensorWrapper;
    gradients: IModelGradients;
  } {
    return this.withErrorHandling('trainStep', () => {
      const { predicted, memoryUpdate } = this.forward(currentInput, state);

      // Calculate prediction loss
      const predictionLoss = tf.losses.meanSquaredError(
        unwrapTensor(nextInput),
        unwrapTensor(predicted)
      );

      // Calculate contrastive loss if enabled
      let contrastiveLoss = tf.scalar(0.0);
      if (this.config.enableContrastiveLearning) {
        // Use current and next inputs as positive pairs
        const currentEncoded = this.encoder(unwrapTensor(currentInput));
        const nextEncoded = this.encoder(unwrapTensor(nextInput));

        contrastiveLoss = unwrapTensor(
          this.contrastiveLearning(
            wrapTensor(currentEncoded),
            wrapTensor(nextEncoded)
          )
        );
      }

      // Combine losses
      const contrastiveWeight = this.config.contrastiveWeight || 0.1;
      const combinedLoss = tf.add(
        predictionLoss,
        tf.mul(contrastiveLoss, tf.scalar(contrastiveWeight))
      );

      // Compute gradients
      const gradients = this.optimizer.computeGradients(() => combinedLoss);

      // Apply gradients
      this.optimizer.applyGradients(gradients.grads);

      // Increment step counter
      this.stepCount++;

      // Clean up
      tf.dispose([predictionLoss, contrastiveLoss]);

      return {
        loss: wrapTensor(combinedLoss),
        gradients: {
          encoder: gradients.grads['encoder'] as tf.Tensor,
          decoder: gradients.grads['decoder'] as tf.Tensor
        }
      };
    });
  }

  public updateMetaMemory(surprise: ISurpriseMetrics, context: ITensor): ITensor {
    return tf.tidy(() => {
      const surpriseGate = tf.sigmoid(surprise.immediate);
      return tf.add(
        tf.mul(this.memoryState.meta, tf.sub(1, surpriseGate)),
        tf.mul(context, surpriseGate)
      );
    });
  }

  public manifoldStep(base: ITensor, velocity: ITensor): ITensor {
    return tf.tidy(() => {
      const norm = tf.norm(velocity);
      const normalized = tf.div(velocity, norm);
      return tf.add(base, tf.mul(normalized, this.config.learningRate));
    });
  }

  public getConfig(): TitanMemoryConfig {
    return { ...this.config };
  }

  /**
   * Saves the model to disk with proper versioning and error handling
   * @param path The path to save the model to
   */
  public async save(path: string): Promise<void> {
    return this.withErrorHandling('save', async () => {
      try {
        // Create directory if it doesn't exist
        const dir = path.split('/').slice(0, -1).join('/');
        await fs.mkdir(dir, { recursive: true });

        // Define model version and format
        const modelMetadata = {
          version: "1.0",
          format: "titan-memory-v1",
          created: new Date().toISOString(),
          config: this.config
        };

        // Save encoder and decoder models
        const encoderPath = `${path}/encoder`;
        const decoderPath = `${path}/decoder`;

        await this.encoder.save(`file://${encoderPath}`);
        await this.decoder.save(`file://${decoderPath}`);

        console.log('Saved encoder and decoder models');

        // Save memory state
        const memoryData = {
          shortTerm: this.memoryState.shortTerm.arraySync(),
          longTerm: this.memoryState.longTerm.arraySync(),
          meta: this.memoryState.meta.arraySync(),
          timestamps: Array.from(this.memoryState.timestamps.dataSync()),
          accessCounts: Array.from(this.memoryState.accessCounts.dataSync()),
          surpriseHistory: Array.from(this.memoryState.surpriseHistory.dataSync())
        };

        // Save hierarchical memory if enabled
        let hierarchicalData = null;
        if (this.config.useHierarchicalMemory && this.hierarchicalMemory) {
          hierarchicalData = {
            levels: this.hierarchicalMemory.levels.map(level => level.arraySync()),
            timestamps: this.hierarchicalMemory.timestamps.map(ts => Array.from(ts.dataSync())),
            accessCounts: this.hierarchicalMemory.accessCounts.map(ac => Array.from(ac.dataSync())),
            surpriseScores: this.hierarchicalMemory.surpriseScores.map(ss => Array.from(ss.dataSync()))
          };
        }

        // Save quantization data if enabled
        let quantizationData = null;
        if (this.config.enableQuantization && this.quantizedMemory) {
          quantizationData = {
            ranges: this.quantizationRanges,
            bits: this.quantizationBits
          };
        }

        // Save telemetry data
        const telemetry = ModelTelemetry.getInstance();
        const telemetryData = telemetry.getAllMetrics();

        // Create complete model data
        const modelData = {
          ...modelMetadata,
          encoderPath,
          decoderPath,
          memoryState: memoryData,
          hierarchicalMemory: hierarchicalData,
          quantization: quantizationData,
          telemetry: telemetryData
        };

        // Save model data as JSON
        const modelPath = `${path}/model.json`;
        await fs.writeFile(modelPath, JSON.stringify(modelData, null, 2));

        console.log(`Model saved to ${path}`);
      } catch (error) {
        console.error('Error saving model:', error);
        throw new MemoryError('Failed to save model: ' + error.message);
      }
    });
  }

  /**
   * Loads the model from disk with proper error handling
   * @param path The path to load the model from
   */
  public async load(path: string): Promise<void> {
    return this.withErrorHandling('load', async () => {
      try {
        // Check if model.json exists (new format)
        const modelPath = `${path}/model.json`;
        let modelData;

        try {
          const modelJson = await fs.readFile(modelPath, 'utf-8');
          modelData = JSON.parse(modelJson);
          safeLog('Found model.json, loading in new format');
        } catch (error) {
          safeLog('No model.json found, trying legacy format');
          await this.loadLegacyFormat(path);
          return;
        }

        // Validate model format
        if (!modelData.format || modelData.format !== 'titan-memory-v1') {
          if (!isMcpContext) {
            console.warn(`Unknown model format: ${modelData.format || 'undefined'}, attempting to load anyway`);
          }
        }

        // Load configuration
        if (modelData.config) {
          this.config = { ...this.config, ...modelData.config };
          safeLog('Loaded model configuration');
        }

        // Load encoder and decoder models
        try {
          if (modelData.encoderPath && modelData.decoderPath) {
            this.encoder = await tf.loadLayersModel(`file://${modelData.encoderPath}/model.json`);
            this.decoder = await tf.loadLayersModel(`file://${modelData.decoderPath}/model.json`);
            safeLog('Loaded encoder and decoder models');
          } else {
            throw new Error('Missing encoder or decoder paths in model data');
          }
        } catch (error) {
          if (!isMcpContext) {
            console.error('Error loading encoder/decoder models:', error);
          }
          throw new MemoryError('Failed to load encoder/decoder models: ' + error.message);
        }

        // Initialize optimizer
        const learningRate = this.config.learningRate || 0.001;
        this.optimizer = tf.train.adam(learningRate);

        // Load memory state
        if (modelData.memoryState) {
          try {
            // Dispose existing memory state
            if (this.memoryState) {
              Object.values(this.memoryState).forEach(tensor => {
                if (tensor && !tensor.isDisposed) {
                  tensor.dispose();
                }
              });
            }

            // Create new memory state from saved data
            this.memoryState = {
              shortTerm: tf.tensor(modelData.memoryState.shortTerm),
              longTerm: tf.tensor(modelData.memoryState.longTerm),
              meta: tf.tensor(modelData.memoryState.meta),
              timestamps: tf.tensor1d(modelData.memoryState.timestamps),
              accessCounts: tf.tensor1d(modelData.memoryState.accessCounts),
              surpriseHistory: tf.tensor1d(modelData.memoryState.surpriseHistory)
            };
            safeLog('Loaded memory state');
          } catch (error) {
            if (!isMcpContext) {
              console.error('Error loading memory state:', error);
            }
            throw new MemoryError('Failed to load memory state: ' + error.message);
          }
        } else {
          if (!isMcpContext) {
            console.warn('No memory state found in model data, initializing new memory state');
          }
          this.initializeMemoryState();
        }

        // Load hierarchical memory if available
        if (modelData.hierarchicalMemory && this.config.useHierarchicalMemory) {
          try {
            const hierarchicalData = modelData.hierarchicalMemory;

            this.hierarchicalMemory = {
              levels: hierarchicalData.levels.map((level: number[][]) => tf.tensor(level)),
              timestamps: hierarchicalData.timestamps.map((ts: number[]) => tf.tensor1d(ts)),
              accessCounts: hierarchicalData.accessCounts.map((ac: number[]) => tf.tensor1d(ac)),
              surpriseScores: hierarchicalData.surpriseScores.map((ss: number[]) => tf.tensor1d(ss))
            };
            safeLog('Loaded hierarchical memory');
          } catch (error) {
            if (!isMcpContext) {
              console.error('Error loading hierarchical memory:', error);
            }
            this.hierarchicalMemory = null;

            // Re-initialize hierarchical memory
            if (this.config.useHierarchicalMemory) {
              this.initializeHierarchicalMemory();
            }
          }
        } else if (this.config.useHierarchicalMemory) {
          if (!isMcpContext) {
            console.warn('No hierarchical memory found but enabled in config, initializing new hierarchical memory');
          }
          this.initializeHierarchicalMemory();
        }

        // Load quantization data if available
        if (modelData.quantization && this.config.enableQuantization) {
          try {
            this.quantizationRanges = modelData.quantization.ranges;
            this.quantizationBits = modelData.quantization.bits;

            // Initialize quantized memory
            this.initializeQuantization();
            safeLog('Loaded quantization data');
          } catch (error) {
            if (!isMcpContext) {
              console.error('Error loading quantization data:', error);
            }
            this.quantizedMemory = null;

            // Re-initialize quantization
            if (this.config.enableQuantization) {
              this.initializeQuantization();
            }
          }
        } else if (this.config.enableQuantization) {
          if (!isMcpContext) {
            console.warn('No quantization data found but enabled in config, initializing new quantization');
          }
          this.initializeQuantization();
        }

        // Initialize tokenizer if text processing is enabled
        if (this.config.enableTextProcessing) {
          this.tokenizer = {
            encode: (text: string) => {
              return Array.from(text).map(c => c.charCodeAt(0) % this.vocabSize);
            },
            decode: (tokens: number[]) => {
              return tokens.map(t => String.fromCharCode(t)).join('');
            }
          };
        }

        safeLog(`Model loaded from ${path}`);
      } catch (error) {
        if (!isMcpContext) {
          console.error('Error loading model:', error);
        }
        throw new MemoryError('Failed to load model: ' + error.message);
      }
    });
  }

  /**
   * Loads the model using the legacy format
   * @param path The path to load the model from
   */
  private async loadLegacyFormat(path: string): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      safeLog('Attempting to load model in legacy format');

      // Try to load configuration
      try {
        const configPath = `${path}/config.json`;
        const configData = await fs.readFile(configPath, 'utf-8');
        this.config = JSON.parse(configData);
        safeLog('Loaded configuration from legacy format');
      } catch (error) {
        if (!isMcpContext) {
          console.warn('No config.json found in legacy format, using default configuration');
        }
      }

      // Initialize components based on config
      this.encoder = this.createEncoder();
      this.decoder = this.createDecoder();

      // Initialize optimizer
      const learningRate = this.config.learningRate || 0.001;
      this.optimizer = tf.train.adam(learningRate);

      // Try to load memory state
      try {
        const memoryPath = `${path}/memory.json`;
        const memoryData = JSON.parse(await fs.readFile(memoryPath, 'utf-8'));

        // Dispose existing memory state
        if (this.memoryState) {
          Object.values(this.memoryState).forEach(tensor => {
            if (tensor && !tensor.isDisposed) {
              tensor.dispose();
            }
          });
        }

        // Create new memory state from saved data
        this.memoryState = {
          shortTerm: tf.tensor(memoryData.shortTerm),
          longTerm: tf.tensor(memoryData.longTerm),
          meta: tf.tensor(memoryData.meta),
          timestamps: tf.tensor1d(memoryData.timestamps),
          accessCounts: tf.tensor1d(memoryData.accessCounts),
          surpriseHistory: tf.tensor1d(memoryData.surpriseHistory)
        };
        safeLog('Loaded memory state from legacy format');
      } catch (error) {
        if (!isMcpContext) {
          console.warn('No memory.json found in legacy format, initializing new memory state');
        }
        this.initializeMemoryState();
      }

      // Try to load hierarchical memory
      if (this.config.useHierarchicalMemory) {
        try {
          const hierarchicalPath = `${path}/hierarchical.json`;
          const hierarchicalData = JSON.parse(await fs.readFile(hierarchicalPath, 'utf-8'));

          this.hierarchicalMemory = {
            levels: hierarchicalData.levels.map((level: number[][]) => tf.tensor(level)),
            timestamps: hierarchicalData.timestamps.map((ts: number[]) => tf.tensor1d(ts)),
            accessCounts: hierarchicalData.accessCounts.map((ac: number[]) => tf.tensor1d(ac)),
            surpriseScores: hierarchicalData.surpriseScores.map((ss: number[]) => tf.tensor1d(ss))
          };
          safeLog('Loaded hierarchical memory from legacy format');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('No hierarchical.json found in legacy format, initializing new hierarchical memory');
          }
          this.initializeHierarchicalMemory();
        }
      }

      // Try to load quantization data
      if (this.config.enableQuantization) {
        try {
          const quantizationPath = `${path}/quantization.json`;
          const quantizationData = JSON.parse(await fs.readFile(quantizationPath, 'utf-8'));

          this.quantizationRanges = quantizationData.ranges;
          this.quantizationBits = quantizationData.bits;

          // Initialize quantized memory
          this.initializeQuantization();
          safeLog('Loaded quantization data from legacy format');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('No quantization.json found in legacy format, initializing new quantization');
          }
          this.initializeQuantization();
        }
      }

      // Initialize tokenizer if text processing is enabled
      if (this.config.enableTextProcessing) {
        this.tokenizer = {
          encode: (text: string) => {
            return Array.from(text).map(c => c.charCodeAt(0) % this.vocabSize);
          },
          decode: (tokens: number[]) => {
            return tokens.map(t => String.fromCharCode(t)).join('');
          }
        };
      }

      safeLog(`Model loaded from ${path} using legacy format`);
    } catch (error) {
      if (!isMcpContext) {
        console.error('Error loading model in legacy format:', error);
      }
      throw new MemoryError('Failed to load model in legacy format: ' + error.message);
    }
  }

  public getMemorySnapshot(): Record<string, tf.Tensor> {
    return {
      shortTerm: this.memoryState.shortTerm.clone(),
      longTerm: this.memoryState.longTerm.clone(),
      meta: this.memoryState.meta.clone(),
      timestamps: this.memoryState.timestamps.clone(),
      accessCounts: this.memoryState.accessCounts.clone(),
      surpriseHistory: this.memoryState.surpriseHistory.clone()
    };
  }

  /**
   * Cleans up resources used by the model
   */
  public dispose(): void {
    return this.withErrorHandling('dispose', () => {
      // Dispose of encoder and decoder
      if (this.encoder) {
        this.encoder.dispose();
      }

      if (this.decoder) {
        this.decoder.dispose();
      }

      // Dispose of memory state
      if (this.memoryState) {
        Object.values(this.memoryState).forEach(tensor => {
          if (tensor && !tensor.isDisposed) {
            tensor.dispose();
          }
        });
      }

      // Dispose of hierarchical memory
      if (this.hierarchicalMemory) {
        this.hierarchicalMemory.levels.forEach(tensor => tensor.dispose());
        this.hierarchicalMemory.timestamps.forEach(tensor => tensor.dispose());
        this.hierarchicalMemory.accessCounts.forEach(tensor => tensor.dispose());
        this.hierarchicalMemory.surpriseScores.forEach(tensor => tensor.dispose());
      }

      // Dispose of contrastive buffer
      this.contrastiveBuffer.forEach(tensor => tensor.dispose());
      this.contrastiveBuffer = [];

      console.log('Model resources disposed');
    });
  }

  private async getWeightData(): Promise<Record<string, number[]>> {
    return tf.tidy(() => {
      const weights: Record<string, number[]> = {};

      // Save transformer stack weights with proper naming
      this.transformerStack.forEach((layer, layerIndex) => {
        layer.getWeights().forEach((w, weightIndex) => {
          if (!w.isDisposed) {
            const weightName = `transformer_${layerIndex}_${weightIndex}`;
            weights[weightName] = Array.from(w.dataSync());
          }
        });
      });

      // Save projector weights with proper naming
      if (this.memoryProjector) {
        this.memoryProjector.getWeights().forEach((w, weightIndex) => {
          if (!w.isDisposed) {
            const weightName = `projector_layer_${weightIndex}`;
            weights[weightName] = Array.from(w.dataSync());
          }
        });
      }

      // Save similarity network weights with proper naming
      if (this.similarityNetwork) {
        this.similarityNetwork.getWeights().forEach((w, weightIndex) => {
          if (!w.isDisposed) {
            const weightName = `similarity_layer_${weightIndex}`;
            weights[weightName] = Array.from(w.dataSync());
          }
        });
      }

      return weights;
    });
  }

  public async save(modelPath: string, weightsPath: string): Promise<void> {
    await this.saveModel(modelPath);
    const weights = await this.getWeightData();
    const weightBuffers: Buffer[] = [];

    for (const data of Object.values(weights)) {
      const buffer = Buffer.from(new Float32Array(data).buffer);
      weightBuffers.push(buffer);
    }

    await fs.writeFile(weightsPath, Buffer.concat(weightBuffers));
  }

  /**
   * Loads weights from a buffer with proper error handling and version checking
   * @param weightsBuffer The buffer containing the weights
   */
  private async loadWeights(weightsBuffer: Buffer): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    return this.withErrorHandling('loadWeights', async () => {
      try {
        safeLog('Loading weights from buffer');

        // First try to parse as JSON
        try {
          const jsonData = JSON.parse(weightsBuffer.toString('utf8'));
          safeLog('Found JSON format weights');
          await this.loadWeightsFromJson(jsonData);
          return;
        } catch (jsonError) {
          // Not JSON format, try binary format
          safeLog('Not JSON format, trying binary format');
        }

        // Try binary format
        try {
          await this.loadWeightsFromBinary(weightsBuffer);
        } catch (binaryError) {
          if (!isMcpContext) {
            console.error('Failed to load weights in binary format:', binaryError);
          }
          throw new MemoryError(`Failed to load weights: ${binaryError.message}`);
        }
      } catch (error) {
        if (!isMcpContext) {
          console.error('Error loading weights:', error);
        }
        throw new MemoryError(`Failed to load weights: ${error.message}`);
      }
    });
  }

  private async loadWeightsFromJson(weightData: any): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      // Check version compatibility
      if (weightData.version && weightData.version !== '1.0') {
        if (!isMcpContext) {
          console.warn(`Weight version mismatch. Expected 1.0, got ${weightData.version}. Attempting to load anyway.`);
        }
      }

      // Load weights into a map
      const weightMap = new Map<string, tf.Tensor>();

      // Process each weight entry
      for (const [name, data] of Object.entries(weightData.weights)) {
        try {
          const { values, shape, dtype } = data as { values: number[], shape: number[], dtype: string };
          const tensor = tf.tensor(values, shape, dtype);
          weightMap.set(name, tensor);
        } catch (error) {
          if (!isMcpContext) {
            console.warn(`Error loading weight ${name}:`, error);
          }
        }
      }

      // Apply weights to model
      this.applyLoadedWeights(weightMap);

      // Load memory state if available
      if (weightData.memoryState) {
        try {
          // Dispose existing memory state
          if (this.memoryState) {
            Object.values(this.memoryState).forEach(tensor => {
              if (tensor && !tensor.isDisposed) {
                tensor.dispose();
              }
            });
          }

          // Create new memory state from saved data
          this.memoryState = {
            shortTerm: tf.tensor(weightData.memoryState.shortTerm.values, weightData.memoryState.shortTerm.shape),
            longTerm: tf.tensor(weightData.memoryState.longTerm.values, weightData.memoryState.longTerm.shape),
            meta: tf.tensor(weightData.memoryState.meta.values, weightData.memoryState.meta.shape),
            timestamps: tf.tensor1d(weightData.memoryState.timestamps.values),
            accessCounts: tf.tensor1d(weightData.memoryState.accessCounts.values),
            surpriseHistory: tf.tensor1d(weightData.memoryState.surpriseHistory.values)
          };
          safeLog('Loaded memory state from weights');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('Error loading memory state from weights:', error);
          }
          this.initializeMemoryState();
        }
      }

      // Load hierarchical memory if available
      if (weightData.hierarchicalMemory && this.config.useHierarchicalMemory) {
        try {
          const hierarchicalData = weightData.hierarchicalMemory;

          this.hierarchicalMemory = {
            levels: hierarchicalData.levels.map((level: any) =>
              tf.tensor(level.values, level.shape)),
            timestamps: hierarchicalData.timestamps.map((ts: any) =>
              tf.tensor1d(ts.values)),
            accessCounts: hierarchicalData.accessCounts.map((ac: any) =>
              tf.tensor1d(ac.values)),
            surpriseScores: hierarchicalData.surpriseScores.map((ss: any) =>
              tf.tensor1d(ss.values))
          };
          safeLog('Loaded hierarchical memory from weights');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('Error loading hierarchical memory from weights:', error);
          }
          if (this.config.useHierarchicalMemory) {
            this.initializeHierarchicalMemory();
          }
        }
      }

      // Load quantization data if available
      if (weightData.quantization && this.config.enableQuantization) {
        try {
          this.quantizationRanges = weightData.quantization.ranges;
          this.quantizationBits = weightData.quantization.bits;

          // Initialize quantized memory
          this.initializeQuantization();
          safeLog('Loaded quantization data from weights');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('Error loading quantization data from weights:', error);
          }
          if (this.config.enableQuantization) {
            this.initializeQuantization();
          }
        }
      }

      safeLog('Successfully loaded weights from JSON format');
    } catch (error) {
      if (!isMcpContext) {
        console.error('Error loading weights from JSON:', error);
      }
      throw new MemoryError(`Failed to load weights from JSON: ${error.message}`);
    }
  }

  private async loadWeightsFromBinary(weightsBuffer: Buffer): Promise<void> {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      safeLog('Loading weights from binary format');

      // Parse header
      const headerSize = weightsBuffer.readUInt32LE(0);
      const headerJson = weightsBuffer.toString('utf8', 4, 4 + headerSize);
      const header = JSON.parse(headerJson);

      // Validate header
      if (!header.format || header.format !== 'titan-memory') {
        if (!isMcpContext) {
          console.warn(`Unknown weight format: ${header.format || 'undefined'}, attempting to load anyway`);
        }
      }

      // Load weights into a map
      const weightMap = new Map<string, tf.Tensor>();
      let offset = 4 + headerSize;

      for (const [name, info] of Object.entries(header.weights)) {
        const { shape, dtype, byteOffset, byteLength } = info as WeightInfo & { byteOffset: number, byteLength: number };

        try {
          // Read tensor data
          const dataBuffer = weightsBuffer.slice(offset + byteOffset, offset + byteOffset + byteLength);

          // Create tensor based on dtype
          let tensor: tf.Tensor;
          if (dtype === 'float32') {
            const values = new Float32Array(dataBuffer.buffer, dataBuffer.byteOffset, byteLength / 4);
            tensor = tf.tensor(Array.from(values), shape, dtype);
          } else if (dtype === 'int32') {
            const values = new Int32Array(dataBuffer.buffer, dataBuffer.byteOffset, byteLength / 4);
            tensor = tf.tensor(Array.from(values), shape, dtype);
          } else {
            if (!isMcpContext) {
              console.warn(`Unsupported dtype: ${dtype} for weight ${name}, skipping`);
            }
            continue;
          }

          weightMap.set(name, tensor);
        } catch (error) {
          if (!isMcpContext) {
            console.warn(`Error loading weight ${name}:`, error);
          }
        }
      }

      // Apply weights to model
      this.applyLoadedWeights(weightMap);

      // Load memory state if available in header
      if (header.memoryState) {
        try {
          // Initialize memory state
          this.initializeMemoryState();
          safeLog('Initialized memory state from binary weights');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('Error initializing memory state from binary weights:', error);
          }
        }
      }

      safeLog('Successfully loaded weights from binary format');
    } catch (error) {
      if (!isMcpContext) {
        console.error('Error loading weights from binary format:', error);
      }
      throw new MemoryError(`Failed to load weights from binary format: ${error.message}`);
    }
  }

  /**
   * Applies loaded weights to model components
   * @param weightMap Map of weight names to tensors
   */
  private applyLoadedWeights(weightMap: Map<string, tf.Tensor>): void {
    const isMcpContext = process.env.MCP_CONTEXT === 'true';

    try {
      // Apply transformer weights
      for (let i = 0; i < this.transformerStack.length; i++) {
        for (let j = 0; j < 10; j++) { // Assuming 10 layers per transformer
          const weightName = `transformer_${i}_${j}`;
          if (weightMap.has(weightName)) {
            const weight = weightMap.get(weightName)!;
            try {
              // Apply weight to transformer layer
              const layer = this.transformerStack[i].getLayer(null, j);
              if (layer) {
                const weights = layer.getWeights();
                if (weights.length > 0 && weight.shape.every((dim, idx) => dim === weights[0].shape[idx])) {
                  layer.setWeights([weight]);
                  weightMap.delete(weightName); // Remove from map to mark as used
                } else {
                  if (!isMcpContext) {
                    console.warn(`Shape mismatch for ${weightName}, expected ${weights[0]?.shape}, got ${weight.shape}`);
                  }
                }
              }
            } catch (error) {
              if (!isMcpContext) {
                console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
              }
            }
          } else {
            if (!isMcpContext) {
              console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
            }
          }
        }
      }

      // Apply memory projector weights
      for (let i = 0; i < 4; i++) {
        const weightName = `projector_layer_${i}`;
        if (weightMap.has(weightName)) {
          const weight = weightMap.get(weightName)!;
          try {
            // Apply weight to projector layer
            const layer = this.memoryProjector.getLayer(null, i);
            if (layer) {
              const weights = layer.getWeights();
              if (weights.length > 0 && weight.shape.every((dim, idx) => dim === weights[0].shape[idx])) {
                layer.setWeights([weight]);
                weightMap.delete(weightName); // Remove from map to mark as used
              } else {
                if (!isMcpContext) {
                  console.warn(`Shape mismatch for ${weightName}, expected ${weights[0]?.shape}, got ${weight.shape}`);
                }
              }
            }
          } catch (error) {
            if (!isMcpContext) {
              console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
            }
          }
        } else {
          if (!isMcpContext) {
            console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
          }
        }
      }

      // Apply similarity network weights
      for (let i = 0; i < 4; i++) {
        const weightName = `similarity_layer_${i}`;
        if (weightMap.has(weightName)) {
          const weight = weightMap.get(weightName)!;
          try {
            // Apply weight to similarity layer
            const layer = this.similarityNetwork.getLayer(null, i);
            if (layer) {
              const weights = layer.getWeights();
              if (weights.length > 0 && weight.shape.every((dim, idx) => dim === weights[0].shape[idx])) {
                layer.setWeights([weight]);
                weightMap.delete(weightName); // Remove from map to mark as used
              } else {
                if (!isMcpContext) {
                  console.warn(`Shape mismatch for ${weightName}, expected ${weights[0]?.shape}, got ${weight.shape}`);
                }
              }
            }
          } catch (error) {
            if (!isMcpContext) {
              console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
            }
          }
        } else {
          if (!isMcpContext) {
            console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
          }
        }
      }

      // Apply encoder/decoder weights if available
      if (weightMap.has('encoder') && this.encoder) {
        try {
          const encoderWeights = weightMap.get('encoder')!;
          // Apply encoder weights
          weightMap.delete('encoder');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('Error applying encoder weights:', error);
          }
        }
      }

      if (weightMap.has('decoder') && this.decoder) {
        try {
          const decoderWeights = weightMap.get('decoder')!;
          // Apply decoder weights
          weightMap.delete('decoder');
        } catch (error) {
          if (!isMcpContext) {
            console.warn('Error applying decoder weights:', error);
          }
        }
      }

      // Clean up any unused tensors
      weightMap.forEach((tensor, name) => {
        if (!tensor.isDisposed) {
          if (!isMcpContext) {
            console.warn(`Unused weight tensor: ${name}, disposing`);
          }
          tensor.dispose();
        }
      });

      safeLog('Applied loaded weights to model');
    } catch (error) {
      if (!isMcpContext) {
        console.error('Error applying weights to model:', error);
      }
      throw new MemoryError(`Failed to apply weights: ${error.message}`);
    }
  }

  private updateMemoryState(input: tf.Tensor2D, surprise: ISurpriseMetrics): IMemoryUpdateResult {
    // Create tensors outside tidy to ensure they're not disposed
    const shortTermUpdate = tf.tidy(() => {
      return SafeTensorOps.add(
        SafeTensorOps.mul(this.memoryState.shortTerm, tf.scalar(this.config.decayRate)),
        input
      );
    });

    const longTermUpdate = tf.tidy(() => {
      return SafeTensorOps.add(
        this.memoryState.longTerm,
        SafeTensorOps.mul(input, tf.expandDims(surprise.accumulated, -1))
      );
    });

    const metaUpdate = this.updateMetaMemory(surprise, input);
    const currentTime = Date.now();
    const newTimestamps = tf.fill(this.memoryState.timestamps.shape, currentTime);
    const newAccessCounts = SafeTensorOps.add(this.memoryState.accessCounts, tf.ones(this.memoryState.accessCounts.shape));
    const attention = this.computeMemoryAttention(input);

    const newState: IMemoryState = {
      shortTerm: shortTermUpdate,
      longTerm: longTermUpdate,
      meta: metaUpdate,
      timestamps: newTimestamps,
      accessCounts: newAccessCounts,
      surpriseHistory: surprise.accumulated
    };

    return {
      newState,
      attention,
      surprise
    };
  }

  private computeGradients(input: tf.Tensor2D, target: tf.Tensor2D): IModelGradients {
    const error = tf.tidy(() => {
      const { values: attended } = this.computeMemoryAttention(input);
      const prediction = SafeTensorOps.add(attended, input);
      return SafeTensorOps.sub(prediction, target);
    });

    const { value: loss } = tf.variableGrads(() => {
      const [keyWeight, valueWeight] = this.similarityNetwork.getWeights() as [tf.Tensor2D, tf.Tensor2D];
      const keys = SafeTensorOps.matMul(this.memoryState.shortTerm, keyWeight);
      const values = SafeTensorOps.matMul(this.memoryState.shortTerm, valueWeight);
      const scores = tf.softmax(SafeTensorOps.matMul(input, keys.transpose()));
      const attended = SafeTensorOps.matMul(scores, values);
      const prediction = SafeTensorOps.add(attended, input);
      return tf.mean(tf.square(SafeTensorOps.sub(prediction, target)));
    });

    return {
      shortTerm: error,
      longTerm: error,
      meta: tf.keep(loss) as tf.Tensor
    };
  }

  /**
   * Resets accumulated gradients and optimizer state
   * This is useful when encountering gradient explosion or NaN values
   */
  public resetGradients(): void {
    tf.tidy(() => {
      // Recreate optimizer with the same learning rate
      const learningRate = this.config.learningRate || 0.001;
      this.optimizer = tf.train.adam(learningRate);

      // Reset step count
      this.stepCount = 0;

      console.log('Gradients and optimizer state reset successfully');
    });
  }

  // Add MCP server compatibility methods
  public async init_model(config: Partial<TitanMemoryConfig>): Promise<{ status: string }> {
    try {
      await this.initialize(config);
      return { status: 'success' };
    } catch (error) {
      return { status: 'error', message: error.message };
    }
  }

  public async forward_pass(x: string | number[], memoryState?: IMemoryState): Promise<{
    predicted: number[];
    memoryUpdate: {
      shortTerm: number[][];
      timestamps: number[];
      accessCounts: number[];
      surpriseHistory: number[];
    };
  }> {
    try {
      // Process input
      let input: tf.Tensor;
      if (typeof x === 'string') {
        input = await this.encodeText(x);
      } else {
        input = tf.tensor(x);
      }

      // Use provided memory state or current state
      const state = memoryState || this.memoryState;

      // Forward pass
      const result = this.forward(input, state);

      // Convert tensors to arrays for JSON serialization
      const predicted = Array.from(await result.predicted.tensor.data()) as number[];

      // Get memory update as arrays
      const shortTermArray = await result.memoryUpdate.newState.shortTerm.array() as number[][];
      const timestampsArray = Array.from(await result.memoryUpdate.newState.timestamps.data());
      const accessCountsArray = Array.from(await result.memoryUpdate.newState.accessCounts.data());
      const surpriseHistoryArray = Array.from(await result.memoryUpdate.newState.surpriseHistory.data());

      // Clean up tensors
      input.dispose();
      result.predicted.tensor.dispose();

      return {
        predicted,
        memoryUpdate: {
          shortTerm: shortTermArray,
          timestamps: timestampsArray,
          accessCounts: accessCountsArray,
          surpriseHistory: surpriseHistoryArray
        }
      };
    } catch (error: unknown) {
      // Return a properly formatted error response
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(JSON.stringify({ error: errorMessage }));
    }
  }

  public async train_step(x_t: string | number[], x_next: string | number[]): Promise<{
    loss: number;
  }> {
    try {
      // Process inputs
      let current: tf.Tensor;
      let next: tf.Tensor;

      if (typeof x_t === 'string') {
        current = await this.encodeText(x_t);
      } else {
        current = tf.tensor(x_t);
      }

      if (typeof x_next === 'string') {
        next = await this.encodeText(x_next);
      } else {
        next = tf.tensor(x_next);
      }

      // Train step
      const result = this.trainStep(
        { tensor: current, shape: current.shape },
        { tensor: next, shape: next.shape },
        this.memoryState
      );

      // Get loss as number
      const lossValue = await result.loss.tensor.data()[0];

      // Clean up tensors
      current.dispose();
      next.dispose();
      result.loss.tensor.dispose();

      return { loss: lossValue };
    } catch (error: unknown) {
      // Return a properly formatted error response
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(JSON.stringify({ error: errorMessage }));
    }
  }

  public get_memory_state(): {
    stats: {
      meanActivation: number;
      patternDiversity: number;
      surpriseScore: number;
    };
    capacity: number;
    status: string;
  } {
    try {
      // Calculate memory statistics
      const shortTermMean = this.memoryState.shortTerm.mean().dataSync()[0];
      const longTermMean = this.memoryState.longTerm.mean().dataSync()[0];
      const metaMean = this.memoryState.meta.mean().dataSync()[0];

      // Calculate pattern diversity (standard deviation across memory)
      const shortTermStd = this.memoryState.shortTerm.std().dataSync()[0];
      const longTermStd = this.memoryState.longTerm.std().dataSync()[0];

      // Get surprise score from history
      const surpriseScore = this.memoryState.surpriseHistory.mean().dataSync()[0];

      // Calculate memory capacity
      const memorySize = this.config.memorySlots || 5000;
      const usedSlots = this.memoryState.accessCounts.greater(tf.scalar(0)).sum().dataSync()[0];
      const capacity = 1 - (usedSlots / memorySize);

      // Determine status
      let status = 'active';
      if (capacity < 0.1) {
        status = 'critical';
      } else if (capacity < 0.3) {
        status = 'warning';
      }

      // Return formatted stats
      return {
        stats: {
          meanActivation: (shortTermMean + longTermMean + metaMean) / 3,
          patternDiversity: (shortTermStd + longTermStd) / 2,
          surpriseScore
        },
        capacity,
        status
      };
    } catch (error: unknown) {
      // Return a properly formatted error response
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        stats: {
          meanActivation: 0,
          patternDiversity: 0,
          surpriseScore: 0
        },
        capacity: 0,
        status: 'error'
      };
    }
  }

  // Add required interface methods
  public getMemoryState(): IMemoryState {
    return {
      shortTerm: this.memoryState.shortTerm.clone(),
      longTerm: this.memoryState.longTerm.clone(),
      meta: this.memoryState.meta.clone(),
      timestamps: this.memoryState.timestamps.clone(),
      accessCounts: this.memoryState.accessCounts.clone(),
      surpriseHistory: this.memoryState.surpriseHistory.clone()
    };
  }

  public resetMemory(): void {
    this.initializeMemoryState();
  }

  /**
   * Initializes hierarchical memory structure if enabled in config
   */
  private initializeHierarchicalMemory(): void {
    if (!this.config.useHierarchicalMemory) {
      this.hierarchicalMemory = null;
      return;
    }

    return this.withErrorHandling('initializeHierarchicalMemory', () => {
      // Create multi-level memory structure
      const levels = this.hierarchicalLevels;
      const slotsPerLevel = Math.floor(this.config.memorySlots / levels);
      const embeddingSize = this.config.memoryDim;

      // Initialize memory levels with decreasing resolution and increasing time spans
      const shortTermLevels = Array(levels).fill(0).map((_, i) => {
        // Each level has fewer slots but covers longer time spans
        const levelSlots = Math.max(1, Math.floor(slotsPerLevel / (i + 1)));
        return tf.zeros([levelSlots, embeddingSize]);
      });

      // Initialize corresponding metadata for each level
      const timestampLevels = Array(levels).fill(0).map((_, i) => {
        const levelSlots = Math.max(1, Math.floor(slotsPerLevel / (i + 1)));
        return tf.zeros([levelSlots]);
      });

      const accessCountLevels = Array(levels).fill(0).map((_, i) => {
        const levelSlots = Math.max(1, Math.floor(slotsPerLevel / (i + 1)));
        return tf.zeros([levelSlots]);
      });

      // Initialize surprise scores for each level
      const surpriseLevels = Array(levels).fill(0).map((_, i) => {
        return tf.zeros([Math.max(10, Math.floor(100 / (i + 1)))]);
      });

      this.hierarchicalMemory = {
        levels: shortTermLevels,
        timestamps: timestampLevels,
        accessCounts: accessCountLevels,
        surpriseScores: surpriseLevels
      };

      console.log(`Initialized hierarchical memory with ${levels} levels`);
    });
  }

  /**
   * Updates hierarchical memory with new information
   * @param input The input tensor to store in memory
   * @param surprise The surprise score for this input
   */
  private updateHierarchicalMemory(input: ITensorWrapper, surprise: ITensorWrapper): void {
    if (!this.hierarchicalMemory || !this.config.useHierarchicalMemory) {
      return;
    }

    return this.withErrorHandling('updateHierarchicalMemory', () => {
      const hierarchicalMemory = this.hierarchicalMemory as {
        levels: tf.Tensor[];
        timestamps: tf.Tensor[];
        accessCounts: tf.Tensor[];
        surpriseScores: tf.Tensor[];
      };
      const { levels, accessCounts } = hierarchicalMemory;

      // Update each level with different time scales
      levels.forEach((levelMemory, levelIndex) => {
        // Higher levels update less frequently
        const shouldUpdateLevel = (this.stepCount % Math.pow(2, levelIndex)) === 0;
        if (!shouldUpdateLevel && levelIndex > 0) {
          return;
        }

        // Find least recently used slot for this level
        const levelTimestamps = hierarchicalMemory.timestamps[levelIndex];
        const oldestSlotIndex = tf.argMin(levelTimestamps).dataSync()[0];

        // Update memory at the selected slot
        const inputArray = unwrapTensor(input).arraySync();
        const newMemory = levelMemory.arraySync();
        newMemory[oldestSlotIndex] = inputArray;

        // Update metadata
        const newTimestamps = levelTimestamps.arraySync();
        newTimestamps[oldestSlotIndex] = Date.now();

        const newAccessCounts = accessCounts[levelIndex].arraySync();
        newAccessCounts[oldestSlotIndex] = 1; // Reset access count for new memory

        // Update surprise history with exponential decay
        const newSurpriseScores = hierarchicalMemory.surpriseScores[levelIndex].arraySync();
        newSurpriseScores.shift(); // Remove oldest
        newSurpriseScores.push(unwrapTensor(surprise).dataSync()[0]); // Add newest

        // Update tensors
        tf.dispose(levels[levelIndex]);
        tf.dispose(hierarchicalMemory.timestamps[levelIndex]);
        tf.dispose(accessCounts[levelIndex]);
        tf.dispose(hierarchicalMemory.surpriseScores[levelIndex]);

        levels[levelIndex] = tf.tensor(newMemory);
        hierarchicalMemory.timestamps[levelIndex] = tf.tensor(newTimestamps);
        accessCounts[levelIndex] = tf.tensor(newAccessCounts);
        hierarchicalMemory.surpriseScores[levelIndex] = tf.tensor(newSurpriseScores);
      });
    });
  }

  /**
   * Retrieves memories from hierarchical structure based on query
   * @param query The query tensor to match against memories
   * @returns The retrieved memory tensor
   */
  private retrieveFromHierarchicalMemory(query: ITensorWrapper): ITensorWrapper {
    if (!this.hierarchicalMemory || !this.config.useHierarchicalMemory) {
      // Fall back to standard memory retrieval
      return this.retrieveFromMemory(query);
    }

    return this.withErrorHandling('retrieveFromHierarchicalMemory', () => {
      const hierarchicalMemory = this.hierarchicalMemory as {
        levels: tf.Tensor[];
        timestamps: tf.Tensor[];
        accessCounts: tf.Tensor[];
        surpriseScores: tf.Tensor[];
      };
      const { levels, accessCounts } = hierarchicalMemory;

      // Calculate attention across all levels
      const attentionResults = levels.map((levelMemory, levelIndex) => {
        // Calculate similarity between query and all memories at this level
        const similarities = tf.matMul(
          levelMemory,
          unwrapTensor(query).reshape([unwrapTensor(query).shape[0], 1]),
          false,
          true
        );

        // Apply temperature scaling
        const temperature = 1.0 / (levelIndex + 1); // Lower temperature for higher levels
        const scaledSimilarities = tf.div(similarities, tf.scalar(temperature));

        // Convert to attention weights
        const attentionWeights = tf.softmax(scaledSimilarities);

        // Update access counts
        const newAccessCounts = accessCounts[levelIndex].add(attentionWeights);
        tf.dispose(accessCounts[levelIndex]);
        accessCounts[levelIndex] = newAccessCounts;

        // Weight memories by attention
        const weightedMemories = tf.matMul(
          attentionWeights,
          levelMemory,
          true,
          false
        );

        // Apply level importance (higher levels have more weight)
        const levelImportance = Math.pow(0.8, levelIndex); // Exponential decay of importance
        return tf.mul(weightedMemories, tf.scalar(levelImportance));
      });

      // Combine results from all levels
      const combinedMemory = attentionResults.reduce((acc, levelResult) => {
        const result = acc ? tf.add(acc, levelResult) : levelResult;
        if (acc) tf.dispose(acc);
        return result;
      }, null);

      // Normalize the result
      const normalizedMemory = tf.div(
        combinedMemory,
        tf.norm(combinedMemory)
      );

      // Dispose intermediate tensors
      attentionResults.forEach(tensor => tf.dispose(tensor));
      tf.dispose(combinedMemory);

      return wrapTensor(normalizedMemory);
    });
  }

  /**
   * Initializes quantization if enabled in config
   */
  private initializeQuantization(): void {
    if (!this.config.enableQuantization) {
      this.quantizedMemory = null;
      return;
    }

    return this.withErrorHandling('initializeQuantization', () => {
      const memorySlots = this.config.memorySlots;
      const embeddingSize = this.config.memoryDim;

      // Initialize quantization ranges for each dimension
      this.quantizationRanges = Array(embeddingSize).fill(0).map(() => ({
        min: -1.0,
        max: 1.0
      }));

      // Initialize quantized memory
      this.quantizedMemory = {
        shortTerm: new Uint8Array(memorySlots * embeddingSize),
        longTerm: new Uint8Array(Math.floor(memorySlots / 2) * embeddingSize),
        meta: new Uint8Array(memorySlots * 5),
        quantizationRanges: this.quantizationRanges
      };

      console.log(`Initialized quantized memory with ${this.quantizationBits} bits precision`);
    });
  }

  /**
   * Quantizes a tensor to lower precision
   * @param tensor The tensor to quantize
   * @returns The quantized data as Uint8Array
   */
  private quantizeTensor(tensor: tf.Tensor): Uint8Array {
    return this.withErrorHandling('quantizeTensor', () => {
      const data = tensor.dataSync();
      const shape = tensor.shape;
      const totalElements = shape.reduce((a, b) => a * b, 1);

      // Create quantized array
      const quantized = new Uint8Array(totalElements);

      // Determine quantization range
      const maxValue = 2 ** this.quantizationBits - 1;

      // Update ranges if needed
      if (tensor.rank === 2 && shape[1] === this.config.embeddingSize) {
        // For embedding tensors, track per-dimension ranges
        const values = tensor.arraySync() as number[][];

        for (let dim = 0; dim < shape[1]; dim++) {
          let min = Infinity;
          let max = -Infinity;

          // Find min/max for this dimension
          for (let i = 0; i < shape[0]; i++) {
            const val = values[i][dim];
            if (val < min) min = val;
            if (val > max) max = val;
          }

          // Update range with exponential moving average
          const alpha = 0.1; // Smoothing factor
          this.quantizationRanges[dim].min = (1 - alpha) * this.quantizationRanges[dim].min + alpha * min;
          this.quantizationRanges[dim].max = (1 - alpha) * this.quantizationRanges[dim].max + alpha * max;

          // Quantize values for this dimension
          for (let i = 0; i < shape[0]; i++) {
            const val = values[i][dim];
            const normalized = (val - this.quantizationRanges[dim].min) /
              (this.quantizationRanges[dim].max - this.quantizationRanges[dim].min);
            const quantizedVal = Math.min(maxValue, Math.max(0, Math.round(normalized * maxValue)));
            quantized[i * shape[1] + dim] = quantizedVal;
          }
        }
      } else {
        // For other tensors, use global min/max
        const min = tf.min(tensor).dataSync()[0];
        const max = tf.max(tensor).dataSync()[0];

        // Quantize all values
        for (let i = 0; i < totalElements; i++) {
          const normalized = (data[i] - min) / (max - min);
          const quantizedVal = Math.min(maxValue, Math.max(0, Math.round(normalized * maxValue)));
          quantized[i] = quantizedVal;
        }
      }

      return quantized;
    });
  }

  /**
   * Dequantizes data back to full precision tensor
   * @param quantized The quantized data
   * @param shape The tensor shape
   * @param ranges Optional quantization ranges for per-dimension dequantization
   * @returns The dequantized tensor
   */
  private dequantizeTensor(quantized: Uint8Array, shape: number[], ranges?: { min: number; max: number }[]): tf.Tensor {
    return this.withErrorHandling('dequantizeTensor', () => {
      const totalElements = shape.reduce((a, b) => a * b, 1);
      const dequantized = new Float32Array(totalElements);

      // Determine dequantization parameters
      const maxValue = 2 ** this.quantizationBits - 1;

      if (ranges && shape.length === 2 && shape[1] === this.config.embeddingSize) {
        // For embedding tensors, use per-dimension ranges
        for (let i = 0; i < shape[0]; i++) {
          for (let dim = 0; dim < shape[1]; dim++) {
            const quantizedVal = quantized[i * shape[1] + dim];
            const normalized = quantizedVal / maxValue;
            const range = ranges[dim];
            dequantized[i * shape[1] + dim] = normalized * (range.max - range.min) + range.min;
          }
        }
      } else {
        // For other tensors, use global min/max
        const min = -1.0;
        const max = 1.0;

        for (let i = 0; i < totalElements; i++) {
          const normalized = quantized[i] / maxValue;
          dequantized[i] = normalized * (max - min) + min;
        }
      }

      return tf.tensor(dequantized, shape);
    });
  }

  /**
   * Updates quantized memory with new tensor data
   * @param tensor The tensor to store in quantized form
   * @param memoryType The type of memory to update ('shortTerm', 'longTerm', or 'meta')
   */
  private updateQuantizedMemory(tensor: tf.Tensor, memoryType: 'shortTerm' | 'longTerm' | 'meta'): void {
    if (!this.quantizedMemory || !this.config.enableQuantization) {
      return;
    }

    return this.withErrorHandling('updateQuantizedMemory', () => {
      // Quantize the tensor
      const quantized = this.quantizeTensor(tensor);

      // Update the appropriate memory
      this.quantizedMemory![memoryType] = quantized;

      // Update quantization ranges
      if (memoryType === 'shortTerm' || memoryType === 'longTerm') {
        this.quantizedMemory!.quantizationRanges = this.quantizationRanges;
      }
    });
  }

  /**
   * Retrieves tensor from quantized memory
   * @param memoryType The type of memory to retrieve ('shortTerm', 'longTerm', or 'meta')
   * @param shape The shape of the tensor to reconstruct
   * @returns The dequantized tensor
   */
  private retrieveQuantizedMemory(memoryType: 'shortTerm' | 'longTerm' | 'meta', shape: number[]): tf.Tensor {
    if (!this.quantizedMemory || !this.config.enableQuantization) {
      throw new MemoryError('Quantized memory not initialized');
    }

    return this.withErrorHandling('retrieveQuantizedMemory', () => {
      // Get the quantized data
      const quantized = this.quantizedMemory![memoryType];

      // Dequantize based on memory type
      if (memoryType === 'shortTerm' || memoryType === 'longTerm') {
        return this.dequantizeTensor(
          quantized,
          shape,
          this.quantizedMemory!.quantizationRanges
        );
      } else {
        return this.dequantizeTensor(quantized, shape);
      }
    });
  }

  /**
   * Updates memory with quantization support
   */
  private updateMemory(
    input: ITensorWrapper,
    surprise: ITensorWrapper,
    state: IMemoryState
  ): IMemoryState {
    return tf.tidy(() => {
      // Find least recently used memory slot
      const oldestSlotIndex = tf.argMin(state.timestamps).dataSync()[0];

      // Update memory at the selected slot
      const inputArray = unwrapTensor(input).arraySync();
      const newShortTerm = state.shortTerm.arraySync();
      newShortTerm[oldestSlotIndex] = inputArray;

      // Update metadata
      const newTimestamps = state.timestamps.arraySync();
      newTimestamps[oldestSlotIndex] = Date.now();

      const newAccessCounts = state.accessCounts.arraySync();
      newAccessCounts[oldestSlotIndex] = 1; // Reset access count for new memory

      // Update surprise history with exponential decay
      const newSurpriseHistory = state.surpriseHistory.arraySync();
      newSurpriseHistory.shift(); // Remove oldest
      newSurpriseHistory.push(unwrapTensor(surprise).dataSync()[0]); // Add newest

      // Create new state
      const newState = {
        shortTerm: tf.tensor(newShortTerm),
        longTerm: state.longTerm.clone(),
        meta: state.meta.clone(),
        timestamps: tf.tensor(newTimestamps),
        accessCounts: tf.tensor(newAccessCounts),
        surpriseHistory: tf.tensor(newSurpriseHistory)
      };

      // Update quantized memory if enabled
      if (this.config.enableQuantization && this.quantizedMemory) {
        this.updateQuantizedMemory(newState.shortTerm, 'shortTerm');
        this.updateQuantizedMemory(newState.longTerm, 'longTerm');
        this.updateQuantizedMemory(newState.meta, 'meta');
      }

      return newState;
    });
  }
}