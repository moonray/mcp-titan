/**
 * @fileovertitle Titan Memory Model 2.0 - Neural Memory Architecture with Transformer-XL Inspired Mechanisms
 */

import * as tf from '@tensorflow/tfjs-node';
import { ITensor, IMemoryState, ISurpriseMetrics, IAttentionBlock, IMemoryUpdateResult, IModelGradients, TensorContainer, unwrapTensor, wrapTensor, IMemoryModel } from './types.js';
import * as fs from 'fs/promises';
import { z } from 'zod';
import { checkNullOrUndefined, validateTensor, validateTensorShape, SafeTensorOps } from './utils.js';

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
});

type TitanMemoryConfig = z.infer<typeof ModelConfigSchema>;

interface WeightInfo {
  shape: number[];
  dtype: string;
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

  public async encodeText(text: string): Promise<tf.Tensor1D> {
    return tf.tidy(() => {
      // Tokenize text into subwords/characters
      const tokens = this.tokenize(text);

      // Convert tokens to IDs and pad sequence
      const tokenIds = this.padSequence(
        tokens.map(token => this.vocabulary.get(token) || this.vocabulary.get('[UNK]')!)
      );

      // Create tensor and apply embedding
      const inputTensor = tf.tensor2d([tokenIds], [1, this.config.maxSequenceLength]);
      let encoding = this.applyPositionalEncoding(inputTensor);

      // Process through transformer stack
      for (const layer of this.transformerStack) {
        encoding = layer.apply(encoding) as tf.Tensor2D;
      }

      // Mean pooling over sequence length
      return tf.mean(encoding, 1).squeeze() as tf.Tensor1D;
    });
  }

  private tokenize(text: string): string[] {
    // Simple character-level tokenization with basic subword units
    const tokens: string[] = [];
    let currentToken = '';

    const addToken = () => {
      if (currentToken) {
        tokens.push(currentToken);
        currentToken = '';
      }
    };

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      // Handle special characters
      if ('.,!?-_\'"`()[]{}:;/\\+=<>'.includes(char)) {
        addToken();
        tokens.push(char);
        continue;
      }

      // Handle whitespace
      if (char === ' ') {
        addToken();
        continue;
      }

      // Build subword tokens
      currentToken += char;

      // Check if current token exists in vocabulary
      if (this.vocabulary.has(currentToken)) {
        if (i === text.length - 1 || !this.vocabulary.has(currentToken + text[i + 1])) {
          addToken();
        }
      }
    }

    addToken();
    return tokens;
  }

  private padSequence(tokens: number[]): number[] {
    const padded = tokens.slice(0, this.config.maxSequenceLength);
    while (padded.length < this.config.maxSequenceLength) {
      padded.push(this.vocabulary.get('[PAD]')!);
    }
    return padded;
  }

  private applyPositionalEncoding(input: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      const position = tf.range(0, input.shape[1]);
      // Always use config dimension since we're working with 2D tensors
      const numDimensions = this.config.inputDim;

      // Create position encodings
      const positionMatrix = position.expandDims(1);
      const divTerm = tf.exp(
        tf.mul(
          tf.range(0, numDimensions, 2).cast('float32'),
          tf.scalar(-(Math.log(10000.0) / numDimensions))
        )
      );

      const sinTerms = tf.sin(tf.matMul(positionMatrix, divTerm.reshape([1, -1])));
      const cosTerms = tf.cos(tf.matMul(positionMatrix, divTerm.reshape([1, -1])));

      const positionalEncoding = tf.concat([sinTerms, cosTerms], 1);

      // Add positional encoding to input
      return tf.add(input, positionalEncoding.expandDims(0));
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
    // Ensure we're in a tidy block for memory management
    tf.engine().startScope();
    try {
      // Verify backend is ready before proceeding
      const backend = tf.getBackend();
      if (!backend) {
        throw new Error('TensorFlow backend not initialized');
      }

      // Initialize with proper memory management
      const initializer = tf.initializers.glorotNormal({});
      const memory = tf.tidy(() => {
        const mem = initializer.apply([this.config.memorySlots, this.config.memoryDim]) as tf.Tensor2D;
        return mem.clone(); // Clone to prevent disposal issues
      });

      // Clean up old state if it exists
      if (this.memoryState) {
        Object.values(this.memoryState).forEach(tensor => {
          if (tensor && !tensor.isDisposed) {
            tensor.dispose();
          }
        });
      }

      // Create new state with proper tensor management
      this.memoryState = {
        shortTerm: tf.keep(memory.clone()),
        longTerm: tf.keep(memory.clone()),
        meta: tf.keep(tf.zeros([this.config.memorySlots, this.config.memoryDim])),
        timestamps: tf.keep(tf.zeros([this.config.memorySlots])),
        accessCounts: tf.keep(tf.zeros([this.config.memorySlots])),
        surpriseHistory: tf.keep(tf.zeros([this.config.memorySlots]))
      };

      // Verify tensors are valid
      Object.entries(this.memoryState).forEach(([key, tensor]) => {
        if (!tensor || tensor.isDisposed) {
          throw new Error(`Failed to initialize ${key} tensor`);
        }
      });

      // Clean up the initial memory tensor
      memory.dispose();
    } catch (error) {
      console.error('Error initializing memory state:', error);
      throw error;
    } finally {
      tf.engine().endScope();
    }
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

  public forward(x: ITensor, memoryState: IMemoryState): { predicted: ITensor; memoryUpdate: IMemoryUpdateResult } {
    const input = unwrapTensor(x) as tf.Tensor2D;
    let transformed = input;
    const tensorsToDispose: tf.Tensor[] = [];

    try {
      // Process through transformer stack
      for (const layer of this.transformerStack) {
        const newTransformed = layer.apply(transformed) as tf.Tensor2D;
        if (transformed !== input) {
          tensorsToDispose.push(transformed);
        }
        transformed = newTransformed;
      }

      // Memory attention mechanisms
      const memoryQuery = this.memoryProjector.apply(transformed) as tf.Tensor2D;
      tensorsToDispose.push(memoryQuery);

      const attention = this.computeMemoryAttention(memoryQuery);
      tensorsToDispose.push(attention.keys, attention.values, attention.scores);

      // Surprise-gated memory update
      const surprise = this.computeSurprise(transformed, attention.values as tf.Tensor2D);
      tensorsToDispose.push(surprise.immediate, surprise.accumulated);

      const updateGate = tf.sigmoid(tf.mul(surprise.immediate, 0.5));
      tensorsToDispose.push(updateGate);

      const newShortTerm = tf.add(
        tf.mul(memoryState.shortTerm, tf.sub(1, updateGate)),
        tf.mul(attention.values, updateGate)
      ) as tf.Tensor2D;

      const newState = {
        ...memoryState,
        shortTerm: newShortTerm
      };

      return {
        predicted: wrapTensor(transformed),
        memoryUpdate: {
          newState,
          attention,
          surprise
        }
      };
    } finally {
      tensorsToDispose.forEach(t => t.dispose());
    }
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

  public trainStep(x_t: ITensor, x_next: ITensor, memoryState: IMemoryState): { loss: ITensor; gradients: IModelGradients } {
    const { predicted, memoryUpdate } = this.forward(x_t, memoryState);
    const target = unwrapTensor(x_next) as tf.Tensor2D;

    const variables = this.transformerStack.flatMap(layer => layer.getWeights())
      .concat(this.memoryProjector.getWeights())
      .concat(this.similarityNetwork.getWeights())
      .map(w => tf.variable(w)) as tf.Variable[];

    const { value: loss, grads } = this.optimizer.computeGradients(() => {
      const predictionLoss = tf.losses.meanSquaredError(target, predicted);
      const surpriseLoss = tf.mul(tf.mean(memoryUpdate.surprise.immediate), 0.1);
      const diversityLoss = tf.neg(tf.mean(tf.square(tf.matMul(
        this.memoryState.shortTerm,
        this.memoryState.shortTerm.transpose()
      ))));

      return tf.add(predictionLoss, tf.add(surpriseLoss, diversityLoss)) as tf.Scalar;
    }, variables);

    this.optimizer.applyGradients(grads);

    return {
      loss,
      gradients: {
        shortTerm: grads['shortTerm'] || tf.zeros([0]),
        longTerm: grads['longTerm'] || tf.zeros([0]),
        meta: grads['meta'] || tf.zeros([0])
      }
    };
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

  public async saveModel(path: string): Promise<void> {
    try {
      // Save model topology and metadata separately
      const modelMetadata = {
        version: "1.0",
        format: "titan-memory-v1",
        created: new Date().toISOString(),
        config: {
          ...this.config,
          architecture: {
            transformerLayers: this.transformerStack.length,
            projectorConfig: this.memoryProjector.getConfig(),
            similarityConfig: this.similarityNetwork.getConfig()
          }
        },
        weights: {
          format: "binary",
          dtype: "float32",
          shapes: {} as Record<string, number[]>
        }
      };

      // Save metadata as JSON with proper formatting
      await fs.writeFile(
        path,
        JSON.stringify(modelMetadata, null, 2),
        { encoding: 'utf8' }
      );

      // Save weights separately using tf.io handlers
      const weightsPath = path.replace('.json', '.weights.bin');
      await tf.io.withSaveHandler(async (tensors) => {
        const weightData = new Map<string, tf.Tensor>();

        // Process transformer stack weights
        this.transformerStack.forEach((layer, layerIndex) => {
          layer.getWeights().forEach((w, weightIndex) => {
            if (!w.isDisposed) {
              const weightName = `transformer_${layerIndex}_${weightIndex}`;
              weightData.set(weightName, w.clone());
              modelMetadata.weights.shapes[weightName] = w.shape;
            }
          });
        });

        // Process projector weights
        if (this.memoryProjector) {
          this.memoryProjector.getWeights().forEach((w, weightIndex) => {
            if (!w.isDisposed) {
              const weightName = `projector_layer_${weightIndex}`;
              weightData.set(weightName, w.clone());
              modelMetadata.weights.shapes[weightName] = w.shape;
            }
          });
        }

        // Process similarity network weights
        if (this.similarityNetwork) {
          this.similarityNetwork.getWeights().forEach((w, weightIndex) => {
            if (!w.isDisposed) {
              const weightName = `similarity_layer_${weightIndex}`;
              weightData.set(weightName, w.clone());
              modelMetadata.weights.shapes[weightName] = w.shape;
            }
          });
        }

        // Write weights to binary file
        const weightBuffers: Buffer[] = [];
        for (const [name, tensor] of weightData.entries()) {
          const data = await tensor.data();
          const buffer = Buffer.from(data.buffer);
          weightBuffers.push(
            Buffer.from(name + ':' + tensor.shape.join(',') + ':'),
            buffer
          );
          tensor.dispose();
        }

        // Update metadata file with final weight info
        await fs.writeFile(
          path,
          JSON.stringify(modelMetadata, null, 2),
          { encoding: 'utf8' }
        );

        await fs.writeFile(weightsPath, Buffer.concat(weightBuffers));
        return {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: 'JSON',
            weightDataBytes: weightBuffers.reduce((sum, buf) => sum + buf.length, 0)
          }
        };
      });
    } catch (error) {
      console.error('Error saving model:', error);
      throw error;
    }
  }

  public async loadModel(path: string): Promise<void> {
    try {
      // Read and parse metadata
      const metadata = JSON.parse(
        await fs.readFile(path, { encoding: 'utf8' })
      );

      // Validate metadata format
      if (!metadata || typeof metadata !== 'object') {
        throw new Error('Invalid model metadata format: not a valid JSON object');
      }

      // Validate format field
      if (!metadata.format) {
        console.log('No format specified in metadata, creating default metadata');
        metadata.format = 'titan-memory-v1';
        metadata.version = '1.0';
        metadata.created = new Date().toISOString();
        await fs.writeFile(path, JSON.stringify(metadata, null, 2));
      } else if (metadata.format !== 'titan-memory-v1') {
        throw new Error(`Invalid model metadata format: expected 'titan-memory-v1', got '${metadata.format}'`);
      }

      // Ensure config exists
      if (!metadata.config) {
        console.log('No config in metadata, using current config');
        metadata.config = this.config;
        await fs.writeFile(path, JSON.stringify(metadata, null, 2));
      }

      // Initialize components with the config
      await this.initialize(metadata.config);

      // Load weights if they exist
      const weightsPath = path.replace('.json', '.weights.bin');
      try {
        const weightsExists = await fs.access(weightsPath).then(() => true).catch(() => false);
        if (weightsExists) {
          console.log('Loading weights from', weightsPath);
          const weightsBuffer = await fs.readFile(weightsPath);
          await this.loadWeights(weightsBuffer);
        } else {
          console.log('No weights file found at', weightsPath);
        }
      } catch (weightsError) {
        console.error('Error loading weights:', weightsError);
        throw weightsError;
      }
    } catch (error) {
      console.error('Error loading model:', error);
      throw error;
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

  public dispose(): void {
    tf.engine().startScope();
    try {
      // Dispose memory state
      if (this.memoryState) {
        Object.values(this.memoryState).forEach(tensor => {
          if (tensor && !tensor.isDisposed) {
            tensor.dispose();
          }
        });
      }

      // Dispose transformer stack
      this.transformerStack.forEach(layer => {
        if (layer) {
          layer.dispose();
        }
      });

      // Dispose other components
      if (this.similarityNetwork) {
        this.similarityNetwork.dispose();
      }
      if (this.memoryProjector) {
        this.memoryProjector.dispose();
      }
    } catch (error) {
      console.error('Error disposing model:', error);
      throw error;
    } finally {
      tf.engine().endScope();
    }
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

  public async initialize(config?: Partial<TitanMemoryConfig>): Promise<void> {
    if (config) {
      this.config = ModelConfigSchema.parse(config);
    }
    await this.initializeBackend();
  }

  private async loadWeights(weightsBuffer: Buffer): Promise<void> {
    try {
      // Parse weight data
      const decoder = new TextDecoder();
      let position = 0;
      const weightTensors = new Map<string, tf.Tensor>();

      while (position < weightsBuffer.length) {
        // Find the end of the header
        let headerEnd = weightsBuffer.indexOf(Buffer.from(':'), position);
        if (headerEnd === -1) break;

        // Parse weight name
        const name = decoder.decode(weightsBuffer.slice(position, headerEnd));
        position = headerEnd + 1;

        // Find shape end
        headerEnd = weightsBuffer.indexOf(Buffer.from(':'), position);
        if (headerEnd === -1) break;

        // Parse shape
        const shapeStr = decoder.decode(weightsBuffer.slice(position, headerEnd));
        const shape = shapeStr.split(',').map(s => parseInt(s, 10));

        // Validate shape
        if (shape.some(dim => isNaN(dim) || dim <= 0)) {
          console.warn(`Invalid shape for weight ${name}: ${shapeStr}, skipping`);
          continue;
        }

        position = headerEnd + 1;

        // Calculate data size and validate
        const dataSize = shape.reduce((a, b) => a * b, 1) * 4; // float32 = 4 bytes
        if (position + dataSize > weightsBuffer.length) {
          console.warn(`Insufficient data for weight ${name}, expected ${dataSize} bytes but only ${weightsBuffer.length - position} remaining`);
          break;
        }

        // Create tensor from buffer
        const data = new Float32Array(
          weightsBuffer.buffer.slice(
            weightsBuffer.byteOffset + position,
            weightsBuffer.byteOffset + position + dataSize
          )
        );

        try {
          const tensor = tf.tensor(Array.from(data), shape);
          weightTensors.set(name, tensor);
        } catch (error) {
          console.warn(`Failed to create tensor for weight ${name}:`, error);
          continue;
        }

        position += dataSize;
      }

      // Apply weights to layers
      tf.tidy(() => {
        // Set transformer weights
        this.transformerStack.forEach((layer, layerIndex) => {
          const layerWeights = layer.getWeights().map((originalWeight, weightIndex) => {
            const weightName = `transformer_${layerIndex}_${weightIndex}`;
            const tensor = weightTensors.get(weightName);
            if (!tensor) {
              console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
              return originalWeight;
            }
            if (!tensor.shape.every((dim, i) => dim === originalWeight.shape[i])) {
              console.warn(`Shape mismatch for ${weightName}: expected ${originalWeight.shape}, got ${tensor.shape}, keeping original weights`);
              return originalWeight;
            }
            return tensor;
          });
          layer.setWeights(layerWeights);
        });

        // Set projector weights
        if (this.memoryProjector) {
          const projectorWeights = this.memoryProjector.getWeights().map((originalWeight, weightIndex) => {
            const weightName = `projector_layer_${weightIndex}`;
            const tensor = weightTensors.get(weightName);
            if (!tensor) {
              console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
              return originalWeight;
            }
            if (!tensor.shape.every((dim, i) => dim === originalWeight.shape[i])) {
              console.warn(`Shape mismatch for ${weightName}: expected ${originalWeight.shape}, got ${tensor.shape}, keeping original weights`);
              return originalWeight;
            }
            return tensor;
          });
          this.memoryProjector.setWeights(projectorWeights);
        }

        // Set similarity network weights
        if (this.similarityNetwork) {
          const similarityWeights = this.similarityNetwork.getWeights().map((originalWeight, weightIndex) => {
            const weightName = `similarity_layer_${weightIndex}`;
            const tensor = weightTensors.get(weightName);
            if (!tensor) {
              console.warn(`Missing weight tensor: ${weightName}, keeping original weights`);
              return originalWeight;
            }
            if (!tensor.shape.every((dim, i) => dim === originalWeight.shape[i])) {
              console.warn(`Shape mismatch for ${weightName}: expected ${originalWeight.shape}, got ${tensor.shape}, keeping original weights`);
              return originalWeight;
            }
            return tensor;
          });
          this.similarityNetwork.setWeights(similarityWeights);
        }
      });

      // Clean up any unused tensors
      weightTensors.forEach(tensor => {
        if (!tensor.isDisposed) {
          tensor.dispose();
        }
      });
    } catch (error) {
      console.error('Error loading weights:', error);
      throw error;
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
}