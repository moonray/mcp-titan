/**
 * @fileovertitle Titan Memory Model 2.0 - Neural Memory Architecture with Transformer-XL Inspired Mechanisms
 */

import * as tf from '@tensorflow/tfjs-node';
import { ITensor, IMemoryState, ISurpriseMetrics, IAttentionBlock, IMemoryUpdateResult, IModelGradients, TensorContainer, unwrapTensor, wrapTensor, IMemoryModel } from './types.js';
import * as fs from 'fs/promises';
import { z } from 'zod';

// Helper function to check if a value is null or undefined
function isNullOrUndefined(value: unknown): value is null | undefined {
  return value === null || value === undefined;
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
});

type TitanMemoryConfig = z.infer<typeof ModelConfigSchema>;

export class TitanMemoryModel implements IMemoryModel {
  private config: TitanMemoryConfig;
  private transformerStack: tf.LayersModel[] = [];
  private memoryProjector!: tf.LayersModel;
  private similarityNetwork!: tf.LayersModel;
  private optimizer!: tf.Optimizer;
  private stepCount = 0;
  private vocabulary: Map<string, number>;
  private reverseVocabulary: Map<number, string>;

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
    this.config = ModelConfigSchema.parse(config || {});
    this.vocabulary = new Map();
    this.reverseVocabulary = new Map();
    this.initializeVocabulary();
    this.initializeComponents();
    this.initializeMemoryState();
  }

  private initializeVocabulary(): void {
    // Initialize with special tokens
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
    // Transformer-XL inspired recurrent segment
    this.transformerStack = Array.from({ length: this.config.transformerLayers }, () =>
      tf.sequential({
        layers: [
          tf.layers.dense({
            units: this.config.hiddenDim,
            inputShape: [this.config.inputDim],
            activation: 'linear',
            useBias: true
          }),
          tf.layers.layerNormalization(),
          tf.layers.dense({ units: this.config.ffDimension, activation: 'gelu' }),
          tf.layers.dropout({ rate: this.config.dropoutRate }),
          tf.layers.dense({ units: this.config.hiddenDim }),
          tf.layers.layerNormalization()
        ]
      })
    );

    // Memory projection network
    this.memoryProjector = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.memoryDim,
          inputShape: [this.config.hiddenDim],
          activation: 'tanh'
        }),
        tf.layers.layerNormalization()
      ]
    });

    // Similarity network with contrastive learning
    this.similarityNetwork = tf.sequential({
      layers: [
        tf.layers.dense({
          units: this.config.hiddenDim,
          inputShape: [this.config.memoryDim],
          activation: 'relu'
        }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });

    // Optimizer with gradient clipping
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  private initializeMemoryState(): void {
    this.memoryState = tf.tidy(() => {
      const initializer = tf.initializers.glorotNormal({});
      const memory = initializer.apply([this.config.memorySlots, this.config.memoryDim]);

      return {
        shortTerm: tf.keep(memory as tf.Tensor2D),
        longTerm: tf.keep((memory as tf.Tensor2D).clone()),
        meta: tf.keep(tf.zeros([this.config.memorySlots, this.config.memoryDim])),
        timestamps: tf.keep(tf.zeros([this.config.memorySlots])),
        accessCounts: tf.keep(tf.zeros([this.config.memorySlots])),
        surpriseHistory: tf.keep(tf.zeros([this.config.memorySlots]))
      };
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
      const keys = tf.matMul(this.memoryState.shortTerm, weights[0] as tf.Tensor2D);
      const values = tf.matMul(this.memoryState.shortTerm, weights[1] as tf.Tensor2D);

      const scores = tf.softmax(tf.matMul(query, keys.transpose()));
      const attended = tf.matMul(scores, values);

      return {
        keys,
        values: attended,
        scores
      };
    });
  }

  private computeSurprise(input: tf.Tensor2D, expected: tf.Tensor2D): ISurpriseMetrics {
    return tf.tidy(() => {
      const error = tf.sub(input, expected);
      const immediate = tf.mean(tf.square(error), 1);
      const accumulated = tf.add(
        tf.mul(this.memoryState.surpriseHistory, this.config.surpriseDecay),
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
    const modelData = {
      config: this.config,
      weights: await this.getWeightData(),
      timestamp: Date.now()
    };

    await fs.writeFile(path, JSON.stringify(modelData, null, 2));
  }

  public async save(modelPath: string, weightsPath: string): Promise<void> {
    await this.saveModel(modelPath);
    await fs.writeFile(weightsPath, JSON.stringify(await this.getWeightData(), null, 2));
  }

  private async getWeightData(): Promise<Record<string, number[]>> {
    const weights: Record<string, number[]> = {};

    // Save transformer stack weights
    this.transformerStack.forEach((layer, i) => {
      layer.getWeights().forEach((w, j) => {
        weights[`transformer_${i}_${j}`] = Array.from(w.dataSync());
      });
    });

    // Save other components
    this.memoryProjector.getWeights().forEach((w, i) => {
      weights[`projector_${i}`] = Array.from(w.dataSync());
    });

    this.similarityNetwork.getWeights().forEach((w, i) => {
      weights[`similarity_${i}`] = Array.from(w.dataSync());
    });

    return weights;
  }

  public async loadModel(path: string): Promise<void> {
    try {
      const data = await fs.readFile(path, 'utf8');
      const { config, weights } = JSON.parse(data);

      // Update config
      this.config = ModelConfigSchema.parse(config);

      // Initialize components with new config
      this.initializeComponents();

      // Initialize memory state
      this.initializeMemoryState();

      // Load weights for each component
      Object.entries(weights).forEach(([name, weightData]) => {
        const [componentName, layerIndexStr, weightIndexStr] = name.split('_');
        const layerIndex = parseInt(layerIndexStr, 10);
        const weightIndex = parseInt(weightIndexStr, 10);

        try {
          const tensor = tf.tensor(weightData as number[]);

          switch (componentName) {
            case 'transformer': {
              if (this.transformerStack[layerIndex]) {
                const layer = this.transformerStack[layerIndex];
                const currentWeights = layer.getWeights();
                if (currentWeights && currentWeights[weightIndex]) {
                  const expectedShape = currentWeights[weightIndex].shape;

                  // Reshape tensor to match expected shape
                  const reshapedTensor = tf.tidy(() => {
                    const flattened = tensor.reshape([-1]);
                    return flattened.reshape(expectedShape);
                  });

                  currentWeights[weightIndex] = reshapedTensor;
                  layer.setWeights(currentWeights);
                }
                tensor.dispose();
              }
              break;
            }
            case 'projector': {
              if (this.memoryProjector) {
                const currentWeights = this.memoryProjector.getWeights();
                if (currentWeights && currentWeights[weightIndex]) {
                  const expectedShape = currentWeights[weightIndex].shape;

                  // Reshape tensor to match expected shape
                  const reshapedTensor = tf.tidy(() => {
                    const flattened = tensor.reshape([-1]);
                    return flattened.reshape(expectedShape);
                  });

                  currentWeights[weightIndex] = reshapedTensor;
                  this.memoryProjector.setWeights(currentWeights);
                }
                tensor.dispose();
              }
              break;
            }
            case 'similarity': {
              if (this.similarityNetwork) {
                const currentWeights = this.similarityNetwork.getWeights();
                if (currentWeights && currentWeights[weightIndex]) {
                  const expectedShape = currentWeights[weightIndex].shape;

                  // Reshape tensor to match expected shape
                  const reshapedTensor = tf.tidy(() => {
                    const flattened = tensor.reshape([-1]);
                    return flattened.reshape(expectedShape);
                  });

                  currentWeights[weightIndex] = reshapedTensor;
                  this.similarityNetwork.setWeights(currentWeights);
                }
                tensor.dispose();
              }
              break;
            }
            default:
              tensor.dispose();
          }
        } catch (error) {
          console.error(`Error loading weights for ${name}:`, error);
        }
      });
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
    Object.values(this.memoryState).forEach(t => t.dispose());
    this.transformerStack.forEach(layer => layer.dispose());
    this.similarityNetwork.dispose();
    this.memoryProjector.dispose();
  }
}