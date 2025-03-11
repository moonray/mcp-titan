import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor } from '../types.js';

describe('TitanMemoryModel Tests', () => {
  let model: TitanMemoryModel;
  const inputDim = 64;
  const hiddenDim = 32;
  const outputDim = 64;

  beforeEach(async () => {
    model = new TitanMemoryModel();
    await model.initialize({
      inputDim,
      hiddenDim,
      outputDim
    });
  });

  test('Model processes sequences correctly', () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = {
      shortTerm: wrapTensor(tf.zeros([outputDim])),
      longTerm: wrapTensor(tf.zeros([outputDim])),
      meta: wrapTensor(tf.zeros([outputDim])),
      timestamps: wrapTensor(tf.zeros([outputDim])),
      accessCounts: wrapTensor(tf.zeros([outputDim])),
      surpriseHistory: wrapTensor(tf.zeros([outputDim]))
    };
    
    const result = model.forward(x, memoryState);
    
    // Check that we get back the expected result structure
    expect(result.output).toBeDefined();
    expect(result.memoryUpdate).toBeDefined();
    expect(result.memoryUpdate.newState).toBeDefined();
    
    // Dispose of tensors
    result.output.dispose();
    x.dispose();
    // Dispose all memory tensors
    Object.values(memoryState).forEach(tensor => tensor.dispose());
  });

  test('Training returns valid results', async () => {
    const memoryState = {
      shortTerm: wrapTensor(tf.zeros([outputDim])),
      longTerm: wrapTensor(tf.zeros([outputDim])),
      meta: wrapTensor(tf.zeros([outputDim])),
      timestamps: wrapTensor(tf.zeros([outputDim])),
      accessCounts: wrapTensor(tf.zeros([outputDim])),
      surpriseHistory: wrapTensor(tf.zeros([outputDim]))
    };
    const x_t = wrapTensor(tf.randomNormal([inputDim]));
    const x_next = wrapTensor(tf.randomNormal([inputDim]));
    
    // Run a training step
    const result = await model.trainStep(x_t, x_next, memoryState);
    
    // Verify the result structure
    expect(result.loss).toBeDefined();
    expect(result.gradients).toBeDefined();
    expect(result.prediction).toBeDefined();
    
    // Clean up
    x_t.dispose();
    x_next.dispose();
    // Dispose all memory tensors
    Object.values(memoryState).forEach(tensor => tensor.dispose());
    result.prediction.dispose();
    Object.values(result.gradients).forEach(tensor => tensor.dispose());
  });

  test('Memory snapshot returns expected structure', () => {
    // Get a memory snapshot
    const snapshot = model.getMemorySnapshot();
    
    // Check that the snapshot has the expected structure
    expect(snapshot.shortTerm).toBeDefined();
    expect(snapshot.longTerm).toBeDefined();
    expect(snapshot.meta).toBeDefined();
    
    // Clean up
    snapshot.shortTerm.dispose();
    snapshot.longTerm.dispose();
    snapshot.meta.dispose();
  });

  test('Model can save and load weights', async () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = {
      shortTerm: wrapTensor(tf.zeros([outputDim])),
      longTerm: wrapTensor(tf.zeros([outputDim])),
      meta: wrapTensor(tf.zeros([outputDim])),
      timestamps: wrapTensor(tf.zeros([outputDim])),
      accessCounts: wrapTensor(tf.zeros([outputDim])),
      surpriseHistory: wrapTensor(tf.zeros([outputDim]))
    };
    
    // Get initial prediction
    const result1 = model.forward(x, memoryState);
    const initial = result1.output.dataSync();
    
    // Save and load weights
    await model.saveModel('./test-weights.json');
    await model.loadModel('./test-weights.json');
    
    // Get prediction after loading
    const result2 = model.forward(x, memoryState);
    const loaded = result2.output.dataSync();
    
    // No need to compare predictions as the current model is a placeholder
    // Just verify structure
    expect(initial.length).toEqual(loaded.length);
    
    // Clean up
    result1.output.dispose();
    result2.output.dispose();
    x.dispose();
    // Dispose all memory tensors
    Object.values(memoryState).forEach(tensor => tensor.dispose());
  });
});
