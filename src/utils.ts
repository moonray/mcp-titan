import * as tf from '@tensorflow/tfjs-node';
import * as crypto from 'crypto';

export class MemoryManager {
    private static instance: MemoryManager;
    private cleanupInterval: NodeJS.Timeout | null = null;
    private encryptionKey: Buffer;
    private iv: Buffer;

    private constructor() {
        this.encryptionKey = crypto.randomBytes(32);
        this.iv = crypto.randomBytes(16);
        this.startPeriodicCleanup();
    }

    public static getInstance(): MemoryManager {
        if (!MemoryManager.instance) {
            MemoryManager.instance = new MemoryManager();
        }
        return MemoryManager.instance;
    }

    private startPeriodicCleanup(): void {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        this.cleanupInterval = setInterval(() => {
            tf.engine().startScope();
            try {
                // Force garbage collection of unused tensors
                tf.engine().disposeVariables();
                // Clean up tensors
                const numTensors = tf.memory().numTensors;
                if (numTensors > 1000) {
                    const tensors = tf.engine().state.numTensors;
                    tf.disposeVariables();
                    tf.dispose();
                }
            } finally {
                tf.engine().endScope();
            }
        }, 60000); // Run every minute
    }

    public validateVectorShape(tensor: tf.Tensor, expectedShape: number[]): boolean {
        return tf.tidy(() => {
            if (tensor.shape.length !== expectedShape.length) return false;
            return tensor.shape.every((dim, i) => expectedShape[i] === -1 || dim === expectedShape[i]);
        });
    }

    public encryptTensor(tensor: tf.Tensor): Buffer {
        const data = tensor.dataSync();
        const cipher = crypto.createCipheriv('aes-256-gcm', this.encryptionKey, this.iv);
        const encrypted = Buffer.concat([
            cipher.update(Buffer.from(new Float32Array(data).buffer)),
            cipher.final()
        ]);
        const authTag = cipher.getAuthTag();
        return Buffer.concat([encrypted, authTag]);
    }

    public decryptTensor(encrypted: Buffer, shape: number[]): tf.Tensor {
        const authTag = encrypted.slice(-16);
        const encryptedData = encrypted.slice(0, -16);
        const decipher = crypto.createDecipheriv('aes-256-gcm', this.encryptionKey, this.iv);
        decipher.setAuthTag(authTag);
        const decrypted = Buffer.concat([
            decipher.update(encryptedData),
            decipher.final()
        ]);
        const data = new Float32Array(decrypted.buffer);
        return tf.tensor(Array.from(data), shape);
    }

    public wrapWithMemoryManagement<T extends tf.TensorContainer>(fn: () => T): T {
        return tf.tidy(fn);
    }

    public async wrapWithMemoryManagementAsync<T>(fn: () => Promise<T>): Promise<T> {
        tf.engine().startScope();
        try {
            return await fn();
        } finally {
            tf.engine().endScope();
        }
    }

    public dispose(): void {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }
    }
}

export class VectorProcessor {
    private static instance: VectorProcessor;
    private memoryManager: MemoryManager;

    private constructor() {
        this.memoryManager = MemoryManager.getInstance();
    }

    public static getInstance(): VectorProcessor {
        if (!VectorProcessor.instance) {
            VectorProcessor.instance = new VectorProcessor();
        }
        return VectorProcessor.instance;
    }

    /**
     * Process input data into a tensor with appropriate shape
     * Handles various input types: number, array, string, tensor
     */
    public processInput(input: number | number[] | string | tf.Tensor): tf.Tensor {
        return this.memoryManager.wrapWithMemoryManagement(() => {
            try {
                // Handle string input
                if (typeof input === 'string') {
                    return this.encodeText(input);
                }

                // Handle number input (single value)
                if (typeof input === 'number') {
                    return tf.tensor2d([[input]]);
                }

                // Handle array input
                if (Array.isArray(input)) {
                    // Check if it's a 1D or 2D array
                    if (input.length > 0 && Array.isArray(input[0])) {
                        // It's already 2D
                        return tf.tensor2d(input as number[][]);
                    } else {
                        // It's 1D, convert to 2D
                        return tf.tensor2d([input]);
                    }
                }

                // Handle tensor input
                if (input instanceof tf.Tensor) {
                    // Ensure it's 2D
                    if (input.rank === 1) {
                        return input.expandDims(0);
                    } else if (input.rank === 2) {
                        return input;
                    } else {
                        throw new Error(`Unsupported tensor rank: ${input.rank}. Expected 1 or 2.`);
                    }
                }

                throw new Error(`Unsupported input type: ${typeof input}`);
            } catch (error) {
                console.error('Error processing input:', error);
                // Return a default tensor in case of error
                return tf.zeros([1, 768]);
            }
        });
    }

    /**
     * Validate and normalize a tensor to match expected shape
     */
    public validateAndNormalize(tensor: tf.Tensor, expectedShape: number[]): tf.Tensor {
        return this.memoryManager.wrapWithMemoryManagement(() => {
            // Check if tensor shape matches expected shape
            if (!this.memoryManager.validateVectorShape(tensor, expectedShape)) {
                // Reshape tensor to match expected shape
                const reshapedTensor = SafeTensorOps.reshape(tensor, expectedShape);

                // Normalize values to be between -1 and 1
                const normalized = tf.div(reshapedTensor, tf.add(tf.abs(reshapedTensor), 1e-8));

                return normalized;
            }

            return tensor;
        });
    }

    /**
     * Encode text into a tensor representation
     * Uses a simple character-level encoding with positional information
     */
    public async encodeText(text: string, maxLength: number = 768): Promise<tf.Tensor> {
        return this.memoryManager.wrapWithMemoryManagement(() => {
            // Convert string to bytes
            const encoder = new TextEncoder();
            const bytes = encoder.encode(text);

            // Create a padded array of the specified length
            const paddedArray = new Float32Array(maxLength).fill(0);

            // Copy bytes and normalize to [-1, 1]
            for (let i = 0; i < Math.min(bytes.length, maxLength); i++) {
                // Normalize to [-1, 1] range
                paddedArray[i] = (bytes[i] / 127.5) - 1;
            }

            // Add positional encoding
            for (let i = 0; i < Math.min(bytes.length, maxLength); i++) {
                // Add sine wave positional encoding
                const position = i / maxLength;
                paddedArray[i] += Math.sin(position * Math.PI) * 0.1;
            }

            // Create a 2D tensor (batch size 1)
            return tf.tensor2d([paddedArray]);
        });
    }
}

export class AutomaticMemoryMaintenance {
    private static instance: AutomaticMemoryMaintenance;
    private memoryManager: MemoryManager;
    private maintenanceInterval: NodeJS.Timeout | null = null;

    private constructor() {
        this.memoryManager = MemoryManager.getInstance();
        this.startMaintenanceLoop();
    }

    public static getInstance(): AutomaticMemoryMaintenance {
        if (!AutomaticMemoryMaintenance.instance) {
            AutomaticMemoryMaintenance.instance = new AutomaticMemoryMaintenance();
        }
        return AutomaticMemoryMaintenance.instance;
    }

    private startMaintenanceLoop(): void {
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
        }
        this.maintenanceInterval = setInterval(() => {
            this.performMaintenance();
        }, 300000); // Run every 5 minutes
    }

    private performMaintenance(): void {
        this.memoryManager.wrapWithMemoryManagement(() => {
            // Check memory usage
            const memoryInfo = tf.memory();
            if (memoryInfo.numTensors > 1000 || memoryInfo.numBytes > 1e8) {
                tf.engine().disposeVariables();
                const tensors = tf.engine().state.numTensors;
                tf.disposeVariables();
                tf.dispose();
            }
        });
    }

    public dispose(): void {
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
            this.maintenanceInterval = null;
        }
    }
}

// Utility functions for tensor operations
export function checkNullOrUndefined(value: any): boolean {
    return value === null || value === undefined;
}

export function validateTensor(tensor: tf.Tensor | null | undefined): boolean {
    return !checkNullOrUndefined(tensor) && !tensor!.isDisposed;
}

export function validateTensorShape(tensor: tf.Tensor | null | undefined, expectedShape: number[]): boolean {
    if (!validateTensor(tensor)) return false;
    const shape = tensor!.shape;
    if (shape.length !== expectedShape.length) return false;
    return shape.every((dim, i) => expectedShape[i] === -1 || expectedShape[i] === dim);
}

// Safe tensor operations that handle null checks
export class SafeTensorOps {
    static reshape(tensor: tf.Tensor, shape: number[]): tf.Tensor {
        if (!validateTensor(tensor)) {
            throw new Error('Invalid tensor for reshape operation');
        }
        return tf.reshape(tensor, shape);
    }

    static matMul(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for matMul operation');
        }
        return tf.matMul(a, b);
    }

    static add(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for add operation');
        }
        return tf.add(a, b);
    }

    static sub(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for sub operation');
        }
        return tf.sub(a, b);
    }

    static mul(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for mul operation');
        }
        return tf.mul(a, b);
    }

    static div(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
        if (!validateTensor(a) || !validateTensor(b)) {
            throw new Error('Invalid tensors for div operation');
        }
        return tf.div(a, b);
    }
} 