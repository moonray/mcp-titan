import { TitanMemoryServer } from '../index.js';

describe('TitanMemoryServer Tests', () => {
  let server: TitanMemoryServer;
  beforeAll(async () => {
    process.env.NODE_ENV = 'test';
    server = new TitanMemoryServer();
    await server.run({ connectTransport: false });
    
    // Initialize the model first with smaller dimensions for testing
    await server.handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'init_model',
        arguments: {
          inputDim: 32,
          outputDim: 32
        }
      },
      id: 1
    });
  });

  afterAll(async () => {
    if (server) {
      try {
        // Call cleanup using the private method accessor
        // We handle exceptions since we expect the cleanup to work even if there are errors
        await server['cleanup']().catch(err => {
          console.warn('Warning during server cleanup:', err instanceof Error ? err.message : String(err));
        });
      } catch (err) {
        console.warn('Warning during server cleanup:', err instanceof Error ? err.message : String(err));
      } finally {
        // Additional cleanup to ensure no open handles
        // Force process.exit not to wait for anything
        jest.clearAllTimers();
        jest.resetAllMocks();
        process.removeAllListeners();
      }
    }
    process.env.NODE_ENV = undefined;
  });

  test('Initialize model with config', async () => {
    const request = {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'init_model',
        arguments: {
          inputDim: 32,
          outputDim: 32
        }
      },
      id: 1
    };

    const response = await server.handleRequest(request);
    expect(response.content[0].text).toBeDefined();
    const result = JSON.parse(response.content[0].text);
    expect(result.config).toMatchObject({
      inputDim: 32,
      outputDim: 32
    });
  });

  test('Training step with valid input', async () => {
    const x_t = Array(32).fill(0).map(() => Math.random());
    const x_next = Array(32).fill(0).map(() => Math.random());

    const response = await server.handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'train_step',
        arguments: { x_t, x_next }
      },
      id: 2
    });

    expect(response.content[0].text).toBeDefined();
    const result = JSON.parse(response.content[0].text);
    expect(result.cost).toBeDefined();
    expect(result.predicted).toBeDefined();
    expect(result.surprise).toBeDefined();
  });

  test('Forward pass with valid input', async () => {
    const x = Array(32).fill(0).map(() => Math.random());

    const response = await server.handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'forward_pass',
        arguments: { x }
      },
      id: 3
    });

    expect(response.content[0].text).toBeDefined();
    const result = JSON.parse(response.content[0].text);
    expect(result.predicted).toBeDefined();
    expect(result.memory).toBeDefined();
    expect(result.surprise).toBeDefined();
  });

  test('Get memory state', async () => {
    const response = await server.handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'get_memory_state',
        arguments: {}
      },
      id: 4
    });

    expect(response.content[0].text).toBeDefined();
    const result = JSON.parse(response.content[0].text);
    expect(result.memoryStats).toBeDefined();
    expect(result.memoryStats.mean).toBeDefined();
    expect(result.memoryStats.std).toBeDefined();
    expect(result.memorySize).toBeDefined();
    expect(result.status).toBe('active');
  });
});
