import { TitanMemoryServer } from '../index.js';
import { CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';

describe('TitanMemoryServer Tests', () => {
  let server: TitanMemoryServer;

  beforeAll(async () => {
    process.env.NODE_ENV = 'test';
    server = new TitanMemoryServer();
    await server.run();
  });

  afterAll(async () => {
    await server['cleanup']();
    process.env.NODE_ENV = undefined;
  });

  test('Initialize model with config', async () => {
    const config = {
      inputDim: 32,
      outputDim: 32,
    };

    const response = await server['server'].handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'init_model',
        arguments: config
      },
      id: 1
    });

    expect(response.result?.content[0].text).toBeDefined();
    const result = JSON.parse(response.result?.content[0].text);
    expect(result.config).toMatchObject(config);
  });

  test('Training step with valid input', async () => {
    // First initialize the model
    await server['server'].handleRequest({
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

    const x_t = Array(32).fill(0).map(() => Math.random());
    const x_next = Array(32).fill(0).map(() => Math.random());

    const response = await server['server'].handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'train_step',
        arguments: { x_t, x_next }
      },
      id: 1
    });

    expect(response.result?.content[0].text).toBeDefined();
    const result = JSON.parse(response.result?.content[0].text);
    expect(result.cost).toBeDefined();
    expect(result.predicted).toBeDefined();
    expect(result.surprise).toBeDefined();
  });

  test('Forward pass with valid input', async () => {
    const x = Array(32).fill(0).map(() => Math.random());

    const response = await server['server'].handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'forward_pass',
        arguments: { x }
      },
      id: 1
    });

    expect(response.result?.content[0].text).toBeDefined();
    const result = JSON.parse(response.result?.content[0].text);
    expect(result.predicted).toBeDefined();
    expect(result.memory).toBeDefined();
    expect(result.surprise).toBeDefined();
  });

  test('Get memory state', async () => {
    const response = await server['server'].handleRequest({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'get_memory_state',
        arguments: {}
      },
      id: 1
    });

    expect(response.result?.content[0].text).toBeDefined();
    const result = JSON.parse(response.result?.content[0].text);
    expect(result.memoryStats).toBeDefined();
    expect(result.memoryStats.mean).toBeDefined();
    expect(result.memoryStats.std).toBeDefined();
    expect(result.memorySize).toBeDefined();
    expect(result.status).toBe('active');
  });
});
