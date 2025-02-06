import { TitanMemoryServer } from '../index.js';

describe('TitanMemoryServer Tests', () => {
  let server: TitanMemoryServer;

  beforeAll(async () => {
    process.env.NODE_ENV = 'test';
    server = new TitanMemoryServer();
    await server.run();
  });

  beforeEach(async () => {
    // Initialize model before each test
    await server.testRequest('init_model', {
      inputDim: 32,
      outputDim: 32,
    });
  });

  afterAll(async () => {
    await server.cleanup();
    process.env.NODE_ENV = undefined;
  });

  test('Initialize model with config', async () => {
    const config = {
      inputDim: 32,
      outputDim: 32,
    };

    const response = await server.testRequest('init_model', config);

    expect(response.content[0].text).toBeDefined();
    const result = JSON.parse(response.content[0].text);
    expect(result.config).toMatchObject(config);
  });

  test('Training step with valid input', async () => {
    const x_t = Array(32).fill(0).map(() => Math.random());
    const x_next = Array(32).fill(0).map(() => Math.random());

    const response = await server.testRequest('train_step', { x_t, x_next });

    expect(response.content[0].text).toBeDefined();
    const result = JSON.parse(response.content[0].text);
    expect(result.cost).toBeDefined();
    expect(result.predicted).toBeDefined();
    expect(result.surprise).toBeDefined();
  });
});
