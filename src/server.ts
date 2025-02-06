import express from 'express';
import bodyParser from 'body-parser';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from './model.js';
import { wrapTensor } from './types.js';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, CallToolResultSchema, ServerCapabilities } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

export class TitanExpressServer {
  private app: express.Application;
  private server: Server;
  private model: TitanMemoryModel | null = null;
  private memoryVec: tf.Variable | null = null;
  private port: number;

  constructor(port: number = 3000) {
    this.port = port;
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();

    const tools = {
      storeMemory: {
        name: 'storeMemory',
        description: 'Stores information in the knowledge graph',
        parameters: {
          type: 'object',
          properties: {
            subject: { type: 'string' },
            relationship: { type: 'string' },
            object: { type: 'string' }
          },
          required: ['subject', 'relationship', 'object']
        }
      },
      recallMemory: {
        name: 'recallMemory',
        description: 'Recalls information from the knowledge graph',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string' }
          },
          required: ['query']
        }
      }
    };

    const capabilities: ServerCapabilities = {
      tools: {
        listChanged: true,
        list: tools
      }
    };

    this.server = new Server({
      name: 'titan-express',
      version: '0.1.0',
      description: 'Titan Memory Express Server',
      capabilities
    });

    this.setupHandlers();
  }

  private setupMiddleware() {
    this.app.use(bodyParser.json());
  }

  private setupHandlers() {
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        switch (request.params.name) {
          case 'storeMemory': {
            const { subject, relationship, object } = request.params.arguments as any;
            // Implementation
            return {
              content: [{
                type: 'text',
                text: JSON.stringify({ stored: true })
              }]
            };
          }
          case 'recallMemory': {
            const { query } = request.params.arguments as any;
            // Implementation
            return {
              content: [{
                type: 'text',
                text: JSON.stringify({ results: [] })
              }]
            };
          }
          default:
            throw new Error(`Unknown tool: ${request.params.name}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
        throw new Error(`Error: ${errorMessage}`);
      }
    });
  }

  private setupRoutes() {
    this.app.get('/status', (req: express.Request, res: express.Response) => {
      res.json({
        status: 'ok',
        model: this.model ? 'initialized' : 'not initialized'
      });
    });
  }

  public async start(): Promise<void> {
    // Connect stdio transport
    const stdioTransport = new StdioServerTransport();
    await this.server.connect(stdioTransport);

    // Start HTTP server
    return new Promise((resolve, reject) => {
      const server = this.app.listen(this.port, () => {
        console.log(`Server running on port ${this.port}`);
        resolve();
      });

      server.on('error', (error: Error) => {
        reject(error);
      });
    });
  }

  public async stop(): Promise<void> {
    if (this.memoryVec) {
      this.memoryVec.dispose();
      this.memoryVec = null;
    }
  }
}
