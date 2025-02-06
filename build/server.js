import express from 'express';
import bodyParser from 'body-parser';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { CallToolRequestSchema, CallToolResultSchema } from '@modelcontextprotocol/sdk/types.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
export class TitanExpressServer {
    constructor(port = 3000) {
        this.model = null;
        this.memoryVec = null;
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
        const capabilities = {
            tools: {
                listChanged: true,
                list: tools,
                call: {
                    enabled: true,
                    schemas: {
                        request: CallToolRequestSchema,
                        result: CallToolResultSchema
                    }
                }
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
    setupMiddleware() {
        this.app.use(bodyParser.json());
    }
    setupHandlers() {
        this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
            try {
                switch (request.params.name) {
                    case 'storeMemory': {
                        const { subject, relationship, object } = request.params.arguments;
                        // Implementation
                        return {
                            content: [{
                                    type: 'text',
                                    text: JSON.stringify({ stored: true })
                                }]
                        };
                    }
                    case 'recallMemory': {
                        const { query } = request.params.arguments;
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
            }
            catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
                throw new Error(`Error: ${errorMessage}`);
            }
        });
    }
    setupRoutes() {
        this.app.get('/status', (req, res) => {
            res.json({
                status: 'ok',
                model: this.model ? 'initialized' : 'not initialized'
            });
        });
    }
    async start() {
        // Connect stdio transport
        const stdioTransport = new StdioServerTransport();
        await this.server.connect(stdioTransport);
        // Start HTTP server
        return new Promise((resolve, reject) => {
            const server = this.app.listen(this.port, () => {
                console.log(`Server running on port ${this.port}`);
                resolve();
            });
            server.on('error', (error) => {
                reject(error);
            });
        });
    }
    async stop() {
        if (this.memoryVec) {
            this.memoryVec.dispose();
            this.memoryVec = null;
        }
    }
}
//# sourceMappingURL=server.js.map