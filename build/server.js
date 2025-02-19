import express from 'express';
import bodyParser from 'body-parser';
import WebSocket from 'ws';
import { z } from 'zod';
import readline from 'readline';
// Schema definitions
export const CallToolRequestSchema = z.object({
    name: z.string(),
    parameters: z.record(z.unknown())
});
const CallToolResultSchema = z.object({
    success: z.boolean(),
    result: z.unknown().optional(),
    error: z.string().optional()
});
export class MCPServer {
    constructor(name, version, capabilities) {
        this.errorHandler = null;
        this.name = name;
        this.version = version;
        this.capabilities = capabilities;
    }
    onError(handler) {
        this.errorHandler = handler;
    }
    onToolCall(handler, p0, p1) {
        this.requestHandler = handler;
    }
    async connect(transport) {
        await transport.connect();
    }
    handleError(error) {
        if (this.errorHandler) {
            this.errorHandler(error);
        }
        else {
            console.error('Unhandled server error:', error);
        }
    }
    async handleRequest(request) {
        if (!this.requestHandler) {
            return {
                success: false,
                error: 'No request handler registered'
            };
        }
        try {
            // Validate request
            const validatedRequest = CallToolRequestSchema.parse(request);
            return await this.requestHandler(validatedRequest);
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred'
            };
        }
    }
    setRequestHandler(handler) {
        this.requestHandler = handler;
    }
}
export class WebSocketTransport {
    constructor(url) {
        if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
            throw new Error('Invalid WebSocket URL: must start with ws:// or wss://');
        }
        this.ws = new WebSocket(url);
    }
    async connect() {
        return new Promise((resolve, reject) => {
            this.ws.onopen = () => resolve();
            this.ws.onerror = (error) => reject(error);
            this.ws.onmessage = async (event) => {
                if (!this.requestHandler)
                    return;
                try {
                    const request = JSON.parse(event.data.toString());
                    const validatedRequest = CallToolRequestSchema.parse(request);
                    const response = await this.requestHandler(validatedRequest);
                    this.ws.send(JSON.stringify(response));
                }
                catch (error) {
                    this.ws.send(JSON.stringify({
                        success: false,
                        error: error instanceof Error ? error.message : 'Unknown error occurred'
                    }));
                }
            };
        });
    }
    async disconnect() {
        this.ws.close();
        return Promise.resolve();
    }
    onRequest(handler) {
        this.requestHandler = handler;
    }
}
export class StdioTransport {
    constructor() {
        this.readline = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }
    async connect() {
        this.readline.on('line', async (line) => {
            if (!this.requestHandler) {
                console.log(JSON.stringify({
                    success: false,
                    error: 'No handler registered'
                }));
                return;
            }
            try {
                const rawRequest = JSON.parse(line);
                const request = CallToolRequestSchema.parse(rawRequest);
                const response = await this.requestHandler(request);
                console.log(JSON.stringify(response));
            }
            catch (error) {
                console.log(JSON.stringify({
                    success: false,
                    error: error instanceof Error ? error.message : 'Invalid request'
                }));
            }
        });
        return Promise.resolve();
    }
    async disconnect() {
        this.readline.close();
        return Promise.resolve();
    }
    onRequest(handler) {
        this.requestHandler = handler;
    }
}
export class StdioServerTransportImpl {
    onRequest(handler) {
        this.requestHandler = handler;
    }
    async connect() {
        // Setup stdin/stdout handlers
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', this.handleInput.bind(this));
    }
    async disconnect() {
        process.stdin.removeAllListeners('data');
    }
    handleInput(data) {
        if (!this.requestHandler)
            return;
        try {
            const message = JSON.parse(data);
            // Handle message...
            process.stdout.write(JSON.stringify({ type: 'response', data: message }) + '\n');
        }
        catch (error) {
            process.stderr.write(`Error handling input: ${error}\n`);
        }
    }
}
export class TitanExpressServer {
    constructor(port = 3000) {
        this.model = null;
        this.memoryVec = null;
        this.port = port;
        this.app = express();
        this.setupMiddleware();
        this.setupRoutes();
        const capabilities = {
            tools: true,
            memory: true
        };
        this.server = new MCPServer('titan-express', '0.1.0', capabilities);
        this.setupHandlers();
    }
    setupMiddleware() {
        this.app.use(bodyParser.json());
    }
    setupHandlers() {
        this.server.handleRequest = async (request) => {
            try {
                switch (request.name) {
                    case 'storeMemory': {
                        const { subject, relationship, object } = request.parameters;
                        // Implementation
                        return {
                            success: true,
                            result: { stored: true }
                        };
                    }
                    case 'recallMemory': {
                        const { query } = request.parameters;
                        // Implementation
                        return {
                            success: true,
                            result: { results: [] }
                        };
                    }
                    default:
                        throw new Error(`Unknown tool: ${request.name}`);
                }
            }
            catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
                return {
                    success: false,
                    error: `Error: ${errorMessage}`
                };
            }
        };
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
        const stdioTransport = new StdioServerTransportImpl();
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