import * as readline from 'readline';
import WebSocket from 'ws';
import { z } from 'zod';
/**
 * WebSocket transport implementation for MCP server
 */
export class WebSocketTransport {
    url;
    ws;
    requestHandler;
    constructor(url) {
        this.url = url;
        this.ws = new WebSocket(url);
    }
    async connect() {
        return new Promise((resolve, reject) => {
            this.ws.on('open', resolve);
            this.ws.on('error', reject);
            this.ws.on('message', async (data) => {
                if (!this.requestHandler)
                    return;
                try {
                    const request = JSON.parse(data.toString());
                    const result = await this.requestHandler(request);
                    this.ws.send(JSON.stringify(result));
                }
                catch (error) {
                    this.ws.send(JSON.stringify({
                        success: false,
                        error: error instanceof Error ? error.message : 'Invalid request'
                    }));
                }
            });
        });
    }
    async disconnect() {
        this.ws.close();
    }
    onRequest(handler) {
        this.requestHandler = handler;
    }
    send(message) {
        this.ws.send(JSON.stringify(message));
    }
}
/**
 * Standard I/O transport implementation for MCP server
 */
export class StdioServerTransport {
    rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    requestHandler;
    async connect() {
        this.rl.on('line', async (line) => {
            if (!this.requestHandler)
                return;
            try {
                const request = JSON.parse(line);
                const result = await this.requestHandler(request);
                console.log(JSON.stringify(result));
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
        this.rl.close();
    }
    onRequest(handler) {
        this.requestHandler = handler;
    }
    send(message) {
        console.log(JSON.stringify(message));
    }
}
/**
 * Enhanced MCP Server Implementation
 */
export class McpServerImpl {
    tools = new Map();
    name;
    version;
    constructor(config) {
        this.name = config.name;
        this.version = config.version;
    }
    tool(name, schema, handler) {
        const zodSchema = typeof schema === 'string'
            ? z.object({})
            : z.object(schema);
        this.tools.set(name, async (params) => {
            const validated = zodSchema.parse(params);
            return handler(validated);
        });
    }
    async connect(transport) {
        transport.onRequest(async (request) => {
            const handler = this.tools.get(request.name);
            if (!handler) {
                return {
                    success: false,
                    error: `Tool ${request.name} not found`,
                    content: [{
                            type: "text",
                            text: `Tool ${request.name} not found`
                        }]
                };
            }
            try {
                return await handler(request.parameters);
            }
            catch (error) {
                return {
                    success: false,
                    error: error instanceof Error ? error.message : 'Unknown error',
                    content: [{
                            type: "text",
                            text: error instanceof Error ? error.message : 'Unknown error'
                        }]
                };
            }
        });
        await transport.connect();
        // Send server info
        transport.send?.({
            jsonrpc: "2.0",
            method: "server_info",
            params: {
                name: this.name,
                version: this.version,
                capabilities: {
                    tools: true,
                    memory: true
                }
            }
        });
    }
}
