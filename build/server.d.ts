import { Transport, CallToolRequest, CallToolResult, McpServer } from './types.js';
import { z } from 'zod';
/**
 * WebSocket transport implementation for MCP server
 */
export declare class WebSocketTransport implements Transport {
    private url;
    private ws;
    private requestHandler?;
    constructor(url: string);
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
    send(message: unknown): void;
}
/**
 * Standard I/O transport implementation for MCP server
 */
export declare class StdioServerTransport implements Transport {
    private rl;
    private requestHandler?;
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
    send(message: unknown): void;
}
/**
 * Enhanced MCP Server Implementation
 */
export declare class McpServerImpl implements McpServer {
    private tools;
    private name;
    private version;
    constructor(config: {
        name: string;
        version: string;
    });
    tool(name: string, schema: z.ZodRawShape | string, handler: Function): void;
    connect(transport: Transport): Promise<void>;
}
