import { Server, ServerCapabilities, CallToolRequest, CallToolResult, Transport } from './types.js';
import { z } from 'zod';
export declare const CallToolRequestSchema: z.ZodObject<{
    name: z.ZodString;
    parameters: z.ZodRecord<z.ZodString, z.ZodUnknown>;
}, "strip", z.ZodTypeAny, {
    name: string;
    parameters: Record<string, unknown>;
}, {
    name: string;
    parameters: Record<string, unknown>;
}>;
export declare class MCPServer implements Server {
    name: string;
    version: string;
    capabilities: ServerCapabilities;
    private errorHandler;
    private requestHandler?;
    constructor(name: string, version: string, capabilities: ServerCapabilities);
    onError(handler: (error: Error) => void): void;
    onToolCall(handler: (request: CallToolRequest) => Promise<CallToolResult>, p0: {
        name: any;
        description: any;
        schema: {
            type: string;
            properties: any;
            required: any;
        };
    }, p1: (params: any) => Promise<unknown>): void;
    connect(transport: Transport): Promise<void>;
    handleError(error: Error): void;
    handleRequest(request: CallToolRequest): Promise<CallToolResult>;
    setRequestHandler(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
}
export declare class WebSocketTransport implements Transport {
    private ws;
    private requestHandler?;
    constructor(url: string);
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
}
export declare class StdioTransport implements Transport {
    private requestHandler?;
    private readline;
    constructor();
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
}
export declare class StdioServerTransportImpl implements Transport {
    private requestHandler?;
    onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void;
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    private handleInput;
}
export declare class TitanExpressServer {
    private app;
    private server;
    private model;
    private memoryVec;
    private port;
    constructor(port?: number);
    private setupMiddleware;
    private setupHandlers;
    private setupRoutes;
    start(): Promise<void>;
    stop(): Promise<void>;
}
