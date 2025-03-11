/**
 * Transport implementation for MCP - These are primarily provided for backward compatibility
 * or for custom transport needs not provided by the standard SDK.
 */
// Using SDK transport interfaces
// No need to import Transport as it's not directly used
import { WebSocket, MessageEvent } from 'ws';
import * as readline from 'readline';
import { z } from 'zod';

// JSON-RPC message types
export interface JsonRpcRequest {
  jsonrpc: string;
  method: string;
  params?: unknown;
  id?: string | number;
}

export interface JsonRpcResponse {
  jsonrpc: string;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
  id: string | number | null;
}

export interface JsonRpcNotification {
  jsonrpc: string;
  method: string;
  params?: unknown;
}

// Define error codes
export enum ErrorCode {
  ParseError = -32700,
  InvalidRequest = -32600,
  MethodNotFound = -32601,
  InvalidParams = -32602,
  InternalError = -32603
}

// Define request schema for validation
export const CallToolRequestSchema = z.object({
  jsonrpc: z.string(),
  method: z.string(),
  params: z.object({
    name: z.string(),
    arguments: z.record(z.unknown())
  }),
  id: z.number()
});

/**
 * WebSocket transport implementation 
 * Note: This is a custom implementation not provided by the SDK
 */
export class WebSocketTransport {
  private readonly ws: WebSocket;
  private messageHandler?: (message: JsonRpcRequest) => Promise<void>;
  private readonly responseHandlers = new Map<string | number, (response: JsonRpcResponse) => void>();

  constructor(url: string) {
    this.ws = new WebSocket(url);
  }

  public async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(new Error(error instanceof Error ? error.message : JSON.stringify(error)));

      this.ws.onmessage = (event) => {
        void this.handleMessage(event);
      };
    });
  }

  private async handleMessage(event: MessageEvent): Promise<void> {
    if (!event.data) {
      console.error('Received empty message');
      return;
    }
    if (!this.messageHandler) return;
    try {
      // Ensure proper string conversion of the message data
      const data = typeof event.data === 'object' ? JSON.stringify(event.data) : String(event.data);
      const message = JSON.parse(data) as JsonRpcRequest;
      
      // Handle response messages
      if ('result' in message || 'error' in message) {
        const responseHandler = this.responseHandlers.get(message.id!);
        if (responseHandler) {
          responseHandler(message as unknown as JsonRpcResponse);
          this.responseHandlers.delete(message.id!);
        }
      } 
      // Handle request/notification messages
      else if ('method' in message) {
        await this.messageHandler(message);
      }
    } catch (error) {
      console.error('Error processing message:', error instanceof Error ? error.message : String(error));
    }
  }

  public async close(): Promise<void> {
    this.ws.close();
    return Promise.resolve();
  }

  public onMessage(handler: (message: JsonRpcRequest) => Promise<void>): void {
    this.messageHandler = handler;
  }

  public async send(message: JsonRpcResponse | JsonRpcRequest | JsonRpcNotification): Promise<void> {
    try {
      this.ws.send(JSON.stringify(message));
    } catch (error: unknown) {
      console.error('Failed to send message:', error instanceof Error ? error.message : error);
      throw error;
    }
  }
}

/**
 * StdioTransport implementation 
 * Note: This is provided for backward compatibility but the SDK's StdioServerTransport
 * should be preferred in most cases.
 */
export class StdioTransport {
  private messageHandler?: (message: JsonRpcRequest) => Promise<void>;
  private readonly readline: readline.Interface;
  private readonly nextId = 1;
  private readonly responseHandlers = new Map<string | number, (response: JsonRpcResponse) => void>();

  constructor() {
    this.readline = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  public async start(): Promise<void> {
    this.readline.on('line', async (line: string) => {
      if (!this.messageHandler) {
        console.error('No message handler registered');
        return;
      }

      try {
        const parsed = JSON.parse(line);
        const validationResult = CallToolRequestSchema.safeParse(parsed);
        
        if (!validationResult.success) {
          console.error('Invalid message format:', validationResult.error);
          if ('id' in parsed) {
            console.log(JSON.stringify({
              jsonrpc: '2.0',
              error: {
                code: ErrorCode.InvalidRequest,
                message: 'Invalid request format',
                data: validationResult.error
              },
              id: parsed.id
            }));
          }
          return;
        }
        
        const message = validationResult.data as JsonRpcRequest;
        
        // If it's a request with an ID, prepare to handle the response
        if ('id' in message) {
          const responsePromise = this.messageHandler(message);
          
          // The message handler should send the response
          responsePromise.catch(error => {
            console.error('Error handling message:', error);
            // Send error response in case of exception
            console.log(JSON.stringify({
              jsonrpc: '2.0',
              error: {
                code: ErrorCode.InternalError, 
                message: error instanceof Error ? error.message : 'Unknown error'
              },
              id: message.id
            }));
          });
        } 
        // Otherwise just process the notification
        else {
          this.messageHandler(message).catch(error => {
            console.error('Error handling notification:', error);
          });
        }
      } catch (error) {
        console.error('Error parsing input:', error);
      }
    });

    return Promise.resolve();
  }

  public async close(): Promise<void> {
    this.readline.close();
    return Promise.resolve();
  }

  public onMessage(handler: (message: JsonRpcRequest) => Promise<void>): void {
    this.messageHandler = handler;
  }

  public async send(message: JsonRpcResponse | JsonRpcRequest | JsonRpcNotification): Promise<void> {
    try {
      console.log(JSON.stringify(message));
    } catch (error: unknown) {
      console.error('Failed to send message:', error instanceof Error ? error.message : error);
      throw error;
    }
  }
}
