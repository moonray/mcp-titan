import { Transport, CallToolRequest, CallToolResult } from './types.js';
import WebSocket from 'ws';
import * as readline from 'readline';

export class WebSocketTransport implements Transport {
  private ws: WebSocket;
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;

  constructor(url: string) {
    this.ws = new WebSocket(url);
  }

  public async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);

      this.ws.onmessage = async (event) => {
        if (!this.requestHandler) return;

        try {
          const request = JSON.parse(event.data.toString()) as CallToolRequest;
          const response = await this.requestHandler(request);
          this.ws.send(JSON.stringify(response));
        } catch (error) {
          this.ws.send(JSON.stringify({
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error occurred'
          }));
        }
      };
    });
  }

  public async disconnect(): Promise<void> {
    return new Promise<void>((resolve) => {
      this.ws.close();
      resolve();
    });
  }

  public onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }
}

export class StdioTransport implements Transport {
  private requestHandler?: (request: CallToolRequest) => Promise<CallToolResult>;
  private readline: readline.Interface;

  constructor() {
    this.readline = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  async connect(): Promise<void> {
    this.readline.on('line', async (line: string) => {
      if (!this.requestHandler) {
        console.error('No request handler registered');
        return;
      }

      try {
        const request = JSON.parse(line) as CallToolRequest;
        const response = await this.requestHandler(request);
        console.log(JSON.stringify(response));
      } catch (error) {
        console.error('Error handling input:', error);
        console.log(JSON.stringify({
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error occurred'
        }));
      }
    });

    return Promise.resolve();
  }

  async disconnect(): Promise<void> {
    this.readline.close();
    return Promise.resolve();
  }

  onRequest(handler: (request: CallToolRequest) => Promise<CallToolResult>): void {
    this.requestHandler = handler;
  }
} 