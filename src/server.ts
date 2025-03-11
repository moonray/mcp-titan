/**
 * MCP Server implementation for Titan Memory
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import WebSocket, { WebSocketServer } from 'ws';

/**
 * WebSocket transport implementation for MCP server
 * Note: The SDK doesn't officially support WebSocket transport yet,
 * so this is a custom implementation
 */
export class WebSocketTransport {
  private readonly wss: WebSocketServer;
  private server?: McpServer;

  constructor(port: number) {
    this.wss = new WebSocketServer({ port });
  }

  connect(server: McpServer): void {
    this.server = server;
    
    this.wss.on('connection', (ws: WebSocket) => {
      ws.on('message', (message) => {
        try {
          // Convert WebSocket message data to string based on its type
          let messageStr: string;
          if (Buffer.isBuffer(message)) {
            messageStr = message.toString();
          } else if (message instanceof ArrayBuffer) {
            messageStr = Buffer.from(message).toString();
          } else if (Array.isArray(message)) {
            messageStr = Buffer.concat(message).toString();
          } else {
            messageStr = String(message);
          }
          const requestData = JSON.parse(messageStr);
          console.log('Received message:', typeof requestData === 'object' 
            ? JSON.stringify(requestData, null, 2) 
            : String(requestData));
        } catch (error) {
          console.error('Error processing message:', error instanceof Error ? error.message : String(error));
        }
      });
    });
  }

  close(): void {
    this.wss.close();
  }
}




