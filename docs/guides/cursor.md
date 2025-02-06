# Cursor IDE Integration Guide

## Overview

This guide explains how to integrate the Titan Memory MCP Server with Cursor IDE for enhanced coding capabilities through memory-augmented learning.

## Prerequisites

- Cursor IDE (latest version)
- Node.js >= 18.0.0
- npm >= 7.0.0

## Installation

### 1. Install the Server

```bash
npm install -g @henryhawke/mcp-titan
```

### 2. Configure Cursor

1. Open Cursor IDE
2. Press `Cmd/Ctrl + Shift + P`
3. Type "Open Settings (JSON)"
4. Add the following configuration:

```json
{
  "mcp": {
    "servers": {
      "titan-memory": {
        "command": "mcp-titan",
        "env": {
          "NODE_ENV": "production",
          "DEBUG": "mcp-titan*"
        }
      }
    }
  }
}
```

### 3. Initialize the Server

1. Press `Cmd/Ctrl + Shift + P`
2. Type "MCP: Restart Servers"
3. Look for the hammer icon in the status bar to confirm connection

## Usage in Cursor

### Available Commands

Access these commands through `Cmd/Ctrl + Shift + P`:

- `MCP: Process Input` - Process current selection or file
- `MCP: Get Memory State` - View current memory insights
- `MCP: Restart Servers` - Restart MCP servers
- `MCP: Show Server Status` - Check server connection status

### Keyboard Shortcuts

Add these to your keybindings.json:

```json
{
  "key": "cmd+shift+m",
  "command": "mcp.processInput",
  "when": "editorTextFocus"
}
```

### Context Menu Integration

Right-click in editor to access:

- Process Selection with Titan Memory
- Get Memory Insights
- Clear Memory State

## Features in Cursor

### 1. Automatic Learning

The server automatically:

- Learns from your code as you type
- Builds context awareness
- Maintains persistent memory

### 2. Intelligent Suggestions

Provides:

- Context-aware code completions
- Relevant documentation
- Similar code patterns

### 3. Memory Insights

Access insights about:

- Code patterns
- Documentation relevance
- Context relationships

## Troubleshooting

### Common Issues

1. Server Not Connecting

   ```bash
   # Check server status
   cursor --mcp-status

   # View logs
   tail -f ~/.cursor/titan-memory/logs/server.log
   ```

2. Memory Not Updating

   ```bash
   # Reset memory state
   rm ~/.cursor/titan-memory/memory.json
   # Restart server
   cursor --mcp-restart
   ```

3. Performance Issues
   ```bash
   # Enable performance logging
   DEBUG=mcp-titan:performance* cursor
   ```

### Debug Mode

Enable detailed logging:

1. Add to settings.json:

   ```json
   {
     "mcp.debug": true,
     "mcp.logLevel": "debug"
   }
   ```

2. View logs:
   ```bash
   tail -f ~/.cursor/logs/mcp.log
   ```

## Best Practices

1. Memory Management

   - Regularly clear unused memory
   - Monitor memory size
   - Use context when processing input

2. Performance

   - Process larger files in chunks
   - Use selective processing for large projects
   - Clear memory state periodically

3. Integration
   - Use with Cursor's AI features
   - Combine with other MCP servers
   - Leverage keyboard shortcuts

## Support

- GitHub Issues: [Report bugs](https://github.com/henryhawke/mcp-titan/issues)
- Documentation: [Full API docs](./API.md)
- Community: [Cursor Discord](https://discord.gg/cursor)
