# Titan Memory MCP Server

A MCP server built with a three-tier memory architecture that handles storage as follows:

- **Short-term memory:** Holds the immediate conversational context in RAM.
- **Long-term memory:** Persists core patterns and knowledge over time. This state is saved automatically.
- **Meta memory:** Keeps higher-level abstractions that support context-aware responses.

## 🚀 Quick Start

1. Basic Installation (uses default memory path):

```bash
npx -y @smithery/cli@latest run @henryhawke/mcp-titan
```

2. With Custom Memory Path:

```bash
npx -y @smithery/cli@latest run @henryhawke/mcp-titan --config '{
  "memoryPath": "/path/to/your/memory/directory"
}'
```

The server will automatically:

- Initialize in the specified directory (or default location)
- Maintain persistent memory state
- Save model weights and configuration
- Learn from interactions

## 📂 Memory Storage

By default, the server stores memory files in:

- **Windows:** `%APPDATA%\.mcp-titan`
- **MacOS/Linux:** `~/.mcp-titan`

You can customize the storage location using the `memoryPath` configuration:

```bash
# Example with all configuration options
npx -y @smithery/cli@latest run @henryhawke/mcp-titan --config '{
  "port": 3000,
  "memoryPath": "/custom/path/to/memory",
  "inputDim": 768,
  "outputDim": 768
}'
```

The following files will be created in the memory directory:

- `memory.json`: Current memory state
- `model.json`: Model architecture
- `weights/`: Model weights directory

## 🤖 LLM Integration

To integrate with your LLM:

1. Copy the contents of `docs/llm-system-prompt.md` into your LLM's system prompt
2. The LLM will automatically:
   - Use the memory system for every interaction
   - Learn from conversations
   - Provide context-aware responses
   - Maintain persistent knowledge

## 🔄 Automatic Features

- Self-initialization
- WebSocket and stdio transport support
- Automatic state persistence
- Real-time memory updates
- Error recovery and reconnection
- Resource cleanup

## 🧠 Memory Architecture

Three-tier memory system:

- Short-term memory for immediate context
- Long-term memory for persistent patterns
- Meta memory for high-level abstractions

## 🛠️ Configuration Options

| Option       | Description                    | Default        |
| ------------ | ------------------------------ | -------------- |
| `port`       | HTTP/WebSocket port            | `0` (disabled) |
| `memoryPath` | Custom memory storage location | `~/.mcp-titan` |
| `inputDim`   | Size of input vectors          | `768`          |
| `outputDim`  | Size of memory state           | `768`          |

## 📚 Technical Details

- Built with TensorFlow.js
- WebSocket and stdio transport support
- Automatic tensor cleanup
- Type-safe implementation
- Memory-efficient design

## 🔒 Security Considerations

When using a custom memory path:

- Ensure the directory has appropriate permissions
- Use a secure location not accessible to other users
- Consider encrypting sensitive memory data
- Backup memory files regularly

## 📝 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- Built with [Model Context Protocol](https://modelcontextprotocol.io)
- Uses [TensorFlow.js](https://tensorflow.org/js)
- Inspired by [synthience/mcp-titan-cognitive-memory](https://github.com/synthience/mcp-titan-cognitive-memory/)
