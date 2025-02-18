# Titan Memory MCP Server


A MCP server built with a three-tier memory architecture that handles storage as follows:

- **Short-term memory:** Holds the immediate conversational context in RAM.
- **Long-term memory:** Persists core patterns and knowledge over time. This state is saved automatically (using mechanisms built into the TensorFlow.js environment) so that it’s available across server restarts.
- **Meta memory:** Keeps higher-level abstractions that support context-aware responses.

Because the MCP server manages state persistence automatically, you don’t have to manually configure or choose a storage backend—this is handled internally (typically via file-based or other persistent storage methods integrated in the TensorFlow.js environment).

To integrate any LLM with it, the server provides an easy pathway:
 
1. **Copy the System Prompt:** Grab the contents of `docs/llm-system-prompt.md` and place them in your LLM’s system prompt. This instructs your LLM to:
   - Leverage the established memory system for every interaction.
   - Learn from conversations.
   - Deliver context-aware and persistent responses.

2. **Use Supported Transports:** The MCP server supports both WebSocket and stdio transports. Any LLM that can communicate using these protocols (and supports the MCP protocol) can integrate seamlessly.

In summary, memory is stored within the server’s internally managed three-tier system with automatic state persistence. You simply integrate the provided system prompt into your LLM, and the MCP server handles the rest.

## Quick Start

1. Install using Smithery:
```bash
npx -y @smithery/cli install @henryhawke/mcp-titan --client claude
```

2. Start the server:
```bash
npx -y @smithery/cli run @henryhawke/mcp-titan
```

The server will automatically initialize and start learning from interactions.

## Features

- Neural memory model with configurable dimensions
- Automatic sequence learning and prediction
- Real-time surprise metric calculation
- Persistent memory state management
- Automatic error recovery
- Full MCP tool integration
- Model persistence (save/load support)

## Technical Details

- Built with TensorFlow.js for efficient tensor operations
- Implements manifold optimization for stable learning
- Three-tier memory architecture:
  - Short-term memory for immediate context
  - Long-term memory for persistent patterns
  - Meta memory for high-level abstractions
- Type-safe implementation with TypeScript
- Memory-efficient design with automatic tensor cleanup

## Installation

### Using Smithery (Recommended)
```bash
npx -y @smithery/cli install @henryhawke/mcp-titan --client claude
```

### Manual Installation
```bash
git clone https://github.com/henryhawke/mcp-titan.git
cd mcp-titan
npm install
```

## MCP Tools

### Available Tools

1. `init_model` - Initialize memory model
2. `train_step` - Train model on a sequence
3. `forward_pass` - Predict next state
4. `save_model` - Save model to disk
5. `load_model` - Load model from disk
6. `get_status` - Retrieve model configuration
7. `train_sequence` - Train on multiple inputs

### Example Usage
```typescript
// Initialize model
await callTool("init_model", { inputDim: 64, outputDim: 64 });

// Train model on a sequence
await callTool("train_sequence", {
  sequence: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
});

// Run forward pass
const result = await callTool("forward_pass", { x: [1, 0, 0] });
```

## LLM Integration

1. Initialize:
```typescript
await init_model({ inputDim: 768, outputDim: 768 });
```

2. Train:
```typescript
await train_step({ 
  x_t: embedText("input"), 
  x_next: embedText("response") 
});
```

3. Predict:
```typescript
const { predicted, surprise } = await forward_pass({
  x: embedText("current input"),
});
```

## Project Structure
```
src/
├── index.ts          # Main server logic
├── model.ts          # Titan Memory model
├── types.ts          # TypeScript definitions
└── __tests__/        # Unit tests
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to GitHub
5. Open a Pull Request

## License

MIT License - free to use and modify.

## Acknowledgments

- Built with [Model Context Protocol](https://modelcontextprotocol.io)
- Uses [TensorFlow.js](https://tensorflow.org/js)
- Inspired by [synthience/mcp-titan-cognitive-memory](https://github.com/synthience/mcp-titan-cognitive-memory/)
