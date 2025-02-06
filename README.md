# 🧠 MCP Titan - An Advanced Memory Server

[![smithery badge](https://smithery.ai/badge/@henryhawke/mcp-titan)](https://smithery.ai/server/@henryhawke/mcp-titan)

An implementation inspired by Google Research's paper ["Generative AI for Programming: A Common Task Framework"](https://arxiv.org/abs/2501.00663). This server provides a neural memory system that can learn and predict sequences while maintaining state through a memory vector, following principles outlined in the research for improved code generation and understanding.

## 📚 Research Background

A new neural long-term memory module that learns to memorize historical context and helps attention to attend to the current context while utilizing long past information. This neural memory has the advantage of fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory.

This implementation draws from the concepts presented in the Google Research paper (Muennighoff et al., 2024) which introduces a framework for evaluating and improving code generation models. The Titan Memory Server implements key concepts from the paper:

- Memory-augmented sequence learning
- Surprise metric for novelty detection
- Manifold optimization for stable learning
- State maintenance through memory vectors

These features align with the paper's goals of improving code understanding and generation through better memory and state management.

## 🚀 Features

- Neural memory model with configurable dimensions
- Sequence learning and prediction
- Surprise metric calculation
- Model persistence (save/load)
- Memory state management
- Full MCP tool integration

## 📦 Installation

### Installing via Smithery

To install Titan Memory Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@henryhawke/mcp-titan):

```bash
npx -y @smithery/cli install @henryhawke/mcp-titan --client claude
```

### Manual Installation

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test
```

## 🛠️ Available MCP Tools

### 1. 🎯 init_model

Initialize the Titan Memory model with custom configuration.

```typescript
{
  inputDim?: number;  // Input dimension (default: 64)
  outputDim?: number; // Output/Memory dimension (default: 64)
}
```

### 2. 📚 train_step

Perform a single training step with current and next state vectors.

```typescript
{
  x_t: number[];    // Current state vector
  x_next: number[]; // Next state vector
}
```

### 3. 🔄 forward_pass

Run a forward pass through the model with an input vector.

```typescript
{
  x: number[]; // Input vector
}
```

### 4. 💾 save_model

Save the model to a specified path.

```typescript
{
  path: string; // Path to save the model
}
```

### 5. 📂 load_model

Load the model from a specified path.

```typescript
{
  path: string; // Path to load the model from
}
```

### 6. ℹ️ get_status

Get current model status and configuration.

```typescript
{
} // No parameters required
```

### 7. 🔄 train_sequence

Train the model on a sequence of vectors.

```typescript
{
  sequence: number[][]; // Array of vectors to train on
}
```

## 🌟 Example Usage

```typescript
// Initialize model
await callTool("init_model", { inputDim: 64, outputDim: 64 });

// Train on a sequence
const sequence = [
  [1, 0, 0 /* ... */],
  [0, 1, 0 /* ... */],
  [0, 0, 1 /* ... */],
];
await callTool("train_sequence", { sequence });

// Run forward pass
const result = await callTool("forward_pass", {
  x: [1, 0, 0 /* ... */],
});
```

## 🔧 Technical Details

- Built with TensorFlow.js for efficient tensor operations
- Uses manifold optimization for stable learning
- Implements surprise metric for novelty detection
- Memory management with proper tensor cleanup
- Type-safe implementation with TypeScript
- Comprehensive error handling

## 🧪 Testing

The project includes comprehensive tests covering:

- Model initialization and configuration
- Training and forward pass operations
- Memory state management
- Model persistence
- Edge cases and error handling
- Tensor cleanup and memory management

Run tests with:

```bash
npm test
```

## 🔍 Implementation Notes

- All tensor operations are wrapped in `tf.tidy()` for proper memory management
- Implements proper error handling with detailed error messages
- Uses type-safe MCP tool definitions
- Maintains memory state between operations
- Handles floating-point precision issues with epsilon tolerance

## 📝 License

MIT License - feel free to use and modify as needed!

Fixed the Implementation originally done by
https://github.com/synthience/mcp-titan-cognitive-memory/

# Titan Memory MCP Server

A Model Context Protocol (MCP) server implementation that provides automatic memory-augmented learning capabilities for Cursor. This server maintains a persistent memory state that evolves based on interactions, enabling contextual awareness and learning over time.

## Features

- 🧠 Automatic memory management and persistence
- 🔄 Real-time memory updates based on input
- 📊 Memory state analysis and insights
- 🔌 Seamless integration with Cursor via MCP
- 🚀 Dynamic port allocation for HTTP endpoints
- 💾 Automatic state saving every 5 minutes

## Installation

```bash
# Install from npm
npm install @henryhawke/mcp-titan

# Or clone and install locally
git clone https://github.com/henryhawke/mcp-titan.git
cd mcp-titan
npm install
```

## Usage

### As a Cursor MCP Server

1. Add the following to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "titan-memory": {
      "command": "node",
      "args": ["/path/to/mcp-titan/build/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

2. Restart Claude Desktop
3. Look for the hammer icon to confirm the server is connected
4. Use the available tools:
   - `process_input`: Process text and update memory
   - `get_memory_state`: Retrieve current memory insights

### As a Standalone Server

```bash
# Build and start the server
npm run build && npm start

# The server will run on stdio for MCP and start an HTTP server on a dynamic port
```

## Development

### Prerequisites

- Node.js >= 18.0.0
- npm >= 7.0.0

### Setup

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test
```

### Project Structure

```
src/
├── __tests__/        # Test files
├── index.ts          # Main server implementation
├── model.ts          # TitanMemory model implementation
└── types.ts          # TypeScript type definitions
```

### Available Scripts

- `npm run build`: Build the project
- `npm start`: Start the server
- `npm test`: Run tests
- `npm run clean`: Clean build artifacts

## API Reference

### Tools

#### process_input

Process text input and update memory state.

```typescript
interface ProcessInputParams {
  text: string;
  context?: string;
}
```

#### get_memory_state

Retrieve current memory state and statistics.

```typescript
interface MemoryState {
  memoryStats: {
    mean: number;
    std: number;
  };
  memorySize: number;
  status: string;
}
```

### HTTP Endpoints

- `GET /`: Server information and status
- `POST /mcp`: MCP protocol endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting PR
5. Keep PRs focused and atomic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Model Context Protocol](https://modelcontextprotocol.io)
- Uses [TensorFlow.js](https://tensorflow.org/js) for memory operations
