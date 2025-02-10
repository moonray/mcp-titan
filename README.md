#I'm having a hard time getting this whole MCP server to work with Cursor. I'm not sure why. If anyone has any suggestions, please let me know.
[![smithery badge](https://smithery.ai/badge/@henryhawke/mcp-titan)](https://smithery.ai/server/@henryhawke/mcp-titan)

An implementation inspired by Google Research's paper ["Generative AI for Programming: A Common Task Framework"](https://arxiv.org/abs/2501.00663). This server provides a neural memory system that can learn and predict sequences while maintaining state through a memory vector, following principles outlined in the research for improved code generation and understanding.

## 📚 Research Background

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

## Quick Start

### Installation

You can run the Titan Memory MCP server directly using npx without installing it globally:

```bash
npx -y @smithery/cli@latest run @henryhawke/mcp-titan --config "{}"
```

### Using with Cursor IDE

1. Open Cursor IDE
2. Create or edit your Cursor MCP configuration file:

**MacOS/Linux**: `~/Library/Application Support/Cursor/cursor_config.json`
**Windows**: `%APPDATA%\Cursor\cursor_config.json`

Add the following configuration:

```json
{
  "mcpServers": {
    "titan": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@henryhawke/mcp-titan",
        "--config",
        "{}"
      ]
    }
  }
}
```

3. Restart Cursor IDE

The Titan Memory server will now be available in your Cursor IDE. You can verify it's working by looking for the hammer icon in the bottom right corner of your editor.

### Configuration Options

You can customize the server behavior by passing configuration options in the JSON config string:

```json
{
  "port": 3000 // Optional: HTTP port for REST API (default: 0 - disabled)
  // Additional configuration options can be added here
}
```

Example with custom port:

```bash
npx -y @smithery/cli@latest run @henryhawke/mcp-titan --config '{"port": 3000}'
```

### Available Tools

The Titan Memory server provides the following MCP tools:

1. `init_model` - Initialize the memory model with custom dimensions
2. `train_step` - Train the model on code patterns
3. `forward_pass` - Get predictions for next likely code patterns
4. `get_memory_state` - Query the current memory state and statistics

## 🤖 LLM Integration Guide

When using the Titan Memory server with an LLM (like Claude), include the following information in your prompt or system context to help the LLM effectively use the memory tools:

### Memory System Overview

The Titan Memory server implements a three-tier memory system:

- Short-term memory: For immediate context and recent patterns
- Long-term memory: For persistent patterns and learned behaviors
- Meta memory: For high-level abstractions and relationships

### Tool Usage Guidelines

1. **Initialization**

   ```typescript
   // Always initialize the model first
   await init_model({
     inputDim: 768, // Match your embedding dimension
     outputDim: 768, // Memory state dimension
   });
   ```

2. **Training**

   - Use `train_step` when you have pairs of sequential states
   - Input vectors should be normalized embeddings
   - The surprise metric indicates pattern novelty

3. **Prediction**

   - Use `forward_pass` to predict likely next states
   - Compare predictions with actual outcomes
   - High surprise values indicate unexpected patterns

4. **Memory State Analysis**
   - Use `get_memory_state` to understand current context
   - Monitor memory statistics for learning progress
   - Use memory insights to guide responses

### Example Workflow

1. Initialize model at the start of a session
2. For each new code or text input:
   - Convert to embedding vector
   - Run forward pass to get prediction
   - Use prediction confidence to guide responses
   - Train on actual outcome
   - Check memory state for context

### Best Practices

1. **Vector Preparation**

   - Normalize input vectors to unit length
   - Use consistent embedding dimensions
   - Handle out-of-vocabulary tokens appropriately

2. **Memory Management**

   - Monitor surprise metrics for anomaly detection
   - Use memory state insights to maintain context
   - Consider both short and long-term patterns

3. **Error Handling**
   - Check if model is initialized before operations
   - Handle missing or invalid vectors gracefully
   - Monitor memory usage and performance

### Integration Example

```typescript
// 1. Initialize model
await init_model({ inputDim: 768, outputDim: 768 });

// 2. Process new input
const currentVector = embedText(currentInput);
const { predicted, surprise } = await forward_pass({ x: currentVector });

// 3. Use prediction and surprise for response
if (surprise > 0.8) {
  // Handle unexpected pattern
} else {
  // Use prediction for response
}

// 4. Train on actual outcome
const nextVector = embedText(actualOutcome);
await train_step({ x_t: currentVector, x_next: nextVector });

// 5. Check memory state
const { memoryStats } = await get_memory_state();
```

### Memory Interpretation

The memory state provides several insights:

- Mean activation indicates general memory utilization
- Standard deviation shows pattern diversity
- Memory size reflects context capacity
- Surprise metrics indicate novelty detection

Use these metrics to:

- Gauge confidence in predictions
- Detect context shifts
- Identify learning progress
- Guide response generation

## 📝 LLM Prompt Template

To enable an LLM to effectively use the Titan Memory system, include the following prompt in your system context:

```
You have access to a Titan Memory system that provides advanced memory capabilities for maintaining context and learning patterns. This system uses a three-tier memory architecture and provides the following tools:

1. init_model: Initialize the memory model
   - Required at start of session
   - Parameters: {
       inputDim: number (default: 768),  // Must match your embedding dimension
       outputDim: number (default: 768)   // Size of memory state
     }
   - Call this FIRST before any other memory operations

2. train_step: Train on sequential patterns
   - Parameters: {
       x_t: number[],    // Current state vector (normalized, length = inputDim)
       x_next: number[]  // Next state vector (normalized, length = inputDim)
     }
   - Use to update memory with new patterns
   - Returns: {
       cost: number,     // Training cost
       predicted: number[],  // Predicted next state
       surprise: number     // Novelty metric (0-1)
     }

3. forward_pass: Predict next likely state
   - Parameters: {
       x: number[]  // Current state vector (normalized, length = inputDim)
     }
   - Use to get predictions
   - Returns: {
       predicted: number[],  // Predicted next state
       memory: number[],    // Current memory state
       surprise: number     // Novelty metric (0-1)
     }

4. get_memory_state: Query memory insights
   - Parameters: {} (none required)
   - Returns: {
       memoryStats: {
         mean: number,    // Average activation
         std: number     // Pattern diversity
       },
       memorySize: number,  // Memory capacity
       status: string      // Memory system status
     }

WORKFLOW INSTRUCTIONS:
1. ALWAYS call init_model first in a new session
2. For each interaction:
   - Convert input to normalized vector
   - Use forward_pass to predict and get surprise metric
   - If surprise > 0.8, treat as novel pattern
   - Use predictions to guide your responses
   - Use train_step to update memory with actual outcomes
   - Periodically check memory_state for context

MEMORY INTERPRETATION:
- High surprise (>0.8) indicates unexpected patterns
- Low surprise (<0.2) indicates familiar patterns
- High mean activation (>0.5) indicates strong memory utilization
- High std (>0.3) indicates diverse pattern recognition

You should:
- Initialize memory at session start
- Monitor surprise metrics for context shifts
- Use memory state to maintain consistency
- Consider both short and long-term patterns
- Handle errors gracefully

You must NOT:
- Skip initialization
- Use non-normalized vectors
- Ignore surprise metrics
- Forget to train on outcomes
```

When using this prompt, the LLM will:

1. Understand the complete tool set available
2. Follow the correct initialization sequence
3. Properly interpret memory metrics
4. Maintain consistent memory state
5. Handle errors appropriately

## 🔍 Testing with MCP Inspector

You can test the Titan Memory server using the MCP Inspector tool:

```bash
# Install and run the inspector
npx @modelcontextprotocol/inspector node build/index.js
```

The inspector will be available at http://localhost:5173. You can use it to:

- Test all available tools
- View tool schemas and descriptions
- Monitor memory state
- Debug tool calls
- Verify server responses

### Testing Steps

1. Build the project first:

   ```bash
   npm run build
   ```

2. Make sure the build/index.js file is executable:

   ```bash
   chmod +x build/index.js
   ```

3. Run the inspector:

   ```bash
   npx @modelcontextprotocol/inspector node build/index.js
   ```

4. Open http://localhost:5173 in your browser

5. Test the tools in sequence:
   - Initialize model with `init_model`
   - Train with sample data using `train_step`
   - Test predictions with `forward_pass`
   - Monitor memory state with `get_memory_state`

### Troubleshooting Inspector

If you encounter issues:

1. Ensure Node.js version >= 18.0.0
2. Verify the build is up to date
3. Check file permissions
4. Monitor the terminal for error messages
