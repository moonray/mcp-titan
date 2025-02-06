# Contributing to Titan Memory MCP Server

First off, thank you for considering contributing to the Titan Memory MCP Server! It's people like you that make it a great tool for everyone.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps which reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include logs from `~/.cursor/titan-memory/logs/`

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- A clear and descriptive title
- A step-by-step description of the suggested enhancement
- Any potential drawbacks or considerations
- Explain why this enhancement would be useful to most users

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs in your pull request whenever possible
- Follow the TypeScript styleguide
- Include thoughtfully-worded, well-structured tests
- Document new code based on the Documentation Styleguide
- End all files with a newline

## Development Process

1. Fork the repo and create your branch from `main`
2. Run `npm install` to install dependencies
3. Make your changes
4. Add tests for any new functionality
5. Ensure the test suite passes (`npm test`)
6. Make sure your code lints (`npm run lint`)
7. Submit your pull request

### Development Setup

```bash
# Clone your fork
git clone git@github.com:your-username/mcp-titan.git

# Add upstream remote
git remote add upstream https://github.com/henryhawke/mcp-titan.git

# Install dependencies
npm install

# Create a branch
git checkout -b feature/my-feature
```

### Testing

```bash
# Run all tests
npm test

# Run specific test file
npm test -- src/__tests__/specific-test.ts

# Run tests in watch mode
npm test -- --watch
```

## Style Guide

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### TypeScript Style Guide

- Use 2 spaces for indentation
- Use camelCase for variables and functions
- Use PascalCase for classes and interfaces
- Use UPPER_CASE for constants
- Always specify return types for functions
- Prefer interfaces over type aliases
- Use readonly where possible
- Add JSDoc comments for public APIs

### Documentation Style Guide

- Use [Markdown](https://guides.github.com/features/mastering-markdown/)
- Reference methods and classes in backticks: \`MyClass.myMethod()\`
- Use code blocks with appropriate language tags
- Keep line length to 80 characters
- Use descriptive link texts: prefer "Read about TensorFlow.js" over "Click here"

## Project Structure

```
.
├── src/                    # Source files
│   ├── __tests__/         # Test files
│   ├── model/             # Model implementation
│   └── types/             # TypeScript type definitions
├── docs/                   # Documentation
├── examples/              # Example implementations
└── scripts/               # Build and maintenance scripts
```

## Additional Notes

### Issue and Pull Request Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## Recognition

Contributors are recognized in several ways:

- Listed in the README.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing to Titan Memory MCP Server!
