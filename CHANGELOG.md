# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-08

### Added

- Initial release as a reusable Go package
- High-level `Agent` API with state management and event subscription
- Low-level `AgentLoop` function for direct control over execution
- Event-driven architecture with streaming support
- Tool execution framework with JSON schema validation
- Steering mechanism to interrupt agent mid-execution
- Follow-up message queuing for sequential processing
- Thread-safe operations with proper mutex handling
- Support for multiple LLM providers via go-pi-ai:
  - NVIDIA API
  - OpenAI API
  - Custom providers
- Comprehensive documentation:
  - Package-level godoc (doc.go)
  - API documentation for all exported types and functions
  - README with quick start guide and examples
- Example code in examples/basic/

### Changed

- Moved core code from `internal/agent/` to root package for external import
- Package is now importable as `github.com/rahulSailesh-shah/go-pi-agent`

### Removed

- Removed `main.go` entry point (moved to examples/)
- Removed `internal/` directory structure
