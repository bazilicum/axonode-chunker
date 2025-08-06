# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Axonode Chunker
- Semantic text chunking with cosine similarity
- Structural markers support for guided chunking
- Token management with OpenAI's cl100k_base tokenizer
- Async/await support for processing large documents
- LangChain integration
- Page tracking functionality
- Configurable token limits and window sizes

### Features
- `AxonodeChunker` class for semantic document chunking
- `StructuralMarker` dataclass for custom chunking rules
- Support for multiple document processing
- Progress callback support
- Comprehensive error handling and logging 