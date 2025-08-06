# Axonode Chunker

A semantic text chunking package for intelligent document processing that maintains text coherence while preserving page references.

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [With Progress Callback](#with-progress-callback)
  - [With Structural Markers](#with-structural-markers)
- [Configuration Parameters](#configuration-parameters)
  - [Flexible Token Limits](#flexible-token-limits)
    - [Below Minimum Tokens](#below-minimum-tokens)
    - [Above Maximum Tokens](#above-maximum-tokens)
- [Installation](#installation)
- [How It Works](#how-it-works)
  - [Whole-Text vs Page-by-Page Chunking](#whole-text-vs-page-by-page-chunking)
  - [Semantic Chunking with Sliding Window](#semantic-chunking-with-sliding-window)
  - [Dynamic Drift Threshold Mechanism](#dynamic-drift-threshold-mechanism)
  - [Progress Tracking](#progress-tracking)
  - [Custom Structural Markers - A Key Feature](#custom-structural-markers---a-key-feature)
    - [Why Custom Markers Matter](#why-custom-markers-matter)
    - [Marker Types](#marker-types)
    - [Marker Parameters](#marker-parameters)
    - [Practical Examples](#practical-examples)
    - [Marker Removal Strategy](#marker-removal-strategy)
  - [How Semantic and Structural Chunking Work Together](#how-semantic-and-structural-chunking-work-together)
- [Documentation](#documentation)
- [License](#license)

## Key Features

- **Whole-Text Chunking**: Processes the entire document as a continuous text while maintaining proper page references, ensuring better text coherence compared to page-by-page chunking
- **Semantic Chunking**: Uses cosine similarity to detect semantic drift and create coherent chunks
- **Custom Structural Markers**: Design your own markers to guide chunking for augmented text and domain-specific patterns
- **Flexible Token Management**: Guidelines rather than hard constraints, prioritizing content quality over strict limits
- **Async Support**: Full async/await support for processing large documents
- **LangChain Integration**: Works seamlessly with LangChain Document objects

## Quick Start

### Basic Usage

```python
import asyncio
import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from axonode_chunker import AxonodeChunker

async def main():
    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Create chunker
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=500,
        min_tokens=100,
        window_size=3,  # Sliding window size for semantic drift detection
        merge_small_chunks=False  # Preserve semantic boundaries over chunk size consistency
    )
    
    document = Document(
        page_content="Your document text here...",
        metadata={"page": 1}
    )
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="my_document.txt"
    )
    
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")

asyncio.run(main())
```

### With Progress Callback

```python
import asyncio
from axonode_chunker import AxonodeChunker

class ProgressCallback:
    async def send_progress(self, progress_type: str, data: dict):
        if progress_type == 'processing_progress':
            print(f"Progress: {data['progress']:.1f}%")
        elif progress_type == 'status':
            print(f"Status: {data['message']}")

async def main():
    # Initialize chunker and callback
    chunker = AxonodeChunker(embedding_model=embedding_model, tokenizer=tokenizer)
    progress_callback = ProgressCallback()
    
    chunks = await chunker.chunk_documents(
        documents=documents,
        original_file_name="my_document.txt",
        progress_callback=progress_callback
    )

asyncio.run(main())
```

### With Structural Markers

```python
import re
from axonode_chunker import AxonodeChunker, StructuralMarker

# Create custom structural markers
custom_markers = [
    # Headers - high weight for cutting
    StructuralMarker(
        "header", 
        "OPTIONAL_CUT", 
        2.0, 
        re.compile(r"^[A-Z][A-Z\s]+$", re.MULTILINE),
        remove_marker=False
    ),
    
    # Section breaks - medium weight, remove from output
    StructuralMarker(
        "section_break", 
        "OPTIONAL_CUT", 
        1.5, 
        re.compile(r"^[-=]{3,}$", re.MULTILINE),
        remove_marker=True
    ),
    
    # Code blocks - don't cut in the middle
    # Note: Use distinct patterns to avoid race conditions
    StructuralMarker(
        "code_block_start", 
        "HOLD", 
        0.0, 
        re.compile(r"^```[a-z]*$", re.MULTILINE),  # Matches: ```, ```python, ```js, etc.
        remove_marker=False
    ),
    
    StructuralMarker(
        "code_block_end", 
        "RESUME", 
        0.0, 
        re.compile(r"^```$", re.MULTILINE),  # Matches: ``` (exactly)
        remove_marker=False
    ),
]

# Create chunker with custom markers
chunker = AxonodeChunker(
    embedding_model=embedding_model,
    tokenizer=tokenizer,
    max_tokens=400,
    min_tokens=100,
    window_size=3,
    structural_markers=custom_markers
)
```

## Configuration Parameters

- **max_tokens**: Maximum tokens per chunk (default: 500)
- **min_tokens**: Minimum tokens per chunk (default: 100)
- **window_size**: Sliding window size for semantic drift detection (default: 3)
- **structural_markers**: List of custom structural markers (default: empty)
- **merge_small_chunks**: Whether to merge chunks below min_tokens with adjacent chunks (default: False)
- **progress_callback**: Optional callback for tracking processing progress (default: None)

### Flexible Token Limits

The `max_tokens` and `min_tokens` parameters are **guidelines rather than hard constraints**. The chunker prioritizes semantic coherence and structural integrity over strict token limits:

#### Below Minimum Tokens
- Candidates between 0 and `min_tokens` are considered "inferior" but will still be used when necessary
- **Example**: User instructed the script to avoid splitting a large table with a small text section at the beginning. The chunker will cut at the small text (even if below `min_tokens`) to minimize the overall chunk size while preserving the table's integrity

#### Above Maximum Tokens
- Chunks may exceed `max_tokens` when no suitable cut points are available
- **Example**: User instructed the script to avoid splitting a large table or code block that exceeds `max_tokens` will remain intact as a single chunk, even if it's significantly larger than the maximum
- **Example**: Long text sections without semantic breaks or structural markers will be kept together to maintain coherence

#### Small Chunk Handling with `merge_small_chunks`
The `merge_small_chunks` parameter controls how chunks below `min_tokens` are handled:

**When `merge_small_chunks=True`:**
- Small chunks are automatically merged with adjacent chunks to meet minimum token requirements
- Prioritizes consistent chunk sizing over preserving semantic boundaries

**When `merge_small_chunks=False` (default):**
- Small chunks are preserved as-is, even if below `min_tokens`
- Prioritizes semantic and structural boundaries over consistent sizing
- **Use case**: When high-quality semantic/structural boundaries are more important than chunk size consistency
- **Example**: A meaningful section header followed by a short paragraph should remain as a separate chunk rather than being merged with unrelated content

This flexible approach ensures that content quality and structural relationships are preserved, even when it means deviating from the ideal token ranges.

## Installation

```bash
pip install axonode-chunker
```

For running examples, install with example dependencies:
```bash
pip install axonode-chunker[examples]
```

**Note**: You'll need to provide your own embedding model (like SentenceTransformer) and tokenizer (like tiktoken) as parameters to the chunker.

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## How It Works

### Whole-Text vs Page-by-Page Chunking

Unlike traditional chunkers that process each page independently, Axonode Chunker takes the entire document as a continuous text stream. This approach:

- **Maintains Text Coherence**: Prevents artificial breaks at page boundaries that could split related content
- **Preserves Page References**: Still tracks which page each chunk originated from for proper attribution
- **Better Semantic Understanding**: Allows the chunker to understand context across page boundaries
- **Optimal Chunk Placement**: Uses dynamic programming to find the best semantic break points across the entire document

### Semantic Chunking with Sliding Window

The semantic chunker uses a sliding window approach to detect when the topic or theme of the text changes:

- **Window Size**: Configurable window size (default: 3) that determines how many sentences to compare
- **Drift Detection**: Compares the average embedding of the previous window with the current window using cosine similarity
- **Score Calculation**: When semantic drift is detected, the sentence becomes a candidate for chunking with a score proportional to the drift magnitude
- **Dynamic Drift Threshold**: Uses a running average of drift values to establish adaptive thresholds, ensuring only significant semantic changes trigger cuts

### Dynamic Drift Threshold Mechanism

The chunker implements an adaptive threshold system that learns from the document's semantic characteristics:

- **Running Average**: Maintains a continuously updated average of all drift values encountered during processing
- **Adaptive Detection**: Only considers a sentence as a cut candidate if its drift value exceeds the current running average
- **Document-Specific**: Each document develops its own threshold based on its unique semantic patterns
- **Noise Reduction**: Prevents false positives from minor semantic variations while capturing genuine topic shifts

This mechanism ensures that chunking decisions are tailored to each document's specific characteristics rather than using fixed thresholds that might not work well across different types of content.

### Progress Tracking

For long-running document processing tasks, the chunker supports progress tracking through an optional callback mechanism:

- **Progress Updates**: The callback receives percentage completion updates as documents are processed
- **Status Messages**: Descriptive status messages indicating which document is currently being processed
- **Async Design**: The callback is called asynchronously and won't block the chunking process
- **Error Handling**: Progress callback errors are logged but don't interrupt the chunking operation

The progress callback must implement a `send_progress(progress_type: str, data: dict)` async method that receives:
- `progress_type`: Either "processing_progress" or "status"
- `data`: Dictionary containing progress information (percentage for progress, message for status)

### Custom Structural Markers - A Key Feature

One of the most powerful features of Axonode Chunker is the ability to design custom structural markers that guide the text splitting process. This is particularly valuable when working with augmented text or documents with specific formatting requirements.

#### Why Custom Markers Matter

- **Content Preservation**: Keep related content together (e.g., tables with their headers)
- **Template-Based Chunking**: Split at specific patterns like Q&A sections, numbered lists, etc.
- **Augmented Text Handling**: Accurately chunk text that has been enhanced with additional content
- **Domain-Specific Patterns**: Handle industry-specific document structures

#### Marker Types

1. **OPTIONAL_CUT** - Suggests a good place to cut, but doesn't force it
   - **Relevant Parameters**: `name`, `weight`, `pattern`
   - **Weight Usage**: Higher weight increases likelihood of being chosen as a cut point
   - Use cases: Headers, section breaks, list items
   - remove_marker is ignored for OPTIONAL_CUT markers. If you want to remove a line you need to use NO_CUT.

2. **NO_CUT** - Prevents cutting at this location
   - **Relevant Parameters**: `name`, `pattern`, `remove_marker`
   - **Weight Usage**: Not used (weight is ignored for NO_CUT markers)
   - Use cases: Important content that shouldn't be split

3. **HOLD** - Temporarily suppresses all cuts until a RESUME marker is found
   - **Relevant Parameters**: `name`, `pattern`, `remove_marker`
   - **Weight Usage**: Not used (weight is ignored for HOLD markers)
   - Use cases: Code blocks, tables, quotes, or other content that should stay together

4. **RESUME** - Resumes normal cutting behavior after a HOLD marker
   - **Relevant Parameters**: `name`, `pattern`, `remove_marker`
   - **Weight Usage**: Not used (weight is ignored for RESUME markers)
   - Use cases: End of code blocks, quotes, etc.

#### Marker Parameters

- **name**: Descriptive name for the marker
- **type**: One of the four types listed above
- **weight**: Score contribution when this marker is found (only used for OPTIONAL_CUT markers)
- **pattern**: Compiled regex pattern to match the structural element
- **remove_marker**: Whether to remove the matched text from the final chunk (useful for section dividers)

#### Practical Examples

**1. Table Preservation**
```python
# Keep tables and their headers together
table_markers = [
    StructuralMarker("table_start", "HOLD", 0.0, re.compile(r"<table>"), remove_marker=True),
    StructuralMarker("table_end", "RESUME", 0.0, re.compile(r"</table>"), remove_marker=True),
]
```

**2. Q&A Template Chunking**
```python
# Split at question boundaries
qa_markers = [
    StructuralMarker("question", "OPTIONAL_CUT", 1.5, re.compile(r"^Q\d+\.", re.MULTILINE), remove_marker=False),
    StructuralMarker("answer", "OPTIONAL_CUT", 0.8, re.compile(r"^A\d+\.", re.MULTILINE), remove_marker=False),
]
```

**3. Code Block Management**
```python
# Keep code blocks intact and remove markers
code_markers = [
    StructuralMarker("code_start", "HOLD", 0.0, re.compile(r"```\w*"), remove_marker=True),
    StructuralMarker("code_end", "RESUME", 0.0, re.compile(r"```"), remove_marker=True),
]
```

**4. Section Headers with Removal**
```python
# Cut at section headers and remove them from output
section_markers = [
    StructuralMarker("section_header", "OPTIONAL_CUT", 2.0, re.compile(r"^##\s+"), remove_marker=True),
]
```

**5. Page Number Removal**
```python
# Remove page numbers without affecting chunking
page_markers = [
    StructuralMarker("page_number", "NO_CUT", 0.0, re.compile(r"^\d+$", re.MULTILINE), remove_marker=True),
]
```

#### Marker Removal Strategy

The `remove_marker` parameter allows you to clean up the final output:
- **`remove_marker=True`**: Useful for structural elements like `<table>`, ```` ``` ````, section dividers, or page numbers that shouldn't appear in the final chunks
- **`remove_marker=False`**: Keep the marker text when it's meaningful content like headers or question numbers

**Note**: Combining `NO_CUT` with `remove_marker=True` is perfect for cleaning up structural elements like page numbers, footers, or other metadata that you want to remove without affecting the chunking logic.

### How Semantic and Structural Chunking Work Together

The chunker combines both approaches for optimal results:

1. **Candidate Generation**: Both semantic drift and structural markers create cut candidates with scores
2. **Score Aggregation**: Multiple markers can contribute to the same sentence's score
3. **Greedy Selection**: The algorithm selects the highest-scoring candidates that respect token limits
4. **Mutual Enhancement**: Structural markers can reinforce semantic breaks, and semantic drift can validate structural boundaries

## Documentation

- [Installation Guide](INSTALL.md)
- [Examples](examples/)
- [Tests](tests/)

## License

MIT License 