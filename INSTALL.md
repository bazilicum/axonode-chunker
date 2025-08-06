# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/axonode/axonode-chunker.git
cd axonode-chunker
```

2. Install the package in development mode:
```bash
pip install -e .
```

### Install Dependencies

Install the required dependencies:
```bash
pip install -r requirements.txt
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

For running examples:
```bash
pip install -e ".[examples]"
```

For all dependencies:
```bash
pip install -e ".[all]"
```

## Usage

The package uses OpenAI's cl100k_base tokenizer for accurate token counting, which is the same tokenizer used by GPT models.

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
        window_size=3
    )
    
    # Create document
    document = Document(
        page_content="Your text here...",
        metadata={"page": 1}
    )
    
    # Chunk the document
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="my_document.txt"
    )
    
    # Process chunks
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")

asyncio.run(main())
```

### With Structural Markers

```python
import re
import tiktoken
from sentence_transformers import SentenceTransformer
from axonode_chunker import AxonodeChunker, StructuralMarker

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = tiktoken.get_encoding("cl100k_base")

# Create custom markers
markers = [
    StructuralMarker(
        "header", 
        "OPTIONAL_CUT", 
        2.0, 
        re.compile(r"^[A-Z][A-Z\s]+$", re.MULTILINE)
    )
]

chunker = AxonodeChunker(
    embedding_model=embedding_model,
    tokenizer=tokenizer,
    max_tokens=400,
    structural_markers=markers
)
```

## Running Examples

Run the basic example:
```bash
python examples/basic_usage.py
```

Run the structural markers example:
```bash
python examples/structural_markers.py
```

## Running Tests

```bash
pytest
```

## Development

### Code Formatting

```bash
black src/ tests/ examples/
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
flake8 src/ tests/ examples/
``` 