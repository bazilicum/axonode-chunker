#!/usr/bin/env python3
"""
Advanced usage example for axonode-chunker with structural markers
"""

import asyncio
import re
import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from axonode_chunker import AxonodeChunker, StructuralMarker


async def main():
    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
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
        
        # Section breaks - medium weight
        StructuralMarker(
            "section_break", 
            "OPTIONAL_CUT", 
            1.5, 
            re.compile(r"^[-=]{3,}$", re.MULTILINE),
            remove_marker=True
        ),
        
        # Lists - lower weight, but still a good break point
        StructuralMarker(
            "list_item", 
            "OPTIONAL_CUT", 
            0.5, 
            re.compile(r"^[-*•]\s+", re.MULTILINE),
            remove_marker=False
        ),
        
        # Code blocks - don't cut in the middle
        StructuralMarker(
            "code_block_start", 
            "HOLD", 
            0.0, 
            re.compile(r"^```", re.MULTILINE),
            remove_marker=False
        ),
        
        StructuralMarker(
            "code_block_end", 
            "RESUME", 
            0.0, 
            re.compile(r"^```$", re.MULTILINE),
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
    
    # Example text with various structural elements
    sample_text = """
    PYTHON PROGRAMMING GUIDE
    
    Python is a high-level programming language known for its simplicity and readability.
    
    ===========================================
    
    BASIC SYNTAX
    
    Variables in Python are dynamically typed:
    
    ```python
    x = 10
    y = "Hello, World!"
    z = [1, 2, 3, 4, 5]
    ```
    
    CONTROL STRUCTURES
    
    Python uses indentation for code blocks:
    
    - If statements use if/elif/else
    - Loops include for and while
    - Functions are defined with def
    
    DATA STRUCTURES
    
    Python provides several built-in data structures:
    
    - Lists: ordered, mutable sequences
    - Tuples: ordered, immutable sequences  
    - Dictionaries: key-value mappings
    - Sets: unordered collections of unique elements
    
    OBJECT-ORIENTED PROGRAMMING
    
    Python supports OOP with classes and objects:
    
    ```python
    class Person:
        def __init__(self, name):
            self.name = name
        
        def greet(self):
            return f"Hello, I'm {self.name}"
    ```
    
    ADVANCED FEATURES
    
    Python includes many advanced features:
    
    - List comprehensions for concise data processing
    - Decorators for function modification
    - Context managers for resource management
    - Generators for memory-efficient iteration
    """
    
    # Create document
    document = Document(
        page_content=sample_text,
        metadata={"page": 1}
    )
    
    # Chunk the document
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="python_guide.txt"
    )
    
    # Print results
    print(f"Created {len(chunks)} chunks with structural markers:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Page: {chunk['page']}")
        print(f"  Preview: {chunk['text'][:80]}...")
        print(f"  Length: {len(chunk['text'])} characters")
        print()


if __name__ == "__main__":
    asyncio.run(main()) 