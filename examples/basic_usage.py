#!/usr/bin/env python3
"""
Basic usage example for axonode-chunker
"""

import asyncio
import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from axonode_chunker import AxonodeChunker, StructuralMarker
import re


async def main():
    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Create a semantic chunker with default settings
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=500,
        min_tokens=50,
        window_size=3
    )
    
    # Example text to chunk
    sample_text = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.
    
    Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised Learning
    
    In supervised learning, the algorithm is trained on labeled data. The goal is to learn a mapping from inputs to outputs.
    
    Unsupervised Learning
    
    Unsupervised learning deals with unlabeled data. The algorithm tries to find hidden patterns or structures in the data.
    
    Reinforcement Learning
    
    Reinforcement learning is about training agents to make sequences of decisions. The agent learns by interacting with an environment.
    
    Applications
    
    Machine learning has applications in various fields including computer vision, natural language processing, and robotics.
    """
    
    # Create a document
    document = Document(
        page_content=sample_text,
        metadata={"page": 1}
    )
    
    # Chunk the document
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="machine_learning_intro.txt"
    )
    
    # Print results
    print(f"Created {len(chunks)} chunks:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Page: {chunk['page']}")
        print(f"  Text: {chunk['text'][:100]}...")
        print(f"  Length: {len(chunk['text'])} characters")
        print()


if __name__ == "__main__":
    asyncio.run(main()) 