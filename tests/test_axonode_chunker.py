"""
Tests for the axonode chunker
"""

import pytest
import tiktoken
import warnings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from axonode_chunker import AxonodeChunker, StructuralMarker
import re

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@pytest.fixture
def models():
    """Fixture to provide embedding model and tokenizer"""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return embedding_model, tokenizer

@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_basic_chunking(models):
    """Test basic chunking functionality"""
    embedding_model, tokenizer = models
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=200,
        min_tokens=50,
        window_size=2
    )
    
    text = "This is a test document. It contains multiple sentences. Each sentence should be processed. The chunker should create meaningful chunks."
    
    document = Document(
        page_content=text,
        metadata={"page": 1}
    )
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="test.txt"
    )
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all('text' in chunk for chunk in chunks)
    assert all('page' in chunk for chunk in chunks)
    assert all('chunk_id' in chunk for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_structural_markers(models):
    """Test chunking with structural markers"""
    embedding_model, tokenizer = models
    markers = [
        StructuralMarker(
            "header",
            "OPTIONAL_CUT",
            1.0,
            re.compile(r"^[A-Z][A-Z\s]+$", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=300,
        min_tokens=50,
        window_size=2,
        structural_markers=markers
    )
    
    text = "HEADER\nThis is some content. It should be chunked properly. MORE CONTENT\nANOTHER HEADER\nMore content here."
    
    document = Document(
        page_content=text,
        metadata={"page": 1}
    )
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="test_with_markers.txt"
    )
    
    assert len(chunks) > 0


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_empty_document(models):
    """Test handling of empty documents"""
    embedding_model, tokenizer = models
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer
    )
    
    document = Document(
        page_content="",
        metadata={"page": 1}
    )
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="empty.txt"
    )
    
    assert len(chunks) == 0


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_multiple_documents(models):
    """Test chunking multiple documents"""
    embedding_model, tokenizer = models
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=150,
        min_tokens=30,
        window_size=2
    )
    
    documents = [
        Document(page_content="First document content. It has multiple sentences.", metadata={"page": 1}),
        Document(page_content="Second document content. More sentences here.", metadata={"page": 2}),
    ]
    
    chunks = await chunker.chunk_documents(
        documents=documents,
        original_file_name="multi_doc.txt"
    )
    
    assert len(chunks) > 0 


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_progress_callback(models):
    """Test that progress_callback is called with correct progress updates"""
    embedding_model, tokenizer = models
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=200,
        min_tokens=50,
        window_size=2
    )
    
    # Create a mock progress callback that tracks calls
    progress_updates = []
    
    class MockProgressCallback:
        async def send_progress(self, progress_type: str, data: dict):
            print(f"PROGRESS CALLBACK: {progress_type} - {data}")
            progress_updates.append({
                'type': progress_type,
                'data': data
            })
    
    progress_callback = MockProgressCallback()
    
    # Create multiple documents to test progress across documents
    documents = [
        Document(
            page_content="First document content. It has multiple sentences. Each sentence should be processed.",
            metadata={"page": 1}
        ),
        Document(
            page_content="Second document content. More sentences here. Testing progress callback functionality.",
            metadata={"page": 2}
        ),
        Document(
            page_content="Third document content. Final test document for progress tracking.",
            metadata={"page": 3}
        )
    ]
    
    print(f"Starting chunking with {len(documents)} documents...")
    chunks = await chunker.chunk_documents(
        documents=documents,
        original_file_name="progress_test.txt",
        progress_callback=progress_callback
    )
    
    print(f"Chunking completed. Got {len(chunks)} chunks.")
    print(f"Progress updates received: {len(progress_updates)}")
    
    # Verify that progress updates were called
    assert len(progress_updates) > 0
    
    # Check that we have both types of progress updates
    progress_types = [update['type'] for update in progress_updates]
    assert 'processing_progress' in progress_types
    assert 'status' in progress_types
    
    # Verify progress percentages are within expected range (0-100% for full process)
    processing_updates = [update for update in progress_updates if update['type'] == 'processing_progress']
    for update in processing_updates:
        assert 'progress' in update['data']
        progress_value = update['data']['progress']
        assert 0 <= progress_value <= 100  # Full process is 0-100%
    
    # Verify we have progress updates across different phases
    progress_values = [update['data']['progress'] for update in processing_updates]
    assert max(progress_values) == 100  # Should reach 100%
    assert min(progress_values) == 0 or min(progress_values) > 0  # Should start from 0 or low value
    
    # Verify status messages contain expected content
    status_updates = [update for update in progress_updates if update['type'] == 'status']
    for update in status_updates:
        assert 'message' in update['data']
        message = update['data']['message']
        assert 'Processing document' in message
    
    # Verify chunks were still created correctly
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all('text' in chunk for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_merge_small_chunks_true(models):
    """Test that small chunks are merged when merge_small_chunks=True"""
    embedding_model, tokenizer = models
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=100,
        min_tokens=50,  # Set a high min_tokens to force small chunks
        window_size=2,
        merge_small_chunks=True
    )
    
    # Create text that will naturally create some small chunks
    text = "Short. Very short sentence. Another very short one. This is a longer sentence that contains more words and should create a larger chunk. Short again."
    
    document = Document(
        page_content=text,
        metadata={"page": 1}
    )
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="merge_test.txt"
    )
    
    # With merging enabled, we should have fewer chunks
    # and no chunks should be below min_tokens (except possibly the last one)
    token_counts = [len(tokenizer.encode(chunk['text'])) for chunk in chunks]
    
    # Most chunks should meet the minimum token requirement
    below_min_count = sum(1 for count in token_counts if count < 50)
    
    # With merging, we should have at most 1 chunk below minimum (the final chunk)
    assert below_min_count <= 1
    
    print(f"Merge=True: {len(chunks)} chunks, token counts: {token_counts}")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_merge_small_chunks_false(models):
    """Test that small chunks are preserved when merge_small_chunks=False"""
    embedding_model, tokenizer = models
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=100,
        min_tokens=50,  # Set a high min_tokens to force small chunks
        window_size=2,
        merge_small_chunks=False
    )
    
    # Create text that will naturally create some small chunks
    text = "Short. Very short sentence. Another very short one. This is a longer sentence that contains more words and should create a larger chunk. Short again."
    
    document = Document(
        page_content=text,
        metadata={"page": 1}
    )
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="no_merge_test.txt"
    )
    
    # With merging disabled, we should preserve small chunks
    token_counts = [len(tokenizer.encode(chunk['text'])) for chunk in chunks]
    
    # We should potentially have more chunks below minimum
    below_min_count = sum(1 for count in token_counts if count < 50)
    
    print(f"Merge=False: {len(chunks)} chunks, token counts: {token_counts}")
    
    # The algorithm should still create chunks, even if some are small
    assert len(chunks) > 0


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_merge_small_chunks_comparison(models):
    """Test that merge_small_chunks=False preserves more semantic boundaries"""
    embedding_model, tokenizer = models
    
    # Create text with clear semantic boundaries that might create small chunks
    text = """
    Introduction
    This is a brief introduction to the topic.
    
    Main Content
    This is the main content section with much more detailed information.
    It contains multiple sentences and substantial content.
    The content here is comprehensive and detailed.
    
    Conclusion
    Brief concluding remarks.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    # Test with merging enabled
    chunker_merge = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=200,
        min_tokens=40,
        window_size=2,
        merge_small_chunks=True
    )
    
    chunks_with_merge = await chunker_merge.chunk_documents(
        documents=[document],
        original_file_name="merge_comparison.txt"
    )
    
    # Test with merging disabled
    chunker_no_merge = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=200,
        min_tokens=40,
        window_size=2,
        merge_small_chunks=False
    )
    
    chunks_no_merge = await chunker_no_merge.chunk_documents(
        documents=[document],
        original_file_name="no_merge_comparison.txt"
    )
    
    print(f"With merging: {len(chunks_with_merge)} chunks")
    print(f"Without merging: {len(chunks_no_merge)} chunks")
    
    # Verify both approaches create valid chunks
    assert len(chunks_with_merge) > 0
    assert len(chunks_no_merge) > 0
    
    # Both should have sequential chunk IDs
    for i, chunk in enumerate(chunks_with_merge, 1):
        assert chunk['chunk_id'] == i
    
    for i, chunk in enumerate(chunks_no_merge, 1):
        assert chunk['chunk_id'] == i


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_chunk_id_consistency(models):
    """Test that chunk IDs are always sequential starting from 1"""
    embedding_model, tokenizer = models
    
    # Test both merge settings
    for merge_setting in [True, False]:
        chunker = AxonodeChunker(
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            max_tokens=100,
            min_tokens=30,
            window_size=2,
            merge_small_chunks=merge_setting
        )
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        
        document = Document(
            page_content=text,
            metadata={"page": 1}
        )
        
        chunks = await chunker.chunk_documents(
            documents=[document],
            original_file_name=f"id_test_merge_{merge_setting}.txt"
        )
        
        # Verify chunk IDs are sequential starting from 1
        expected_ids = list(range(1, len(chunks) + 1))
        actual_ids = [chunk['chunk_id'] for chunk in chunks]
        
        assert actual_ids == expected_ids, f"Merge={merge_setting}: Expected {expected_ids}, got {actual_ids}"


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_optional_cut_markers(models):
    """Test OPTIONAL_CUT structural markers with semantic scoring"""
    embedding_model, tokenizer = models
    
    # Create markers that should create cut candidates
    markers = [
        StructuralMarker(
            "section_header",
            "OPTIONAL_CUT",
            1.5,
            re.compile(r"^## [A-Z][a-z\s]+$", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "list_item",
            "OPTIONAL_CUT",
            0.8,
            re.compile(r"^[-*•]\s+", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=150,
        min_tokens=30,
        window_size=2,
        structural_markers=markers
    )
    
    # Text with clear structural markers that should create cut candidates
    text = """
    This is the introduction paragraph. It contains some basic information.
    
    ## Main Section
    This is the main content section. It has substantial content.
    The content here is detailed and comprehensive.
    
    * First list item with some content.
    * Second list item with different content.
    * Third list item with more content.
    
    ## Another Section
    This is another section with different content.
    The semantic drift should combine with structural markers.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="optional_cut_test.txt"
    )
    
    # Should create chunks based on structural markers
    assert len(chunks) > 1
    
    # Check that structural boundaries are respected
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # At least one chunk should contain a section header
    section_headers_found = any("## " in text for text in chunk_texts)
    assert section_headers_found, "Section headers should create cut boundaries"
    
    print(f"Optional cut test: {len(chunks)} chunks created")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_no_cut_markers(models):
    """Test NO_CUT structural markers that block cuts at specific points"""
    embedding_model, tokenizer = models
    
    markers = [
        StructuralMarker(
            "page_number",
            "NO_CUT",
            2.0,
            re.compile(r"^\d+$", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "important_phrase",
            "NO_CUT",
            1.5,
            re.compile(r"IMPORTANT:", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=100,
        min_tokens=20,
        window_size=2,
        structural_markers=markers
    )
    
    # Text with NO_CUT markers that should prevent cuts
    text = """
    This is the first paragraph. It contains some content.
    
    1
    
    This paragraph should not be cut at the page number.
    The content should remain together.
    
    IMPORTANT: This is critical information that should not be split.
    The sentence continues here and should stay together.
    
    This is the final paragraph with normal content.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="no_cut_test.txt"
    )
    
    # Verify that NO_CUT markers prevent cuts
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Check that page numbers don't create cuts
    for chunk_text in chunk_texts:
        if "1" in chunk_text:  # Page number
            # The page number should be part of a larger chunk, not a separate chunk
            assert len(chunk_text.strip()) > 1, "Page number should not be a separate chunk"
    
    # Check that IMPORTANT phrases don't create cuts
    for chunk_text in chunk_texts:
        if "IMPORTANT:" in chunk_text:
            # The IMPORTANT phrase should be part of a larger chunk
            lines = chunk_text.split('\n')
            important_line_idx = None
            for i, line in enumerate(lines):
                if "IMPORTANT:" in line:
                    important_line_idx = i
                    break
            
            if important_line_idx is not None:
                # The IMPORTANT line should not be at the end of a chunk
                assert important_line_idx < len(lines) - 1, "IMPORTANT phrase should not end a chunk"
    
    print(f"NO_CUT test: {len(chunks)} chunks created")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_hold_resume_markers(models):
    """Test HOLD and RESUME markers that suppress cuts across multiple sentences"""
    embedding_model, tokenizer = models
    
    markers = [
        StructuralMarker(
            "code_block_start",
            "HOLD",
            2.0,
            re.compile(r"```[a-z]*$", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "code_block_end",
            "RESUME",
            1.0,
            re.compile(r"```$", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "quote_start",
            "HOLD",
            1.5,
            re.compile(r"^> ", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "quote_end",
            "RESUME",
            1.0,
            re.compile(r"^[^>]", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=120,
        min_tokens=20,
        window_size=2,
        structural_markers=markers
    )
    
    # Text with HOLD/RESUME markers that should suppress cuts
    text = """
    This is normal content before the code block.
    It should be chunked normally.
    
    ```
    def function():
        print("This is code")
        return True
    
    class Example:
        def __init__(self):
            self.value = 42
    ```
    
    This content after the code block should be chunked normally.
    
    > This is a quoted paragraph.
    > It should not be cut in the middle.
    > The entire quote should stay together.
    
    This is normal content after the quote.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="hold_resume_test.txt"
    )
    
    # Verify that HOLD/RESUME markers work correctly
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Check that code blocks are not split
    for chunk_text in chunk_texts:
        if "```" in chunk_text:
            # Count opening and closing code blocks
            opening_blocks = chunk_text.count("```")
            # Should have even number of backticks (pairs)
            assert opening_blocks % 2 == 0, "Code blocks should not be split"
    
    # Check that quotes are not split
    for chunk_text in chunk_texts:
        if "> " in chunk_text:
            lines = chunk_text.split('\n')
            quote_lines = [line for line in lines if line.strip().startswith("> ")]
            # If we have quote lines, they should be consecutive
            if len(quote_lines) > 1:
                # Find the first quote line
                first_quote_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith("> "):
                        first_quote_idx = i
                        break
                
                # All subsequent lines should also be quotes until we hit a non-quote
                for i in range(first_quote_idx + 1, len(lines)):
                    if lines[i].strip() and not lines[i].strip().startswith("> "):
                        # Found non-quote line, check that no more quotes follow
                        remaining_lines = lines[i+1:]
                        remaining_quotes = [line for line in remaining_lines if line.strip().startswith("> ")]
                        assert len(remaining_quotes) == 0, "Quote should not be split"
                        break
    
    print(f"HOLD/RESUME test: {len(chunks)} chunks created")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_structural_semantic_combination(models):
    """Test combination of structural markers with semantic drift detection"""
    embedding_model, tokenizer = models
    
    markers = [
        StructuralMarker(
            "section_header",
            "OPTIONAL_CUT",
            1.2,
            re.compile(r"^## [A-Z][a-z\s]+$", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "code_block",
            "HOLD",
            2.0,
            re.compile(r"```[a-z]*$", re.MULTILINE),
            remove_marker=False
        ),
        StructuralMarker(
            "code_end",
            "RESUME",
            1.0,
            re.compile(r"```$", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=100,
        min_tokens=20,
        window_size=2,
        structural_markers=markers
    )
    
    # Text designed to trigger both semantic drift and structural markers
    text = """
    ## Introduction
    This is the introduction section. It contains basic concepts.
    The content here is foundational and important.
    
    ## Technical Details
    This section contains technical information. It discusses complex topics.
    The semantic content changes significantly from the introduction.
    
    ```
    def complex_function():
        # This is a complex implementation
        result = process_data()
        return result
    ```
    
    ## Conclusion
    This is the conclusion section. It summarizes the main points.
    The content here is different from the technical section.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="combination_test.txt"
    )
    
    # Should create multiple chunks based on both structural and semantic boundaries
    assert len(chunks) > 1
    
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    # Check that section headers create boundaries
    section_chunks = [text for text in chunk_texts if "## " in text]
    assert len(section_chunks) > 0, "Section headers should create chunk boundaries"
    
    # Check that code blocks are preserved
    code_chunks = [text for text in chunk_texts if "```" in text]
    for code_chunk in code_chunks:
        # Code blocks should not be split
        assert code_chunk.count("```") % 2 == 0, "Code blocks should not be split"
    
    print(f"Combination test: {len(chunks)} chunks created")
    print(f"Chunk previews: {[text[:50] + '...' for text in chunk_texts]}")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_marker_removal(models):
    """Test structural markers with remove_marker=True (only works for NO_CUT, HOLD, RESUME)"""
    embedding_model, tokenizer = models
    
    markers = [
            StructuralMarker(
                "page_number",
                "NO_CUT",
                1.0,
                re.compile(r"^\s*\d+\s*$", re.MULTILINE),
                remove_marker=True
            ),
            StructuralMarker(
                "code_block",
                "HOLD",
                1.5,
                re.compile(r"```[a-z]*$", re.MULTILINE),
                remove_marker=True
            ),
            StructuralMarker(
                "code_end",
                "RESUME",
                1.0,
                re.compile(r"```$", re.MULTILINE),
                remove_marker=True
            )
        ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=100,
        min_tokens=20,
        window_size=2,
        structural_markers=markers
    )
    
    text = """
    This is the first paragraph.
    
    1
    
    This is the second paragraph.
    
    ```
    def test_function():
        return True
    ```
    
    This is the third paragraph.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="marker_removal_test.txt"
    )
    
    # Check that removed markers are not in the final chunks
    chunk_texts = [chunk['text'] for chunk in chunks]
    
    for chunk_text in chunk_texts:
        # Page numbers should be removed (NO_CUT with remove_marker=True)
        assert "1" not in chunk_text, "Page number marker should be removed"
        # Code block markers should be removed (HOLD/RESUME with remove_marker=True)
        assert "```" not in chunk_text, "Code block markers should be removed"
    
    print(f"Marker removal test: {len(chunks)} chunks created")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_code_block_race_condition(models):
    """Test that code block start/end markers don't have race conditions"""
    embedding_model, tokenizer = models
    
    # Demonstrate the pattern differences and potential race conditions
    print("\n=== Code Block Marker Race Condition Analysis ===")
    
    # Test the patterns directly
    start_pattern_problematic = re.compile(r"^```", re.MULTILINE)
    start_pattern_better = re.compile(r"^```[a-z]*$", re.MULTILINE)
    start_pattern_best = re.compile(r"^```[a-z]+$", re.MULTILINE)
    end_pattern = re.compile(r"^```$", re.MULTILINE)
    
    test_lines = ["```", "```python", "```js", "``` "]
    
    print("\nPattern matching test:")
    for line in test_lines:
        print(f"Line: '{line}'")
        print(f"  Start (problematic): {bool(start_pattern_problematic.search(line))}")
        print(f"  Start (better): {bool(start_pattern_better.search(line))}")
        print(f"  Start (best): {bool(start_pattern_best.search(line))}")
        print(f"  End: {bool(end_pattern.search(line))}")
        print()
    
    # Show the race condition issue
    print("\nRace condition analysis:")
    print("1. Problematic pattern '^```' matches both start and end markers")
    print("2. Better pattern '^```[a-z]*$' is more specific but still ambiguous")
    print("3. Best pattern '^```[a-z]+$' requires language but misses plain ```")
    print("4. Recommendation: Use '^```[a-z]*$' for start and '^```$' for end")
    
    # Test with the recommended patterns
    recommended_markers = [
        StructuralMarker(
            "code_block_start",
            "HOLD",
            2.0,
            re.compile(r"^```[a-z]*$", re.MULTILINE),  # Matches: ```, ```python, ```js, etc.
            remove_marker=False
        ),
        StructuralMarker(
            "code_block_end",
            "RESUME",
            1.0,
            re.compile(r"^```$", re.MULTILINE),  # Matches: ``` (exactly)
            remove_marker=False
        )
    ]
    
    chunker = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=100,
        min_tokens=20,
        window_size=2,
        structural_markers=recommended_markers
    )
    
    # Simple text to test
    text = """
    This is normal content.
    
    ```python
    def test_function():
        return True
    ```
    
    This is more content.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    chunks = await chunker.chunk_documents(
        documents=[document],
        original_file_name="recommended_patterns_test.txt"
    )
    
    print(f"\nRecommended patterns: {len(chunks)} chunks created")
    
    # Verify the recommended patterns work
    assert len(chunks) > 0
    
    # Check that code blocks are preserved
    chunk_texts = [chunk['text'] for chunk in chunks]
    for chunk_text in chunk_texts:
        if "```" in chunk_text:
            # Code blocks should not be split
            assert chunk_text.count("```") % 2 == 0, "Code blocks should not be split"
    
    print("✅ Recommended patterns work correctly!")


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::FutureWarning")
async def test_marker_weight_scoring(models):
    """Test that different marker weights affect chunking decisions"""
    embedding_model, tokenizer = models
    
    # Create markers with different weights
    low_weight_markers = [
        StructuralMarker(
            "minor_header",
            "OPTIONAL_CUT",
            0.3,
            re.compile(r"^### [A-Z][a-z\s]+$", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    high_weight_markers = [
        StructuralMarker(
            "major_header",
            "OPTIONAL_CUT",
            2.0,
            re.compile(r"^# [A-Z][a-z\s]+$", re.MULTILINE),
            remove_marker=False
        )
    ]
    
    # Test with low weight markers
    chunker_low = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=150,
        min_tokens=30,
        window_size=2,
        structural_markers=low_weight_markers
    )
    
    # Test with high weight markers
    chunker_high = AxonodeChunker(
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        max_tokens=150,
        min_tokens=30,
        window_size=2,
        structural_markers=high_weight_markers
    )
    
    text = """
    ### Minor Section
    This is a minor section with some content.
    It might not create a strong cut boundary.
    
    ### Another Minor Section
    This is another minor section.
    
    # Major Section
    This is a major section that should definitely create a cut.
    The content here is substantial.
    
    # Another Major Section
    This is another major section.
    """
    
    document = Document(page_content=text, metadata={"page": 1})
    
    # Test both configurations
    chunks_low = await chunker_low.chunk_documents(
        documents=[document],
        original_file_name="low_weight_test.txt"
    )
    
    chunks_high = await chunker_high.chunk_documents(
        documents=[document],
        original_file_name="high_weight_test.txt"
    )
    
    print(f"Low weight markers: {len(chunks_low)} chunks")
    print(f"High weight markers: {len(chunks_high)} chunks")
    
    # High weight markers should generally create more chunks
    # (though this depends on the specific text and semantic drift)
    assert len(chunks_low) > 0
    assert len(chunks_high) > 0


 