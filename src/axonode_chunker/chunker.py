import re
import numpy as np
import logging
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Literal, Callable
from langchain_core.documents import Document

# Setup logging
logger = logging.getLogger(__name__)

# Example usage of StructuralMarker:
# 
# # Create custom structural markers
# custom_markers = [
#     StructuralMarker("custom_heading", 1.5, re.compile(r"^[A-Z][A-Z\s]+$", re.MULTILINE)),
#     StructuralMarker("list_item", 0.3, re.compile(r"^[-*•]\s+", re.MULTILINE)),
# ]
# 
# # Use with AxonodeChunker
# chunker = AxonodeChunker(structural_markers=custom_markers)
# 
# # Or use default markers
# chunker = AxonodeChunker()  # Uses default structural markers

@dataclass
class StructuralMarker:
    """Data class for structural markers with name, weight, and compiled regex pattern.
    Args:
        name: Name of the structural marker
        type: Type of the structural marker: OPTIONAL_CUT, NO_CUT, HOLD, RESUME
        weight: Weight of the structural marker
        pattern: Compiled regex pattern for the structural marker
    """
    name: str
    type: Literal["OPTIONAL_CUT", "NO_CUT", "HOLD", "RESUME"]
    weight: float
    pattern: re.compile
    remove_marker: bool = False


class PageTracker:
    """Tracks page numbers for character positions in continuous text."""
    def __init__(self):
        self.page_boundaries: List[Tuple[int, int]] = []
        self.current_page = 1

    def add_page_end(self, char_position: int):
        self.page_boundaries.append((char_position, self.current_page))
        self.current_page += 1

    def get_page_for_position(self, char_position: int) -> int:
        for boundary_pos, page_num in self.page_boundaries:
            if char_position <= boundary_pos:
                return page_num
        return self.current_page


class AxonodeChunker:
    """
    Streams chunked text page-by-page, records semantic drift candidates,
    then uses DP to choose the optimal thematic breaks under token constraints.
    """
    def __init__(self,
                 embedding_model: Any,
                 tokenizer: Any,
                 max_tokens: int = 500,
                 min_tokens: int = 100,
                 window_size: int = 3,
                 structural_markers: Optional[List[StructuralMarker]] = None,
                 merge_small_chunks: bool = False):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_model: SentenceTransformer model or similar embedding model
            tokenizer: Tokenizer object (e.g., tiktoken encoder)
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk
            window_size: Window size for semantic drift detection
            structural_markers: List of structural markers for guided chunking
            merge_small_chunks: Whether to merge chunks below min_tokens (default: False)
        """
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.window_size = window_size
        self.merge_small_chunks = merge_small_chunks
        
        self.cut_suppressed = False
        
        # Use provided structural markers or default to empty dictionary with expected structure
        if structural_markers:
            self.structural_markers = self._sort_structural_markers(structural_markers)
        else:
            self.structural_markers = {"OPTIONAL_CUT": [], "NO_CUT": [], "HOLD": [], "RESUME": []}

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for given text using the provided embedding model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding as numpy array
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        
        embedding = self.embedding_model.encode([text], convert_to_tensor=False)
        return embedding[0]

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the provided tokenizer.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        return len(self.tokenizer.encode(text))

    def _sort_structural_markers(self, structural_markers: List[StructuralMarker]) -> Dict[str, List[StructuralMarker]]:
        """
        Sort structural markers by weight and type.
        """
        sorted_markers = {"OPTIONAL_CUT": [], "NO_CUT": [], "HOLD": [], "RESUME": []}
        for marker in structural_markers:
            sorted_markers[marker.type].append(marker)
        return sorted_markers
    
    def _add_structural_cut_candidates(self,
                                   sentence_text: str,
                                   sentence_idx: int) -> Dict[int, float]:
        """
        Return a mapping of candidate sentence indexes to their aggregated structural scores.
        """
        candidates: Dict[int, float] = {}

        sentence_text = sentence_text.strip()

        def add(marker: StructuralMarker):
            candidates[sentence_idx] = candidates.get(sentence_idx, 0.0) + marker.weight

        # Check each structural marker against the sentence
        for marker in self.structural_markers["OPTIONAL_CUT"]:
            if marker.pattern.search(sentence_text):
                add(marker)
                logger.debug(f"STRUCTURAL: Found marker '{marker.name}' at sentence {sentence_idx} (weight={marker.weight:.3f})")

        return candidates

    def _check_for_structural_cuts_suppression(self, sentence_text: str) -> Tuple[Literal["NO_CUT", "HOLD", "RESUME", "NONE"], bool]:
        """
        Check if the sentence contains any structural markers that should suppress cuts.
        Implements a state machine for managing cut suppression across sentence boundaries.
        """

        # === STRUCTURAL MARKER STATE MACHINE ===
        # Priority order: NO_CUT > HOLD > RESUME > current state
        
        # 1. NO_CUT markers: Block cutting at this specific sentence only
        #    Use case: important content that shouldn't be split (e.g., page numbers, key phrases)
        for marker in self.structural_markers["NO_CUT"]:
            if marker.pattern.search(sentence_text):
                return "NO_CUT", marker.remove_marker

        # 2. HOLD markers: Start suppression mode - block all cuts until RESUME found
        #    Use case: entering protected content (e.g., code blocks, tables, quotes)
        for marker in self.structural_markers["HOLD"]:
            if marker.pattern.search(sentence_text):
                self.cut_suppressed = True  # Set global suppression state
                return "HOLD", marker.remove_marker

        # 3. RESUME markers: End suppression mode - restore normal cutting behavior
        #    Use case: exiting protected content (e.g., end of code blocks, tables)
        for marker in self.structural_markers["RESUME"]:
            if marker.pattern.search(sentence_text):
                self.cut_suppressed = False  # Clear global suppression state
                return "RESUME", marker.remove_marker

        # 4. Inherit current suppression state: if HOLD is active, continue suppressing
        #    This maintains protection across sentences between HOLD and RESUME markers
        if self.cut_suppressed:
            return "HOLD", False
        
        # 5. No structural markers found and no active suppression
        return "NONE", False


    def _find_optimal_cuts_greedy(self, candidates_list: List[Tuple[int, float]], 
                                 prefix_tokens: np.ndarray, N: int) -> List[int]:
        """
        Find optimal cuts using greedy approach based on highest scoring candidates.
        
        This algorithm iteratively selects the best available cut point for each chunk,
        balancing semantic quality (scores) with token constraints (min/max limits).
        
        Algorithm:
        1. Start from beginning of document
        2. For each chunk position, find all valid cut candidates within token limits
        3. Select the highest-scoring candidate that creates a valid chunk
        4. Move to the selected cut point and repeat until document end
        
        The greedy approach prioritizes local optimization at each step, which works well
        for text chunking where later decisions don't significantly affect earlier ones.
        
        Args:
            candidates_list: List of (sentence_index, score) tuples sorted by index
            prefix_tokens: Cumulative token counts array
            N: Total number of sentences
            
        Returns:
            List of sentence indices where cuts should be made
        """
        logger.debug(f"GREEDY: Processing {len(candidates_list)} candidates for {N} sentences")
        
        cuts_idx = []
        current_start = 0
        
        # === GREEDY CHUNKING ALGORITHM ===
        # Process document sequentially, making locally optimal cut decisions
        while current_start < N:
            # Find all candidates that would create valid chunks from current position
            # Considers both token constraints and semantic/structural scores
            valid_candidates = self._find_valid_candidates_for_chunk(
                candidates_list, prefix_tokens, current_start, N
            )
            
            if not valid_candidates:
                # Emergency case: no valid candidates found within constraints
                # Cut at document end to avoid infinite loop
                logger.debug(f"GREEDY: No valid candidates from position {current_start}, cutting at end {N}")
                cuts_idx.append(N)
                break
            
            # Select the candidate with highest combined score (semantic + structural)
            # Scores are already adjusted for token preference in _find_valid_candidates_for_chunk
            best_cut = self._select_best_cut_from_candidates(valid_candidates)
            cuts_idx.append(best_cut)    

            # Advance to next chunk: start where the current chunk ended
            current_start = best_cut
        
        return cuts_idx

    def _find_valid_candidates_for_chunk(self, candidates_list: List[Tuple[int, float]], 
                                        prefix_tokens: np.ndarray, start_pos: int, N: int) -> List[Tuple[int, float]]:
        """
        Find all candidates that would create a valid chunk from start_pos.
        
        Args:
            candidates_list: List of (sentence_index, score) tuples sorted by index
            prefix_tokens: Cumulative token counts array
            start_pos: Starting position for current chunk
            N: Total number of sentences
            
        Returns:
            List of valid (sentence_index, score) candidates for this chunk
        """
        valid_candidates = []
        
        for idx, score in candidates_list:
            if idx <= start_pos:
                continue  # Skip candidates at or before current start
            
            # Calculate token count for chunk from start_pos to this candidate
            token_count = prefix_tokens[idx] - prefix_tokens[start_pos]

            if token_count <= self.max_tokens:
                # === TOKEN-BASED PRIORITY SCORING ===
                # Implement flexible token limits with quality prioritization
                if token_count < self.min_tokens:
                    # Below minimum: considered "inferior" but still usable when necessary
                    # Keep original score without boost - will be used as fallback option
                    valid_candidates.append((idx, score))
                    logger.debug(f"GREEDY: Inferior below min_tokens candidate at {idx} (tokens: {token_count}, score: {score:.3f})")
                else:
                    # Within ideal range: boost score to prioritize over sub-minimum cuts
                    # +1 boost ensures these candidates are preferred over inferior ones
                    valid_candidates.append((idx, score+1))
                    logger.debug(f"GREEDY: Valid candidate at {idx} (tokens: {token_count}, score: {score:.3f})")
            else:
                if not valid_candidates:
                    valid_candidates.append((idx, score))
                    logger.warning(f"GREEDY: Emergency cut at {idx} (tokens: {token_count}) - exceeds max but preserves semantic coherence")
                    break
                else:
                    # Exceeds max_tokens - stop collecting and return what we have
                    logger.debug(f"GREEDY: Candidate at {idx} exceeds max_tokens ({token_count}), stopping collection")
                    break
        
        return valid_candidates

    def _select_best_cut_from_candidates(self, valid_candidates: List[Tuple[int, float]]) -> int:
        """
        Select the candidate with the highest score from the valid candidates.
        
        Args:
            valid_candidates: List of (sentence_index, score) tuples
            
        Returns:
            Sentence index of the best candidate
        """
        if not valid_candidates:
            raise ValueError("No valid candidates provided")
        
        # Find candidate with highest score
        best_candidate = max(valid_candidates, key=lambda x: x[1])
        best_idx, best_score = best_candidate
        
        logger.debug(f"GREEDY: Selected candidate at {best_idx} with score {best_score:.3f} from {len(valid_candidates)} options")
        
        return best_idx

    async def chunk_documents(self,
                              documents: List[Document],
                              original_file_name: str,
                              progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """
        Process a collection of documents and return semantically chunked text.
        This method is agnostic to the source of the documents.
        
        Args:
            documents: List of document objects with 'page_content' and 'metadata' 
            original_file_name: Name of the original file for metadata
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of dictionaries containing chunked text with metadata
        """
        if not documents:
            return []

        # Buffers for drift computation
        buffer_embeddings: List[np.ndarray] = []
        candidates: Dict[int, float] = {}  # sentence_index -> aggregated score

        # Store sentences and metadata with line tracking
        sentences: List[str] = []
        token_counts: List[int] = []
        char_positions: List[int] = []
        sentence_to_page: List[int] = []  # Track which page each sentence belongs to
        sentence_position_in_line: List[int] = []  # Position within line (0-based)
        is_last_sentence_in_line: List[bool] = []  # Whether sentence is last in its line
        full_text_parts: List[str] = []

        char_pos = 0
        sentence_idx = 0
        total_docs = len(documents)
        avg_drift = 0.00
        no_of_drifts = 0
        # Process each document
        for doc_idx, doc in enumerate(documents):
            page_content = doc.page_content + "\n"  # making sure there is a new line between pages
            full_text_parts.append(page_content)
            page_num = 1  # Default page number
            
            # Try to get page number from metadata
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                page_num = doc.metadata.get('page', 1)

            
            # NEW: Split by newlines first (outer loop)
            lines = page_content.split('\n')
            for line_idx, line in enumerate(lines):

                # Sentence tokenize this line
                line_sentences = sent_tokenize(line)
                
                # Process each sentence in this line
                for sent_pos, sent in enumerate(line_sentences):

                    # Check if this sentence is the last in the line
                    is_last_in_line = (sent_pos == len(line_sentences) - 1)
                    struct_cut_suppression, remove_marker = self._check_for_structural_cuts_suppression(sent)
                    if remove_marker:
                        logger.debug(f"STRUCTURAL: Removing sentence {sentence_idx} due to structural marker")
                        continue
                    if struct_cut_suppression in ["NONE", "RESUME"]:
                        # Check structural markers at sentence level
                        struct_cands = self._add_structural_cut_candidates(sent, sentence_idx)
                        for idx_cand, score_cand in struct_cands.items():
                            candidates[idx_cand] = candidates.get(idx_cand, 0.0) + score_cand
                            logger.debug(f"CANDIDATE: Structural candidate at sentence {idx_cand} (score={score_cand:.3f}) from sentence {sentence_idx}")

                        # === SEMANTIC DRIFT DETECTION ===
                        # Generate embedding for current sentence
                        emb = self._generate_embedding(sent)
                        
                        # Maintain sliding window buffer of sentence embeddings
                        buffer_embeddings.append(emb)
                        if len(buffer_embeddings) > self.window_size + 1:
                            buffer_embeddings.pop(0)  # Keep window size + 1 for overlapping comparison
                        
                        # Compute semantic drift when we have enough embeddings for comparison
                        # Need window_size + 1 embeddings to create two overlapping windows
                        if len(buffer_embeddings) >= self.window_size + 1:
                            w = self.window_size
                            # Compare two overlapping windows: previous vs current
                            v1 = np.mean(buffer_embeddings[-w-1:-1], axis=0)  # Previous window (older sentences)
                            v2 = np.mean(buffer_embeddings[-w:], axis=0)      # Current window (newer sentences)
                            
                            # Calculate semantic drift (1 - cosine_similarity)
                            # drift=0: identical meaning, drift=1: completely different topics
                            drift = 1 - cosine_similarity([v1], [v2])[0][0]
                            
                            # Update running average for adaptive thresholding
                            # Each document establishes its own "normal" drift baseline
                            avg_drift = (avg_drift * no_of_drifts + drift) / (no_of_drifts + 1)
                            no_of_drifts += 1
                            
                            # Only significant semantic changes (above document average) become cut candidates
                            # This adaptive approach prevents false positives from minor variations
                            if drift > avg_drift:
                                # Place cut candidate at current sentence (end of the newer window)
                                candidates[sentence_idx] = candidates.get(sentence_idx, 0.0) + float(drift)
                                logger.debug(f"CANDIDATE: Semantic drift candidate at sentence {sentence_idx} (drift={drift:.3f})")
                    else:
                        logger.debug(f"CUT SUPPRESSED: Skipping sentence {sentence_idx}: '{sent[:10]}...'")
                    
                    # record text & metadata with line tracking
                    sentences.append(sent)
                    tok = self._count_tokens(sent)
                    token_counts.append(tok)
                    char_positions.append(char_pos)
                    sentence_to_page.append(page_num)
                    sentence_position_in_line.append(sent_pos)
                    is_last_sentence_in_line.append(is_last_in_line)
                    
                    # advance
                    char_pos += len(sent) + 1
                    sentence_idx += 1

            # Send progress if callback provided
            if progress_callback:
                try:
                    progress = ((doc_idx + 1) / total_docs) * 100
                    await progress_callback.send_progress('processing_progress', {'progress': progress})
                    await progress_callback.send_progress('status', {
                        'message': f'Processing document {doc_idx + 1} of {total_docs}...'
                    })
                except Exception as e:
                    logger.error(f"Error sending progress update: {e}")

        N = len(sentences)
        if N == 0:
            logger.warning("No text extracted from documents")
            return []

        # Prefix sum of tokens
        prefix_tokens = np.zeros(N+1, dtype=int)
        for i in range(N):
            prefix_tokens[i+1] = prefix_tokens[i] + token_counts[i]

        # Ensure the end of the document is a candidate so the final chunk respects max_tokens
        if N not in candidates:
            candidates[N] = 0.0

        # Create a sorted list of (idx, score) for DP
        candidates_list = sorted(candidates.items(), key=lambda x: x[0])
        logger.debug(f"Total sentences: {N}")
        logger.debug(f"Candidates found: {[(idx, score) for idx, score in candidates_list]}")


        M = len(candidates_list)
        if M == 0:
            logger.warning("No candidates found, returning single chunk")
            # no semantic breaks: single chunk
            return [{
                'file_name': original_file_name,
                'page': sentence_to_page[0] if sentence_to_page else 1,
                'chunk_id': 0,
                'text': ' '.join(sentences)
            }]

        # Run greedy algorithm to find optimal cuts
        cuts_idx = self._find_optimal_cuts_greedy(candidates_list, prefix_tokens, N)
        logger.debug(f"Final cuts selected: {cuts_idx}")

        # NEW: slice sentences into chunks with line-aware reconstruction
        cuts = cuts_idx
        starts = [0] + cuts_idx[:-1]
        output: List[Dict[str, Any]] = []
        logger.debug(f"CHUNKING: Creating {len(cuts)} chunks with starts={starts} and cuts={cuts}")
        
        for chunk_id, (s, e) in enumerate(zip(starts, cuts), start=1):
            logger.debug(f"CHUNK {chunk_id}: Sentences {s} to {e} (inclusive)")
            
            # Reconstruct text preserving newline structure
            chunk_text_parts = []
            for sent_idx in range(s, e):
                sent = sentences[sent_idx]
                chunk_text_parts.append(sent)
                # Add newline if this sentence is the last in its original line
                # and it's not the last sentence in the chunk
                if is_last_sentence_in_line[sent_idx] and sent_idx < e:
                    chunk_text_parts.append('\n')
            
            text = ''.join(chunk_text_parts)

            # Use the page of the first sentence in the chunk
            page = sentence_to_page[s] if s < len(sentence_to_page) else 1
            
            chunk_tokens = self._count_tokens(text)
            logger.debug(f"CHUNK {chunk_id}: {chunk_tokens} tokens, page {page}, text preview: '{text[:20]}...'")
            
            output.append({
                'file_name': original_file_name,
                'page': page,
                'chunk_id': chunk_id,
                'text': text
            })
        # === POST-PROCESSING: CHUNK MERGING ===
        # Handle remaining sub-minimum chunks by merging them with adjacent chunks
        # This ensures no final chunk is below min_tokens while preserving content quality
        if self.merge_small_chunks:
            merged_output: List[Dict[str, Any]] = []
            i = 0
            while i < len(output):
                # Forward merge: if current chunk is too small and there's a next chunk
                if i < len(output) - 1 and self._count_tokens(output[i]['text']) < self.min_tokens:
                    # Merge current chunk into the next chunk (preserves next chunk's semantic boundary)
                    output[i + 1]['text'] = output[i]['text'] + ' ' + output[i + 1]['text']
                    # Keep the earliest page number for proper attribution
                    output[i + 1]['page'] = output[i]['page']
                    i += 1  # Skip adding the merged chunk to output
                    continue
                merged_output.append(output[i])
                i += 1

            # Backward merge: handle final chunk if it's still too small
            # Can't merge forward anymore, so merge with previous chunk
            if len(merged_output) > 1 and self._count_tokens(merged_output[-1]['text']) < self.min_tokens:
                merged_output[-2]['text'] = merged_output[-2]['text'] + ' ' + merged_output[-1]['text']
                # Keep the page of the larger (previous) chunk
                merged_output.pop()

            # Reassign sequential chunk IDs after all merging operations
            for new_id, ch in enumerate(merged_output, start=1):
                ch['chunk_id'] = new_id

            output = merged_output
        # When merge_small_chunks=False, no reassignment needed since no chunks were removed
        return output
