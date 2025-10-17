"""GPU-accelerated word permutation analyzer using CuPy and Numba CUDA."""

import itertools
from typing import Set, List, Dict, Optional
import numpy as np
import pandas as pd

# GPU imports (will be installed in notebook environment)
try:
    import cupy as cp
    import cudf
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("WARNING: GPU libraries not available. Install cupy-cuda12x, numba, and cudf-cu12")


def char_count_vector(word: str) -> np.ndarray:
    """
    Convert a word to a 26-element character count vector.

    Args:
        word: Input word (lowercase)

    Returns:
        Array of shape (26,) with counts for a-z
    """
    counts = np.zeros(26, dtype=np.int8)
    for char in word.lower():
        if 'a' <= char <= 'z':
            counts[ord(char) - ord('a')] += 1
    return counts


def words_to_char_matrix(words: List[str]) -> np.ndarray:
    """
    Convert list of words to character count matrix.

    Args:
        words: List of words

    Returns:
        Array of shape (N_words, 26) with character counts
    """
    n_words = len(words)
    matrix = np.zeros((n_words, 26), dtype=np.int8)

    for i, word in enumerate(words):
        matrix[i] = char_count_vector(word)

    return matrix


@cuda.jit
def count_valid_words_kernel(perm_counts, word_counts, results):
    """
    CUDA kernel to count valid words for each permutation.

    Each thread processes one permutation and checks against all words.

    Args:
        perm_counts: (N_perms, 26) character counts for permutations
        word_counts: (N_words, 26) character counts for words
        results: (N_perms,) output array for word counts
    """
    # Get thread position
    perm_idx = cuda.grid(1)

    if perm_idx >= perm_counts.shape[0]:
        return

    count = 0
    n_words = word_counts.shape[0]

    # Check each word
    for word_idx in range(n_words):
        # Check if word can be formed from permutation
        can_form = True
        for char_idx in range(26):
            if word_counts[word_idx, char_idx] > perm_counts[perm_idx, char_idx]:
                can_form = False
                break

        if can_form:
            count += 1

    results[perm_idx] = count


class GPUWordPermutationAnalyzer:
    """GPU-accelerated analyzer for word permutations using CUDA."""

    def __init__(self, words_csv_path: str, word_column: str = None, gpu_id: int = 0):
        """
        Initialize GPU analyzer.

        Args:
            words_csv_path: Path to CSV file with words
            word_column: Column name for words (None = first column)
            gpu_id: GPU device ID to use
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU libraries not installed. Run: uv pip install cupy-cuda12x numba cudf-cu12")

        self.gpu_id = gpu_id
        self.words_csv_path = words_csv_path
        self.word_column = word_column

        # Set GPU device
        cp.cuda.Device(gpu_id).use()

        # Load words
        self._load_words()

    def _load_words(self):
        """Load words from CSV using cuDF for speed."""
        print(f"Loading words from {self.words_csv_path} on GPU {self.gpu_id}...")

        try:
            # Try cuDF first (faster)
            df = cudf.read_csv(self.words_csv_path)

            if self.word_column is None:
                self.word_column = df.columns[0]

            # Get words as list
            words_series = df[self.word_column].str.lower()
            self.all_words = words_series.to_pandas().tolist()

        except Exception as e:
            print(f"cuDF failed ({e}), falling back to pandas...")
            # Fallback to pandas
            df = pd.read_csv(self.words_csv_path)

            if self.word_column is None:
                self.word_column = df.columns[0]

            self.all_words = df[self.word_column].str.lower().tolist()

        print(f"Loaded {len(self.all_words):,} total words")

        # Organize by length for efficient processing
        self.words_by_length = {}
        for word in self.all_words:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word)

        # Create GPU character count matrices for each length
        self.word_matrices_gpu = {}
        for length, words in self.words_by_length.items():
            print(f"  {length}-letter words: {len(words):,}")
            char_matrix = words_to_char_matrix(words)
            self.word_matrices_gpu[length] = cp.asarray(char_matrix)

    def generate_permutations_gpu(self, word_length: int, batch_size: int = None) -> cp.ndarray:
        """
        Generate all permutations of given length on GPU.

        Args:
            word_length: Length of permutations (5, 6, or 7)
            batch_size: If specified, yield batches instead of all at once

        Returns:
            CuPy array of shape (N_perms, 26) with character counts
        """
        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        print(f"Generating all {word_length}-letter permutations...")

        # Generate all permutations (on CPU first, then transfer)
        all_perms = list(itertools.permutations(alphabet, word_length))
        n_perms = len(all_perms)

        print(f"Total permutations: {n_perms:,}")
        print(f"Memory required: {n_perms * 26 / (1024**3):.2f} GB")

        if batch_size and n_perms > batch_size:
            # Process in batches
            print(f"Processing in batches of {batch_size:,}")
            return self._generate_permutations_batched(all_perms, batch_size)

        # Convert all permutations to character count matrix
        perm_matrix = np.zeros((n_perms, 26), dtype=np.int8)

        for i, perm in enumerate(all_perms):
            if (i + 1) % 1_000_000 == 0:
                print(f"  Processed {i+1:,} / {n_perms:,} permutations...")
            perm_str = ''.join(perm)
            perm_matrix[i] = char_count_vector(perm_str)

        # Transfer to GPU
        print(f"Transferring to GPU {self.gpu_id}...")
        perm_matrix_gpu = cp.asarray(perm_matrix)

        return perm_matrix_gpu

    def _generate_permutations_batched(self, all_perms, batch_size):
        """Generator for batched permutation processing."""
        n_perms = len(all_perms)

        for start_idx in range(0, n_perms, batch_size):
            end_idx = min(start_idx + batch_size, n_perms)
            batch = all_perms[start_idx:end_idx]
            batch_size_actual = len(batch)

            print(f"Batch {start_idx:,} to {end_idx:,}")

            # Convert batch to matrix
            perm_matrix = np.zeros((batch_size_actual, 26), dtype=np.int8)
            for i, perm in enumerate(batch):
                perm_matrix[i] = char_count_vector(''.join(perm))

            yield cp.asarray(perm_matrix), start_idx, end_idx

    def analyze_permutations_gpu(
        self,
        word_length: int,
        batch_size: Optional[int] = None,
        return_details: bool = False
    ) -> Dict:
        """
        Analyze all permutations for a given word length on GPU.

        Args:
            word_length: Length of words/permutations
            batch_size: Batch size for large permutation sets
            return_details: If True, return full permutation details

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*60}")
        print(f"Analyzing {word_length}-letter permutations on GPU {self.gpu_id}")
        print(f"{'='*60}\n")

        # Get word matrix for this length
        if word_length not in self.word_matrices_gpu:
            print(f"No {word_length}-letter words in dictionary!")
            return {'word_length': word_length, 'permutations': 0, 'results': []}

        word_matrix_gpu = self.word_matrices_gpu[word_length]
        n_words = word_matrix_gpu.shape[0]

        print(f"Dictionary has {n_words:,} words of length {word_length}")

        # Determine batch size based on memory
        if batch_size is None:
            # Estimate: 7-letter has 3.3B perms × 26 bytes = 86GB
            # For H100 80GB, use batches for 7-letter
            if word_length >= 7:
                batch_size = 100_000_000  # 100M per batch ≈ 2.6GB
            else:
                batch_size = None  # Process all at once

        # Generate and process permutations
        perm_gen = self.generate_permutations_gpu(word_length, batch_size)

        all_results = []
        perm_strings = []

        # Configure CUDA kernel
        threads_per_block = 256

        if batch_size:
            # Batched processing
            for perm_batch_gpu, start_idx, end_idx in perm_gen:
                batch_size_actual = perm_batch_gpu.shape[0]
                blocks_per_grid = (batch_size_actual + threads_per_block - 1) // threads_per_block

                # Allocate results on GPU
                results_gpu = cp.zeros(batch_size_actual, dtype=cp.int32)

                # Launch kernel
                print(f"Launching CUDA kernel for batch {start_idx:,} to {end_idx:,}...")
                count_valid_words_kernel[blocks_per_grid, threads_per_block](
                    perm_batch_gpu, word_matrix_gpu, results_gpu
                )

                # Get results back to CPU
                batch_results = results_gpu.get()
                all_results.extend(batch_results)

                print(f"Batch complete. Progress: {end_idx:,} permutations processed")

        else:
            # Process all at once
            perm_matrix_gpu = perm_gen
            n_perms = perm_matrix_gpu.shape[0]
            blocks_per_grid = (n_perms + threads_per_block - 1) // threads_per_block

            # Allocate results on GPU
            results_gpu = cp.zeros(n_perms, dtype=cp.int32)

            # Launch kernel
            print(f"Launching CUDA kernel with {blocks_per_grid:,} blocks × {threads_per_block} threads...")
            count_valid_words_kernel[blocks_per_grid, threads_per_block](
                perm_matrix_gpu, word_matrix_gpu, results_gpu
            )

            # Get results back to CPU
            all_results = results_gpu.get()

        print(f"\nProcessing complete!")
        print(f"Total permutations analyzed: {len(all_results):,}")

        # Create summary statistics
        results_array = np.array(all_results)
        summary = {
            'word_length': word_length,
            'total_permutations': len(all_results),
            'mean_words': float(results_array.mean()),
            'median_words': float(np.median(results_array)),
            'std_words': float(results_array.std()),
            'min_words': int(results_array.min()),
            'max_words': int(results_array.max()),
            'results': all_results if return_details else None
        }

        print(f"\nStatistics:")
        print(f"  Mean words per permutation: {summary['mean_words']:.2f}")
        print(f"  Median: {summary['median_words']:.0f}")
        print(f"  Std dev: {summary['std_words']:.2f}")
        print(f"  Range: {summary['min_words']} - {summary['max_words']}")

        return summary

    def analyze_all_lengths(
        self,
        lengths: List[int] = [5, 6, 7],
        batch_size: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Analyze permutations for multiple word lengths.

        Args:
            lengths: List of word lengths to analyze
            batch_size: Batch size for large permutation sets

        Returns:
            Dictionary mapping word length to results
        """
        results = {}

        for length in lengths:
            results[length] = self.analyze_permutations_gpu(
                word_length=length,
                batch_size=batch_size,
                return_details=False
            )

        return results
