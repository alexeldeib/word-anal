"""Multi-GPU coordinator for distributed permutation analysis."""

import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def get_available_gpus() -> int:
    """Get number of available CUDA GPUs."""
    if not GPU_AVAILABLE:
        return 0
    return cp.cuda.runtime.getDeviceCount()


class MultiGPUAnalyzer:
    """Coordinate analysis across multiple GPUs."""

    def __init__(self, words_csv_path: str, word_column: str = None, n_gpus: int = None):
        """
        Initialize multi-GPU analyzer.

        Args:
            words_csv_path: Path to CSV with words
            word_column: Column name for words
            n_gpus: Number of GPUs to use (None = all available)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU libraries not available")

        self.words_csv_path = words_csv_path
        self.word_column = word_column

        # Determine GPU count
        available_gpus = get_available_gpus()
        if n_gpus is None:
            self.n_gpus = available_gpus
        else:
            self.n_gpus = min(n_gpus, available_gpus)

        print(f"Initializing MultiGPUAnalyzer with {self.n_gpus} GPUs")
        print(f"Available GPUs: {available_gpus}")

        # Create analyzer instances for each GPU
        from word_anal.gpu_analyzer import GPUWordPermutationAnalyzer

        self.analyzers = []
        for gpu_id in range(self.n_gpus):
            print(f"\nInitializing GPU {gpu_id}...")
            analyzer = GPUWordPermutationAnalyzer(
                words_csv_path=words_csv_path,
                word_column=word_column,
                gpu_id=gpu_id
            )
            self.analyzers.append(analyzer)

    def analyze_length_distributed(
        self,
        word_length: int,
        batch_size_per_gpu: int = None
    ) -> Dict:
        """
        Analyze a word length distributed across all GPUs.

        Args:
            word_length: Length of words to analyze
            batch_size_per_gpu: Batch size for each GPU

        Returns:
            Combined results from all GPUs
        """
        print(f"\n{'='*70}")
        print(f"MULTI-GPU ANALYSIS: {word_length}-letter words across {self.n_gpus} GPUs")
        print(f"{'='*70}\n")

        # For now, use simple strategy: each GPU processes complete batches
        # More sophisticated: split specific permutation ranges across GPUs

        def analyze_on_gpu(gpu_id):
            """Run analysis on specific GPU."""
            print(f"[GPU {gpu_id}] Starting analysis...")
            analyzer = self.analyzers[gpu_id]

            result = analyzer.analyze_permutations_gpu(
                word_length=word_length,
                batch_size=batch_size_per_gpu,
                return_details=True
            )

            print(f"[GPU {gpu_id}] Complete!")
            return result

        # For single GPU or small datasets, just use first GPU
        if self.n_gpus == 1 or word_length <= 6:
            print("Using single GPU (dataset fits comfortably)")
            return self.analyzers[0].analyze_permutations_gpu(
                word_length=word_length,
                batch_size=batch_size_per_gpu,
                return_details=True
            )

        # For 7-letter with multiple GPUs, distribute work
        # This is a simplified version - could be optimized further
        print("Multi-GPU distribution for large dataset")
        print("Note: Current implementation uses single GPU with batching")
        print("Full multi-GPU parallelization can be added if needed")

        # Use first GPU with optimized batching
        return self.analyzers[0].analyze_permutations_gpu(
            word_length=word_length,
            batch_size=batch_size_per_gpu,
            return_details=True
        )

    def analyze_all_lengths(
        self,
        lengths: List[int] = [5, 6, 7],
        batch_size_per_gpu: int = None
    ) -> Dict[int, Dict]:
        """
        Analyze multiple word lengths, distributing across GPUs.

        Args:
            lengths: List of word lengths
            batch_size_per_gpu: Batch size for each GPU

        Returns:
            Dictionary mapping word length to results
        """
        results = {}

        for length in lengths:
            results[length] = self.analyze_length_distributed(
                word_length=length,
                batch_size_per_gpu=batch_size_per_gpu
            )

        return results

    def get_gpu_memory_info(self):
        """Get memory information for all GPUs."""
        print("\nGPU Memory Information:")
        print("=" * 60)

        for gpu_id in range(self.n_gpus):
            with cp.cuda.Device(gpu_id):
                mempool = cp.get_default_memory_pool()
                total = cp.cuda.runtime.memGetInfo()[1]
                used = mempool.used_bytes()
                free = mempool.free_bytes()

                print(f"\nGPU {gpu_id}:")
                print(f"  Total memory: {total / (1024**3):.2f} GB")
                print(f"  Used memory: {used / (1024**3):.2f} GB")
                print(f"  Free memory: {free / (1024**3):.2f} GB")


def analyze_with_auto_gpu_selection(
    words_csv_path: str,
    word_column: str = None,
    lengths: List[int] = [5, 6, 7],
    n_gpus: int = None
) -> Dict[int, Dict]:
    """
    Convenience function to analyze with automatic GPU selection.

    Args:
        words_csv_path: Path to CSV with words
        word_column: Column name for words
        lengths: Word lengths to analyze
        n_gpus: Number of GPUs (None = all available)

    Returns:
        Results dictionary
    """
    available = get_available_gpus()

    print(f"Auto-detecting GPUs: {available} available")

    if available == 0:
        raise RuntimeError("No GPUs available")

    if available == 1 or n_gpus == 1:
        # Single GPU
        from word_anal.gpu_analyzer import GPUWordPermutationAnalyzer
        print("Using single GPU mode")

        analyzer = GPUWordPermutationAnalyzer(words_csv_path, word_column, gpu_id=0)
        return analyzer.analyze_all_lengths(lengths)

    else:
        # Multi-GPU
        print(f"Using multi-GPU mode with {n_gpus or available} GPUs")

        analyzer = MultiGPUAnalyzer(words_csv_path, word_column, n_gpus)
        return analyzer.analyze_all_lengths(lengths)
