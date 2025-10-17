"""Core word permutation analysis module."""

import itertools
from typing import Set, List, Dict, Tuple
import pandas as pd
import numpy as np
import polars as pl
from collections import defaultdict


class WordPermutationAnalyzer:
    """Analyzes valid English words that can be formed from letter permutations."""

    def __init__(self, words_csv_path: str, word_column: str = None):
        """
        Initialize the analyzer with a CSV file of valid English words.

        Args:
            words_csv_path: Path to CSV file containing valid English words
            word_column: Name of column containing words (uses first column if None)
        """
        self.words_csv_path = words_csv_path
        self.word_column = word_column
        self.valid_words: Set[str] = set()
        self._load_words()

    def _load_words(self):
        """Load valid words from CSV file."""
        # Use polars for fast CSV reading
        df = pl.read_csv(self.words_csv_path)

        # Get first column if no column specified
        if self.word_column is None:
            self.word_column = df.columns[0]

        # Convert to set of lowercase words for fast lookup
        self.valid_words = set(
            df[self.word_column].str.to_lowercase().to_list()
        )

        print(f"Loaded {len(self.valid_words):,} valid words from {self.words_csv_path}")

    def get_words_by_length(self, length: int) -> Set[str]:
        """Get all valid words of a specific length."""
        return {word for word in self.valid_words if len(word) == length}

    def analyze_permutation(self, letters: Tuple[str, ...]) -> Dict[str, any]:
        """
        Analyze a single permutation of letters.

        Args:
            letters: Tuple of letters (e.g., ('a', 'b', 'c', 'd', 'e'))

        Returns:
            Dictionary with analysis results
        """
        letters_set = set(letters)
        word_length = len(letters)

        # Find all valid words that can be formed using only these letters
        valid_words_from_permutation = []

        # Check words of the same length
        for word in self.get_words_by_length(word_length):
            # Check if word can be formed from the letters
            word_letters = list(word)
            available_letters = list(letters)

            can_form = True
            for char in word_letters:
                if char in available_letters:
                    available_letters.remove(char)
                else:
                    can_form = False
                    break

            if can_form:
                valid_words_from_permutation.append(word)

        return {
            'permutation': ''.join(letters),
            'letters': letters,
            'word_count': len(valid_words_from_permutation),
            'words': valid_words_from_permutation
        }

    def analyze_all_permutations(
        self,
        word_length: int,
        sample_size: int = None,
        alphabet_subset: str = None
    ) -> pd.DataFrame:
        """
        Analyze all permutations of a given length.

        Args:
            word_length: Length of permutations to analyze (5, 6, or 7)
            sample_size: If specified, randomly sample this many permutations
            alphabet_subset: If specified, use only these letters (default: a-z)

        Returns:
            DataFrame with analysis results
        """
        if alphabet_subset is None:
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
        else:
            alphabet = alphabet_subset.lower()

        print(f"\nAnalyzing {word_length}-letter permutations from alphabet: {alphabet}")

        # Generate permutations
        all_perms = list(itertools.permutations(alphabet, word_length))
        total_perms = len(all_perms)

        print(f"Total possible permutations: {total_perms:,}")

        # Sample if requested
        if sample_size and sample_size < total_perms:
            import random
            all_perms = random.sample(all_perms, sample_size)
            print(f"Sampling {sample_size:,} permutations")

        # Analyze each permutation
        results = []
        for i, perm in enumerate(all_perms):
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1:,} / {len(all_perms):,} permutations analyzed")

            result = self.analyze_permutation(perm)
            results.append({
                'permutation': result['permutation'],
                'word_count': result['word_count'],
                'words': ','.join(result['words']) if result['words'] else ''
            })

        df = pd.DataFrame(results)
        print(f"\nCompleted analysis of {len(df):,} permutations")
        print(f"Statistics:")
        print(f"  Mean words per permutation: {df['word_count'].mean():.2f}")
        print(f"  Median words per permutation: {df['word_count'].median():.0f}")
        print(f"  Max words per permutation: {df['word_count'].max():.0f}")
        print(f"  Min words per permutation: {df['word_count'].min():.0f}")

        return df

    def compare_word_lengths(
        self,
        lengths: List[int] = [5, 6, 7],
        sample_size: int = None,
        alphabet_subset: str = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Compare permutation analysis across different word lengths.

        Args:
            lengths: List of word lengths to analyze
            sample_size: Number of permutations to sample per length
            alphabet_subset: Subset of alphabet to use

        Returns:
            Dictionary mapping word length to analysis DataFrame
        """
        results = {}

        for length in lengths:
            print(f"\n{'='*60}")
            print(f"Analyzing {length}-letter permutations")
            print(f"{'='*60}")

            df = self.analyze_all_permutations(
                word_length=length,
                sample_size=sample_size,
                alphabet_subset=alphabet_subset
            )
            results[length] = df

        # Print comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        for length, df in results.items():
            print(f"\n{length}-letter permutations:")
            print(f"  Permutations analyzed: {len(df):,}")
            print(f"  Mean word count: {df['word_count'].mean():.2f}")
            print(f"  Std dev: {df['word_count'].std():.2f}")
            print(f"  Median: {df['word_count'].median():.0f}")
            print(f"  Max: {df['word_count'].max():.0f}")

        return results
