"""Data processing pipeline for word permutation analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DistributionStats:
    """Statistics for a distribution of word counts."""
    word_length: int
    total_permutations: int
    mean: float
    median: float
    std: float
    min: int
    max: int
    q25: float
    q75: float
    skewness: float
    kurtosis: float


class DataProcessor:
    """Process and aggregate permutation analysis data."""

    def __init__(self):
        """Initialize the data processor."""
        self.results_cache: Dict[int, pd.DataFrame] = {}
        self.stats_cache: Dict[int, DistributionStats] = {}

    def add_results(self, word_length: int, df: pd.DataFrame):
        """
        Add analysis results for a specific word length.

        Args:
            word_length: Length of words analyzed
            df: DataFrame with permutation analysis results
        """
        self.results_cache[word_length] = df.copy()
        self.stats_cache[word_length] = self._calculate_stats(word_length, df)

    def _calculate_stats(self, word_length: int, df: pd.DataFrame) -> DistributionStats:
        """Calculate distribution statistics for word counts."""
        word_counts = df['word_count']

        return DistributionStats(
            word_length=word_length,
            total_permutations=len(df),
            mean=float(word_counts.mean()),
            median=float(word_counts.median()),
            std=float(word_counts.std()),
            min=int(word_counts.min()),
            max=int(word_counts.max()),
            q25=float(word_counts.quantile(0.25)),
            q75=float(word_counts.quantile(0.75)),
            skewness=float(word_counts.skew()),
            kurtosis=float(word_counts.kurtosis())
        )

    def get_stats(self, word_length: int) -> DistributionStats:
        """Get distribution statistics for a word length."""
        if word_length not in self.stats_cache:
            raise ValueError(f"No data available for word length {word_length}")
        return self.stats_cache[word_length]

    def get_distribution_data(self, word_length: int, bins: int = 50) -> pd.DataFrame:
        """
        Get histogram data for a word length distribution.

        Args:
            word_length: Length of words
            bins: Number of histogram bins

        Returns:
            DataFrame with bin edges and counts
        """
        if word_length not in self.results_cache:
            raise ValueError(f"No data available for word length {word_length}")

        df = self.results_cache[word_length]
        counts, edges = np.histogram(df['word_count'], bins=bins)

        return pd.DataFrame({
            'bin_start': edges[:-1],
            'bin_end': edges[1:],
            'bin_center': (edges[:-1] + edges[1:]) / 2,
            'count': counts,
            'frequency': counts / counts.sum()
        })

    def get_comparison_data(self) -> pd.DataFrame:
        """
        Get comparison data across all word lengths.

        Returns:
            DataFrame with comparison statistics
        """
        comparison_rows = []

        for word_length, stats in sorted(self.stats_cache.items()):
            comparison_rows.append({
                'word_length': word_length,
                'total_permutations': stats.total_permutations,
                'mean': stats.mean,
                'median': stats.median,
                'std': stats.std,
                'min': stats.min,
                'max': stats.max,
                'q25': stats.q25,
                'q75': stats.q75,
                'skewness': stats.skewness,
                'kurtosis': stats.kurtosis
            })

        return pd.DataFrame(comparison_rows)

    def prepare_visualization_data(self, bins: int = 50) -> Dict:
        """
        Prepare all data needed for visualization.

        Args:
            bins: Number of bins for histograms

        Returns:
            Dictionary containing all visualization data
        """
        viz_data = {
            'comparison_stats': self.get_comparison_data().to_dict('records'),
            'distributions': {}
        }

        for word_length in sorted(self.results_cache.keys()):
            dist_data = self.get_distribution_data(word_length, bins)
            viz_data['distributions'][word_length] = {
                'histogram': dist_data.to_dict('records'),
                'raw_counts': self.results_cache[word_length]['word_count'].tolist(),
                'stats': {
                    'mean': self.stats_cache[word_length].mean,
                    'median': self.stats_cache[word_length].median,
                    'std': self.stats_cache[word_length].std,
                    'min': self.stats_cache[word_length].min,
                    'max': self.stats_cache[word_length].max,
                }
            }

        return viz_data

    def get_top_permutations(self, word_length: int, n: int = 10) -> pd.DataFrame:
        """
        Get top N permutations by word count.

        Args:
            word_length: Length of words
            n: Number of top permutations to return

        Returns:
            DataFrame with top permutations
        """
        if word_length not in self.results_cache:
            raise ValueError(f"No data available for word length {word_length}")

        df = self.results_cache[word_length]
        return df.nlargest(n, 'word_count')

    def get_bottom_permutations(self, word_length: int, n: int = 10) -> pd.DataFrame:
        """
        Get bottom N permutations by word count.

        Args:
            word_length: Length of words
            n: Number of bottom permutations to return

        Returns:
            DataFrame with bottom permutations
        """
        if word_length not in self.results_cache:
            raise ValueError(f"No data available for word length {word_length}")

        df = self.results_cache[word_length]
        return df.nsmallest(n, 'word_count')

    def export_to_csv(self, word_length: int, output_path: str):
        """Export results for a word length to CSV."""
        if word_length not in self.results_cache:
            raise ValueError(f"No data available for word length {word_length}")

        self.results_cache[word_length].to_csv(output_path, index=False)
        print(f"Exported {word_length}-letter results to {output_path}")

    def export_all_to_csv(self, output_dir: str = "."):
        """Export all results to CSV files."""
        import os

        for word_length in self.results_cache.keys():
            output_path = os.path.join(
                output_dir,
                f"permutation_analysis_{word_length}letter.csv"
            )
            self.export_to_csv(word_length, output_path)
