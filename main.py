#!/usr/bin/env python3
"""Command-line interface for word permutation analysis."""

import argparse
import os
from word_anal.analyzer import WordPermutationAnalyzer
from word_anal.data_processing import DataProcessor
from word_anal.visualizations import VisualizationGenerator
from word_anal.kaggle_helper import get_dictionary_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Analyze valid English words from letter permutations"
    )
    parser.add_argument(
        "words_csv",
        nargs="?",
        help="Path to CSV file containing valid English words (default: download from Kaggle)"
    )
    parser.add_argument(
        "--word-column",
        help="Name of column containing words (default: first column)",
        default=None
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[5, 6, 7],
        help="Word lengths to analyze (default: 5 6 7)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of permutations to sample per length (default: 10000)"
    )
    parser.add_argument(
        "--alphabet",
        help="Subset of alphabet to use (default: full a-z)",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output path for HTML visualization (default: word_analysis.html)",
        default="word_analysis.html"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV files"
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download Kaggle dictionary dataset (requires Kaggle credentials in ~/.kaggle/kaggle.json)"
    )
    parser.add_argument(
        "--kaggle-username",
        help="Kaggle username (alternative to kaggle.json)",
        default=None
    )
    parser.add_argument(
        "--kaggle-key",
        help="Kaggle API key (alternative to kaggle.json)",
        default=None
    )

    args = parser.parse_args()

    print("Word Permutation Analysis")
    print("=" * 60)

    # Handle dataset download
    words_csv_path = args.words_csv

    if args.download_dataset or not words_csv_path:
        print("\nDownloading Kaggle dictionary dataset...")

        credentials = None
        if args.kaggle_username and args.kaggle_key:
            credentials = {
                "username": args.kaggle_username,
                "key": args.kaggle_key
            }

        words_csv_path = get_dictionary_dataset(
            credentials=credentials,
            download_path="data",
            force=False
        )

        # Use 'word' column for Kaggle dataset
        if args.word_column is None:
            args.word_column = "word"

    if not words_csv_path:
        parser.error("Either provide words_csv path or use --download-dataset")

    # Initialize analyzer
    analyzer = WordPermutationAnalyzer(
        words_csv_path=words_csv_path,
        word_column=args.word_column
    )

    # Run analysis
    results = analyzer.compare_word_lengths(
        lengths=args.lengths,
        sample_size=args.sample_size,
        alphabet_subset=args.alphabet
    )

    # Process data
    processor = DataProcessor()
    for word_length, df in results.items():
        processor.add_results(word_length, df)

    # Display comparison
    comparison_df = processor.get_comparison_data()
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Generate visualization
    viz_gen = VisualizationGenerator(processor)
    viz_gen.generate_html(output_path=args.output)

    # Export to CSV if requested
    if args.export_csv:
        processor.export_all_to_csv()
        comparison_df.to_csv("comparison_statistics.csv", index=False)
        print("\nExported results to CSV files")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Open {args.output} in a web browser to view interactive visualizations")


if __name__ == "__main__":
    main()
