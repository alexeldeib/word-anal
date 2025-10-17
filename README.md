# Word Permutation Analysis

Analyze how many valid English words can be formed from different permutations of 5-7 letters in the alphabet. Includes **GPU acceleration** for processing billions of permutations and interactive D3.js visualizations.

## Features

### Core Analysis
- Load valid English words from CSV files (Kaggle dataset included)
- Generate and analyze ALL letter permutations (5, 6, or 7 letters)
- Count valid words that can be formed from each permutation
- Compare distributions across different word lengths

### GPU Acceleration (NEW!)
- **CUDA/GPU support** for processing 3.5+ billion permutations
- Custom CUDA kernels using Numba for maximum efficiency
- Multi-GPU support for 8× H100 or similar hardware
- Process all 7-letter permutations (3.3 billion) in minutes
- CuPy + cuDF + Numba stack for optimal performance

### Visualizations
- Interactive D3.js visualizations:
  - Summary statistics tables
  - Overlay histograms comparing distributions
  - Individual distribution histograms
  - Box plot comparisons
- Matplotlib/Seaborn static plots
- CSV export for further analysis

### Modes
- **CPU Mode**: Sample-based analysis (original notebook)
- **GPU Mode**: Full permutation analysis (GPU notebook)

## Installation

This project uses `uv` for dependency management:

```bash
# Create virtual environment
uv venv --python-preference only-managed --seed

# Install dependencies (CPU mode)
uv sync --editable

# For GPU mode, install GPU libraries in your CUDA environment:
# uv pip install cupy-cuda12x numba cudf-cu12
```

## Usage

### GPU Mode (Recommended for 8× H100 Hardware)

Use `word_analysis_gpu.ipynb` for full permutation analysis:

```bash
# Launch Jupyter
uv run jupyter notebook word_analysis_gpu.ipynb

# The notebook will install GPU dependencies automatically:
# - cupy-cuda12x (GPU arrays)
# - numba (CUDA kernels)
# - cudf-cu12 (GPU DataFrames)
```

**Performance:**
- Processes ALL 3.5+ billion permutations (5, 6, 7-letter)
- With 8× H100 (80GB): Completes in minutes
- Single H100: Processes with batching, still very fast
- Requires CUDA 12.x compatible GPU

### CPU Mode (Original)

### Quick Start with Kaggle Dataset

The easiest way to get started is to use the included Kaggle English dictionary dataset:

```bash
# Download dataset and run analysis (uses Kaggle credentials from ~/.kaggle/kaggle.json)
uv run python main.py --download-dataset

# Or provide credentials directly (not recommended for security)
uv run python main.py --kaggle-username YOUR_USERNAME --kaggle-key YOUR_KEY
```

### Command Line Interface

```bash
# Basic usage with a CSV file containing valid words
uv run python main.py words.csv

# Specify word column name
uv run python main.py words.csv --word-column word

# Analyze specific word lengths
uv run python main.py words.csv --lengths 5 6

# Use a smaller sample size for faster analysis
uv run python main.py words.csv --sample-size 1000

# Use a subset of the alphabet
uv run python main.py words.csv --alphabet "aeiou"

# Export results to CSV
uv run python main.py words.csv --export-csv

# Custom output path
uv run python main.py words.csv --output my_analysis.html
```

### Jupyter Notebook

Open `word_analysis.ipynb` in Jupyter:

```bash
uv run jupyter notebook word_analysis.ipynb
```

The notebook provides:
- Step-by-step analysis workflow
- Configuration options
- Static matplotlib/seaborn visualizations
- Interactive D3.js visualizations (inline and HTML export)
- Statistical analysis
- Top/bottom permutation analysis
- CSV export

## Kaggle Dataset Setup

This project includes support for the [Dictionary of English Words and Definitions](https://www.kaggle.com/datasets/anthonytherrien/dictionary-of-english-words-and-definitions) Kaggle dataset.

### Option 1: Using kaggle.json (Recommended)

1. Get your Kaggle API credentials from https://www.kaggle.com/settings/account
2. Click "Create New API Token" to download `kaggle.json`
3. Place it in `~/.kaggle/kaggle.json`
4. Run: `uv run python main.py --download-dataset`

### Option 2: Direct Credentials

```bash
uv run python main.py --kaggle-username YOUR_USERNAME --kaggle-key YOUR_KEY
```

### Option 3: In Python/Jupyter

```python
from word_anal.kaggle_helper import get_dictionary_dataset

credentials = {
    "username": "your_username",
    "key": "your_api_key"
}

csv_path = get_dictionary_dataset(credentials=credentials)
```

**Important**: Never commit credentials to git! The `.gitignore` is configured to exclude credential files.

## Input Data Format

Your CSV file should contain valid English words. By default, the first column is used, or you can specify a column name:

```csv
word
apple
banana
cherry
...
```

The Kaggle dataset uses the following format:
```csv
word,definition
abbacy,"The word 'abbacy' refers to..."
abductor,"The word 'abductor' refers to..."
...
```

## Output

### Interactive HTML Visualization

The tool generates an HTML file with interactive D3.js visualizations:
- **Summary Statistics Table**: Mean, median, std dev, quartiles for each word length
- **Distribution Comparison**: Overlay histogram showing all distributions
- **Individual Distributions**: Separate histograms for each word length
- **Box Plot**: Statistical comparison with quartiles and outliers

### CSV Exports

When using `--export-csv`, the following files are generated:
- `permutation_analysis_5letter.csv`: All 5-letter permutation results
- `permutation_analysis_6letter.csv`: All 6-letter permutation results
- `permutation_analysis_7letter.csv`: All 7-letter permutation results
- `comparison_statistics.csv`: Summary statistics across all word lengths

## Python API

```python
from word_anal.analyzer import WordPermutationAnalyzer
from word_anal.data_processing import DataProcessor
from word_anal.visualizations import VisualizationGenerator

# Initialize analyzer
analyzer = WordPermutationAnalyzer(
    words_csv_path="words.csv",
    word_column=None  # Use first column
)

# Run analysis
results = analyzer.compare_word_lengths(
    lengths=[5, 6, 7],
    sample_size=10000,
    alphabet_subset=None
)

# Process and visualize
processor = DataProcessor()
for word_length, df in results.items():
    processor.add_results(word_length, df)

viz_gen = VisualizationGenerator(processor)
viz_gen.generate_html("output.html")
```

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **polars**: Fast CSV reading
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **jupyter**: Notebook interface
- **D3.js**: Interactive web visualizations

## How It Works

1. **Load Words**: Reads valid English words from CSV using Polars for speed
2. **Generate Permutations**: Creates all possible permutations of N letters from the alphabet
3. **Match Words**: For each permutation, counts how many valid words can be formed using those letters
4. **Statistical Analysis**: Calculates mean, median, standard deviation, skewness, kurtosis, and quartiles
5. **Visualization**: Creates interactive charts comparing distributions across word lengths

## Performance Notes

- Analysis time grows significantly with word length (5! = 120 vs 7! = 5,040 permutations per letter combo)
- Use `--sample-size` to limit permutations analyzed for faster results
- Polars is used for fast CSV reading of large word lists
- Consider using `--alphabet` to analyze smaller subsets

## Example Analysis

With a standard English dictionary:
- **5-letter permutations**: Average ~2-3 valid words per permutation
- **6-letter permutations**: Average ~1-2 valid words per permutation
- **7-letter permutations**: Average ~0.5-1 valid words per permutation

Distributions tend to be right-skewed, with most permutations producing few valid words and a long tail of high-producing permutations.

## License

MIT
