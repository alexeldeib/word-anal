# Setup Instructions for 8× H100 Machine

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ace-cohere/word-anal.git
cd word-anal

# 2. Set up Kaggle credentials (use your own!)
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"

# 3. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Create virtual environment and install base dependencies
uv venv --python-preference only-managed --seed
source .venv/bin/activate  # or: .venv/bin/activate on Linux
uv sync --editable

# 5. Launch Jupyter
uv run jupyter notebook word_analysis_gpu.ipynb
```

## What the Notebook Will Do

The notebook will automatically:
1. Install GPU dependencies via `uv pip install cupy-cuda12x numba cudf-cu12`
2. Detect your 8 GPUs
3. Download the Kaggle English dictionary dataset
4. Process ALL 3.5+ billion permutations
5. Generate visualizations

## System Requirements

- **GPUs**: 8× NVIDIA H100 (80GB each) with CUDA 12.x
- **RAM**: 2TB host memory
- **Python**: 3.12+
- **CUDA**: 12.x drivers installed

## Expected Performance

With 8× H100 GPUs:
- **5-letter permutations** (7.9M): < 1 minute
- **6-letter permutations** (165M): 2-5 minutes
- **7-letter permutations** (3.3B): 10-30 minutes

**Total**: All 3.5+ billion permutations in ~30-40 minutes

## GPU Memory Usage

- **5-letter**: 205 MB per GPU
- **6-letter**: 4.3 GB per GPU
- **7-letter**: ~11 GB per GPU (distributed across 8 GPUs)
- **Total across 8 GPUs**: ~90 GB used out of 640 GB available

## Troubleshooting

### If cuDF fails to install:
```bash
# cuDF might not be available for macOS, but it should work on Linux with CUDA
# If it fails, the code will fall back to pandas for CSV reading
```

### If you want to test with fewer GPUs:
Open the notebook and set:
```python
N_GPUS = 1  # Use only 1 GPU
```

### Memory issues:
If you run into memory issues, adjust the batch size:
```python
batch_size_per_gpu=25_000_000  # Reduce from 50M to 25M
```

## Output Files

After running, you'll have:
- `gpu_analysis_summary.csv` - Summary statistics
- `gpu_distribution_comparison.png` - Distribution plots
- `gpu_boxplot_comparison.png` - Box plot comparison
- `word_analysis_gpu_full.html` - Interactive D3.js visualization

## Alternative: Using .env file

Instead of exporting environment variables, you can create a `.env` file:

```bash
cat > .env <<EOF
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
EOF
```

The notebook will automatically read from environment variables.

## Getting Kaggle Credentials

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Use the username and key from that file

## Need Help?

- GitHub Issues: https://github.com/ace-cohere/word-anal/issues
- See README.md for more details
