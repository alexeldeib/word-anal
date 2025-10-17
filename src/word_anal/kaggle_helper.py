"""Helper functions for Kaggle dataset download and authentication."""

import os
import json
from pathlib import Path
from typing import Optional, Dict


def setup_kaggle_credentials(username: str, key: str, credentials_dir: Optional[str] = None):
    """
    Set up Kaggle API credentials.

    Args:
        username: Kaggle username
        key: Kaggle API key
        credentials_dir: Directory to store credentials (default: ~/.kaggle)

    IMPORTANT: Do not commit credentials to git. Use environment variables or
    secure configuration files that are gitignored.
    """
    if credentials_dir is None:
        credentials_dir = os.path.expanduser("~/.kaggle")

    os.makedirs(credentials_dir, exist_ok=True)
    credentials_path = os.path.join(credentials_dir, "kaggle.json")

    credentials = {
        "username": username,
        "key": key
    }

    with open(credentials_path, 'w') as f:
        json.dump(credentials, f)

    # Set proper permissions (Kaggle API requires this)
    os.chmod(credentials_path, 0o600)

    print(f"Kaggle credentials saved to {credentials_path}")


def load_kaggle_credentials_from_dict(credentials: Dict[str, str]):
    """
    Load Kaggle credentials from a dictionary.

    Args:
        credentials: Dictionary with 'username' and 'key'

    Example:
        credentials = {"username": "yourname", "key": "yourkey"}
        load_kaggle_credentials_from_dict(credentials)
    """
    setup_kaggle_credentials(
        username=credentials["username"],
        key=credentials["key"]
    )


def download_kaggle_dataset(
    dataset_name: str,
    download_path: str = "data",
    force: bool = False,
    unzip: bool = True
) -> Path:
    """
    Download a Kaggle dataset.

    Args:
        dataset_name: Kaggle dataset identifier (e.g., 'username/dataset-name')
        download_path: Directory to download to (default: 'data')
        force: Force re-download even if files exist
        unzip: Unzip downloaded files

    Returns:
        Path to downloaded dataset directory

    Example:
        download_kaggle_dataset('anthonytherrien/dictionary-of-english-words-and-definitions')
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Create download directory
    download_dir = Path(download_path)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists
    dataset_dir = download_dir / dataset_name.split('/')[-1]
    if dataset_dir.exists() and not force:
        print(f"Dataset already exists at {dataset_dir}")
        return dataset_dir

    print(f"Downloading Kaggle dataset: {dataset_name}")
    print(f"Destination: {download_dir}")

    # Download dataset
    api.dataset_download_files(
        dataset=dataset_name,
        path=str(download_dir),
        unzip=unzip,
        force=force,
        quiet=False
    )

    print(f"Download complete: {dataset_dir}")
    return dataset_dir


def get_dictionary_dataset(
    credentials: Optional[Dict[str, str]] = None,
    download_path: str = "data",
    force: bool = False
) -> Path:
    """
    Download and get path to the English dictionary dataset.

    Args:
        credentials: Optional Kaggle credentials dict {"username": "...", "key": "..."}
        download_path: Directory to download to
        force: Force re-download

    Returns:
        Path to the CSV file with words and definitions
    """
    # Set up credentials if provided
    if credentials:
        load_kaggle_credentials_from_dict(credentials)

    # Download dataset
    dataset_name = "anthonytherrien/dictionary-of-english-words-and-definitions"
    dataset_dir = download_kaggle_dataset(
        dataset_name=dataset_name,
        download_path=download_path,
        force=force
    )

    # Find the CSV file
    csv_files = list(Path(download_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {download_path}")

    csv_file = csv_files[0]
    print(f"Dictionary CSV: {csv_file}")
    print(f"File size: {csv_file.stat().st_size / (1024*1024):.2f} MB")

    return csv_file


def inspect_dataset(csv_path: Path, n_rows: int = 5):
    """
    Inspect the structure of the dataset.

    Args:
        csv_path: Path to CSV file
        n_rows: Number of rows to display
    """
    import pandas as pd

    print(f"\nInspecting dataset: {csv_path}")
    print("=" * 60)

    df = pd.read_csv(csv_path, nrows=1000)

    print(f"\nColumns: {list(df.columns)}")
    print(f"Total rows (sample): {len(df)}")
    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows))

    print(f"\nColumn types:")
    print(df.dtypes)

    print(f"\nNull values:")
    print(df.isnull().sum())

    # Show word length distribution if 'word' column exists
    word_column = None
    for col in df.columns:
        if 'word' in col.lower():
            word_column = col
            break

    if word_column:
        df['word_length'] = df[word_column].str.len()
        print(f"\nWord length distribution:")
        print(df['word_length'].describe())
        print(f"\n5-letter words: {len(df[df['word_length'] == 5]):,}")
        print(f"6-letter words: {len(df[df['word_length'] == 6]):,}")
        print(f"7-letter words: {len(df[df['word_length'] == 7]):,}")
