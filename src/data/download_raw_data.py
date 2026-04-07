from pathlib import Path
import os
from dotenv import load_dotenv
import kagglehub  # make sure you have this installed

def download_chexpert_dataset(target_dir: str = "src/data") -> Path:
    """
    Downloads CheXpert dataset from Kaggle and extracts it directly to target_dir.

    Args:
        target_dir (str): Relative path to store dataset

    Returns:
        Path: Final dataset path
    """

    # Load environment variables from .env
    load_dotenv()

    # Validate credentials
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise ValueError(
            "Missing Kaggle credentials. Make sure .env contains "
            "KAGGLE_USERNAME and KAGGLE_KEY"
        )

    # Resolve target directory
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    # This downloads the dataset as a zip (or folder depending on kagglehub version)
    download_path = Path(kagglehub.dataset_download("ashery/chexpert"))
    print(f"Downloaded to: {download_path}")

    # If it's a zip file, extract directly to target folder
    if download_path.suffix == ".zip":
        import zipfile
        print("Extracting dataset...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        print("Extraction complete.")
    else:
        # If it's already extracted, move contents (fast) into target_path
        import shutil
        shutil.move(str(download_path), str(target_path))

    print("Dataset ready at:", target_path)
    return target_path

if __name__ == "__main__":
    download_chexpert_dataset()