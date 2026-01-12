from datasets import load_dataset
from pathlib import Path

DATASETS = [
    "nasa-ibm-ai4science/Surya-bench-solarwind",
    "nasa-ibm-ai4science/surya-bench-flare-forecasting",
    "nasa-ibm-ai4science/surya-bench-ar-segmentation",
    "nasa-ibm-ai4science/euv-spectra",
    "nasa-ibm-ai4science/surya-bench-coronal-extrapolation",
    "nasa-ibm-ai4science/ar_emergence",
]

BASE_DIR = Path("/tmp/huggingface_data")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_name: str):
    dataset_dir = BASE_DIR / dataset_name.replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} ...")

    # This triggers the full download into the Hugging Face cache
    ds = load_dataset(dataset_name)

    # Save a local copy in Arrow format for reproducibility
    ds.save_to_disk(dataset_dir)

    print(f"Saved to {dataset_dir}")


if __name__ == "__main__":
    for ds_name in DATASETS:
        download_dataset(ds_name)

    print("All datasets downloaded successfully.")
