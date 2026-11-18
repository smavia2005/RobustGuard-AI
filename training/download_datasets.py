
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"

DATASETS.mkdir(exist_ok=True)

def download_file(repo, filename, destination):
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading: {filename}")

    source = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset"
    )

    target = destination / Path(filename).name
    shutil.copy2(source, target)

    print(f"Saved: {target}")
    return target


# IMAGE DATASET - CIFAKE
image_repo = "dragonintelligence/CIFAKE-image-dataset"

download_file(
    image_repo,
    "data/train-00000-of-00001.parquet",
    DATASETS / "image"
)

download_file(
    image_repo,
    "data/test-00000-of-00001.parquet",
    DATASETS / "image"
)


# TEXT DATASET - HC3
download_file(
    "Hello-SimpleAI/HC3",
    "all.jsonl",
    DATASETS / "text"
)

print("\nDATASET DOWNLOAD COMPLETE")
print("Image and Text datasets are ready.")
print("Video and Audio datasets require separate handling.")
