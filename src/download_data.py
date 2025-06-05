import kagglehub
import shutil
import os


def download_data(destination: str = "data/raw") -> str:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("uciml/adult-census-income")

    os.makedirs(destination, exist_ok=True)
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            shutil.copy(full_path, destination)

    print(f"Dataset downloaded to: {destination}")
    return destination


if __name__ == "__main__":
    download_data()
