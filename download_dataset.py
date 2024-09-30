import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    api = KaggleApi()
    api.authenticate()

    os.makedirs("data", exist_ok=True)
    name = "rsna-2024-lumbar-spine-degenerative-classification"

    # Download a dataset (replace with the dataset you want)
    api.competition_download_files(name, path="data/", force=True, quiet=False)

    # unzip the dataset
    with zipfile.ZipFile(f"data/{name}.zip", "r") as zip_ref:
        zip_ref.extractall(f"data/{name}")


if __name__ == "__main__":
    main()
    # %%
