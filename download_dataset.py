import argparse
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


def main(args):
    api = KaggleApi()
    api.authenticate()

    # Download a dataset (replace with the dataset you want)
    api.competition_download_files(name, path="data/", force=True, quiet=False)
    # unzip the dataset

    with zipfile.ZipFile(f"data/{name}.zip", "r") as zip_ref:
        zip_ref.extractall("data/isic-2024-challenge")


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--competition", "-c", type=str, required=True, help="Competition name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert (args.dataset and not args.competition) or (
        not args.dataset and args.competition
    ), "Please provide either dataset or competition name"
    main(args)
