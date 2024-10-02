import argparse
import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi


def version_kaggle_dataset(
    dataset_dir,
    version_note="Updated models with improved accuracy.",
    title="Two-Stage Machine Learning Models Dataset",
    id=None,  # Format: 'your_username/two-stage-model-dataset'
    licenses=[{"name": "CC0-1.0"}],
    keywords=["machine learning", "two-stage model", "PyTorch", "models"],
    description=(
        "This dataset contains two PyTorch models for a two-stage machine learning pipeline: "
        "first_stage_best_model.pth and second_stage_best_model.pth."
    ),
):
    """
    Versions an existing Kaggle dataset by generating/updating the metadata file and uploading the new version.

    Parameters:
    - dataset_dir (str): Path to the dataset directory containing data and metadata.
    - version_note (str): Description of the changes in this version.
    - title (str): Title of the dataset.
    - id (str): Unique identifier in the format 'your_username/two-stage-model-dataset'.
                If None, it will be inferred from the Kaggle API configuration.
    - licenses (list): List of license dictionaries, e.g., [{"name": "CC0-1.0"}].
    - keywords (list): List of keywords for the dataset.
    - description (str): Detailed description of the dataset.

    Returns:
    - None
    """
    api = KaggleApi()
    api.authenticate()

    # Verify that dataset_dir exists
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The dataset directory '{dataset_dir}' does not exist.")

    # Gather all file paths
    resources = []
    for root, dirs, files in os.walk(dataset_dir):
        # Optionally skip hidden directories/files
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.startswith("."):
                continue  # Skip hidden files
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, dataset_dir)
            resources.append({"path": relative_path})

    # Check if 'id' is provided
    if not id:
        # Infer from the current user and dataset name based on the dataset_dir
        username = api.user()["username"]
        dataset_name = (
            os.path.basename(os.path.abspath(dataset_dir)).replace(" ", "-").lower()
        )
        id = f"{username}/{dataset_name}"

    # Create metadata dictionary
    metadata = {
        "title": title,
        "id": id,
        "licenses": licenses,
        "keywords": keywords,
        "description": description,
        "resources": resources,
    }

    # Write dataset-metadata.json
    metadata_path = os.path.join(dataset_dir, "dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated 'dataset-metadata.json' at {metadata_path}")

    # Version the dataset on Kaggle
    try:
        api.dataset_create_version(
            folder=dataset_dir,
            version_notes=version_note,
            convert_to_csv=False,
            dir_mode="zip",  # Options: 'tar', 'zip', 'file'
        )
        print("Dataset versioned successfully on Kaggle!")
    except Exception as e:
        print(f"An error occurred while versioning the dataset: {e}")


def create_kaggle_dataset(
    dataset_dir,
    title="Two-Stage Machine Learning Models Dataset",
    id=None,  # Format: 'your_username/two-stage-model-dataset'
    licenses=[{"name": "CC0-1.0"}],
    keywords=["machine learning", "two-stage model", "PyTorch", "models"],
    description=(
        "This dataset contains two PyTorch models for a two-stage machine learning pipeline: "
        "first_stage_best_model.pth and second_stage_best_model.pth."
    ),
    public=True,
):
    """
    Creates a new Kaggle dataset by generating the metadata file and uploading the dataset.

    Parameters:
    - dataset_dir (str): Path to the dataset directory containing data and metadata.
    - title (str): Title of the dataset.
    - id (str): Unique identifier in the format 'your_username/two-stage-model-dataset'.
                If None, it will be inferred from the Kaggle API configuration.
    - licenses (list): List of license dictionaries, e.g., [{"name": "CC0-1.0"}].
    - keywords (list): List of keywords for the dataset.
    - description (str): Detailed description of the dataset.
    - public (bool): If True, the dataset is public. Otherwise, it's private.

    Returns:
    - None
    """
    api = KaggleApi()
    api.authenticate()

    # Verify that dataset_dir exists
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The dataset directory '{dataset_dir}' does not exist.")

    # Gather all file paths
    resources = []
    for root, dirs, files in os.walk(dataset_dir):
        # Optionally skip hidden directories/files
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.startswith("."):
                continue  # Skip hidden files
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, dataset_dir)
            resources.append({"path": relative_path})

    # Check if 'id' is provided
    if not id:
        # Infer from the current user and dataset name based on the dataset_dir
        username = api.user()["username"]
        dataset_name = (
            os.path.basename(os.path.abspath(dataset_dir)).replace(" ", "-").lower()
        )
        id = f"{username}/{dataset_name}"

    # Create metadata dictionary
    metadata = {
        "title": title,
        "id": id,
        "licenses": licenses,
        "keywords": keywords,
        "description": description,
        "resources": resources,
    }

    # Write dataset-metadata.json
    metadata_path = os.path.join(dataset_dir, "dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated 'dataset-metadata.json' at {metadata_path}")

    # Create the dataset on Kaggle
    try:
        api.dataset_create_new(
            folder=dataset_dir,
            convert_to_csv=False,
            public=public,
            dir_mode="zip",  # Options: 'tar', 'zip', 'file'
        )
        print("Dataset created successfully on Kaggle!")
    except Exception as e:
        print(f"An error occurred while creating the dataset: {e}")


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Manage Kaggle datasets for two-stage PyTorch models."
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        description="Valid commands",
        help="Additional help",
        dest="command",
        required=True,
    )

    # Create Dataset Subparser
    create_parser = subparsers.add_parser("create", help="Create a new Kaggle dataset.")
    create_parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset directory containing models and metadata.",
    )
    create_parser.add_argument(
        "--title",
        type=str,
        default="Two-Stage RSNA Models Dataset",
        help="Title of the dataset.",
    )
    create_parser.add_argument(
        "--id",
        type=str,
        default="patrico49/two-stage-model-rsna-dataset",
        help="Unique identifier in the format 'your_username/two-stage-model-dataset'. If not provided, it will be inferred.",
    )
    create_parser.add_argument(
        "--public",
        action="store_true",
        help="If set, the dataset will be public. Default is private.",
    )
    create_parser.add_argument(
        "--licenses",
        type=json.loads,
        default='[{"name": "CC0-1.0"}]',
        help='List of license dictionaries, e.g., \'[{"name": "CC0-1.0"}]\'.',
    )
    create_parser.add_argument(
        "--keywords",
        type=json.loads,
        default='["machine learning", "two-stage model", "PyTorch", "models"]',
        help="List of keywords for the dataset.",
    )
    create_parser.add_argument(
        "--description",
        type=str,
        default=(
            "This dataset contains two PyTorch models for a two-stage machine learning pipeline: "
            "first_stage_best_model.pth and second_stage_best_model.pth."
        ),
        help="Detailed description of the dataset.",
    )

    # Version Dataset Subparser
    version_parser = subparsers.add_parser(
        "version", help="Version an existing Kaggle dataset."
    )
    version_parser.add_argument(
        "--dataset_dir",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset directory containing models and metadata.",
    )
    version_parser.add_argument(
        "--version_note",
        type=str,
        required=True,
        help="Description of the changes in this version.",
    )
    version_parser.add_argument(
        "--title",
        type=str,
        default="Two-Stage RSNA Models Dataset",
        help="Title of the dataset.",
    )
    version_parser.add_argument(
        "--id",
        type=str,
        default="patrico49/two-stage-model-rsna-dataset",
        help="Unique identifier in the format 'your_username/dataset-name'. If not provided, it will be inferred.",
    )
    version_parser.add_argument(
        "--licenses",
        type=json.loads,
        default='[{"name": "CC0-1.0"}]',
        help='List of license dictionaries, e.g., \'[{"name": "CC0-1.0"}]\'.',
    )
    version_parser.add_argument(
        "--keywords",
        type=json.loads,
        default='["machine learning", "two-stage model", "PyTorch", "models"]',
        help="List of keywords for the dataset.",
    )
    version_parser.add_argument(
        "--description",
        type=str,
        default=(
            "This dataset contains two PyTorch models for a two-stage machine learning pipeline: "
            "first_stage_best_model.pth and second_stage_best_model.pth."
        ),
        help="Detailed description of the dataset.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.command == "create":
        create_kaggle_dataset(
            dataset_dir=args.dataset_dir,
            title=args.title,
            id=args.id,
            public=args.public,
        )
    elif args.command == "version":
        # For versioning, we might need to handle additional parameters like version_note
        version_kaggle_dataset(
            dataset_dir=args.dataset_dir,
            version_note=args.version_note,
            title=args.title,
            id=args.id,
        )
    else:
        print("Invalid command. Use 'create' or 'version'.")


if __name__ == "__main__":
    main()
