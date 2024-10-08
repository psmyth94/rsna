# %%
import datetime
import os

import pandas as pd
import torch
import torch.amp
from torch.utils.data import DataLoader

from src.rsna.rsna import (
    FirstStageModel,
    LumbarSpineDataset,
    SecondStageModelV2,
    custom_collate_fn,
    dotdict,
    filter_by_study_ids,
    logger,
    logger_setup,
    split_study_ids,
)
from src.rsna.train import train


def main(args):

    logger_setup()

    base_path = "data/rsna-2024-lumbar-spine-degenerative-classification"
    base_model_path = "models"

    today_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = f"checkpoints_sagittal_t2_stir_{today_str}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    logger.info(f"Using device: {device}")

    logger.info("Loading data...")

    series_df = pd.read_csv(os.path.join(base_path, "train_series_descriptions.csv"))
    label_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    coords_df = pd.read_csv(os.path.join(base_path, "train_label_coordinates.csv"))

    train_study_ids, val_study_ids = split_study_ids(series_df)

    batch_size = 1
    accumulation_steps = 4  # Gradient accumulation steps

    if not args.all:
        logger.info("Splitting data into training and validation sets...")
        train_series = filter_by_study_ids(series_df, train_study_ids)
        val_series = filter_by_study_ids(series_df, val_study_ids)

        train_labels = filter_by_study_ids(label_df, train_study_ids)
        val_labels = filter_by_study_ids(label_df, val_study_ids)

        train_coords = filter_by_study_ids(coords_df, train_study_ids)
        val_coords = filter_by_study_ids(coords_df, val_study_ids)

        logger.info("Creating datasets and data loaders...")
        train_dataset = LumbarSpineDataset(
            image_dir=os.path.join(base_path, "train_images"),
            label=train_labels,
            coords=train_coords,
            series=train_series,
            transform=None,
            train_on=args.train_on,
            max_depth=50 if args.max_depth is None else args.max_depth,
            series_type="sagittal t2/stir",
        )

        val_dataset = LumbarSpineDataset(
            image_dir=os.path.join(base_path, "train_images"),
            label=val_labels,
            coords=val_coords,
            series=val_series,
            transform=None,
            train_on=args.train_on,
            max_depth=50 if args.max_depth is None else args.max_depth,
            series_type="sagittal t2/stir",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
    else:
        logger.info("Creating dataset and data loader...")
        train_dataset = LumbarSpineDataset(
            image_dir=os.path.join(base_path, "train_images"),
            label=label_df,
            coords=coords_df,
            series=series_df,
            transform=None,
            train_on=args.train_on,
            max_depth=50 if args.max_depth is None else args.max_depth,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
        val_loader = None

    # %%
    logger.info("Creating model and optimizer...")
    model = FirstStageModel(
        train_on=args.train_on, dynamic_matching=False, pretrained=True
    )
    # state_dict = torch.load(
    #     f"{base_model_path}/first_stage_best_model.pth",
    #     map_location=lambda storage, loc: storage,
    #     weights_only=True,
    # )

    # print(model.load_state_dict(state_dict, strict=False))  # True
    model = model.to(torch_device)
    # %%
    if args.stage == 2:
        model = SecondStageModelV2(model, pretrained=True, crop_size=64, depth_size=10)
        model = model.to(torch_device)

    # %%
    train(
        logger,
        model,
        train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=11,
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max_depth", type=int, default=50)
    parser.add_argument("--train_on", nargs="+", default=["zxy", "grade"])
    parser.add_argument("--stage", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
