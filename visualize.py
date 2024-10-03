# %%
import datetime
import os

import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from src.model.model import (
    AverageMeter,
    FirstStageModel,
    LumbarSpineDataset,
    SecondStageModelV2,
    custom_collate_fn,
    dotdict,
    filter_by_study_ids,
    logger,
    logger_setup,
    split_study_ids,
    visualize_predictions_and_crop,
)
from torch.utils.data import DataLoader

# %%

logger_setup()

# %%

base_path = "data/rsna-2024-lumbar-spine-degenerative-classification"
base_model_path = "models"

args = dotdict(all=False, max_depth=50, train_on=["zxy", "grade"], stage=2)

today_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = f"checkpoints_{today_str}"
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
    )

    val_dataset = LumbarSpineDataset(
        image_dir=os.path.join(base_path, "train_images"),
        label=val_labels,
        coords=val_coords,
        series=val_series,
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
model = FirstStageModel(train_on=args.train_on)
state_dict = torch.load(
    f"{base_model_path}/first_stage_best_model.pth",
    map_location=lambda storage, loc: storage,
    weights_only=True,
)

print(model.load_state_dict(state_dict, strict=False))  # True
model = model.to(torch_device)
if args.stage == 2:
    model = SecondStageModelV2(model, pretrained=True)
    model = model.to(torch_device)


def get_output(model, batch_input, debug=False):
    if debug:
        with torch.no_grad():
            return model(batch_input)
    else:
        return model(batch_input)


# %%
needed_examples = ["sagittal t2", "sagittal t1", "axial"]
for batch_idx, batch in enumerate(train_loader):
    image = batch["image"].to(torch_device, non_blocking=True)
    D = batch["D"].to(torch_device, non_blocking=True)
    heatmap_target = batch["heatmap"].to(torch_device, non_blocking=True)
    z_target = batch["z"].to(torch_device, non_blocking=True)
    xy_target = batch["xy"].to(torch_device, non_blocking=True)
    grade_target = batch["grade"].to(torch_device, non_blocking=True)

    batch_input = {
        "image": image,
        "D": D,
        "heatmap": heatmap_target,
        "z": z_target,
        "xy": xy_target,
        "grade": grade_target,
        "study_id": batch["study_id"],
        "series_description": batch["series_description"],
    }
    visualize_predictions_and_crop(
        model,
        batch_input,
        output_dir=f"/mnt/d/plots/{batch['study_id'][0]}/{batch['series_description'][0]}",
    )
    if any([x in batch["series_description"][0].lower() for x in needed_examples]):
        # remove from needed examples
        needed_examples = [
            x
            for x in needed_examples
            if x not in batch["series_description"][0].lower()
        ]
        logger.info(f"Remaining needed examples: {needed_examples}")
    if len(needed_examples) == 0:
        break
# %%
