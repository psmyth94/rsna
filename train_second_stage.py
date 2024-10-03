# %%
import datetime
import os

import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
)

# %%


logger_setup()

# %%

base_path = "data/rsna-2024-lumbar-spine-degenerative-classification"
base_model_path = "models"

args = dotdict(all=False, max_depth=50, train_on=["zxy", "grade"], stage=1)

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

batch_size = 2
accumulation_steps = 1  # Gradient accumulation steps

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
model = FirstStageModel(train_on=args.train_on, dynamic_matching=True)
state_dict = torch.load(
    f"{base_model_path}/first_stage_best_model.pth",
    map_location=lambda storage, loc: storage,
    weights_only=True,
)

print(model.load_state_dict(state_dict, strict=False))  # True
model = model.to(torch_device)
if args.stage == 2:
    model = SecondStageModelV2(model, pretrained=True, backbone="efficientnet_b0")
    model = model.to(torch_device)

# %%
DEBUG = False
log_frequency = 1
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
)

num_epochs = 100

# Calculate the total number of steps for OneCycleLR considering gradient accumulation
steps_per_epoch = len(train_loader) // accumulation_steps
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-1,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=10000.0,
)

# Initialize GradScaler for mixed-precision training
scaler = torch.amp.GradScaler(device)

logger.info("Starting training...")

best_val_loss = float("inf")  # Initialize best validation loss

max_norm = 1.0

for epoch in range(num_epochs):
    if DEBUG:
        model.eval()
    else:
        model.train()

    losses = {
        "loss": AverageMeter(),
        "z_loss": AverageMeter(),
        "xy_loss": AverageMeter(),
        "grade_loss": AverageMeter(),
        "heatmap_loss": AverageMeter(),
        "val_loss": AverageMeter(),
        "val_z_loss": AverageMeter(),
        "val_xy_loss": AverageMeter(),
        "val_grade_loss": AverageMeter(),
        "val_heatmap_loss": AverageMeter(),
    }

    optimizer.zero_grad()  # Reset gradients at the start of each epoch

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
        }

        def get_output(model, batch_input):
            if DEBUG:
                with torch.no_grad():
                    return model(batch_input)
            else:
                return model(batch_input)

        outputs = {}
        with torch.amp.autocast(device):
            outputs = get_output(model, batch_input)
            # heatmap_loss = outputs["heatmap_loss"]
            loss = torch.tensor(0.0, device=torch_device)
            if "z_loss" in outputs:
                loss += outputs["z_loss"]
                losses["z_loss"].update(outputs["z_loss"].item(), batch_size)
            if "xy_loss" in outputs:
                loss += outputs["xy_loss"]
                losses["xy_loss"].update(outputs["xy_loss"].item(), batch_size)
            if "grade_loss" in outputs:
                loss += outputs["grade_loss"]
                losses["grade_loss"].update(outputs["grade_loss"].item(), batch_size)
            if "heatmap_loss" in outputs:
                loss += outputs["heatmap_loss"]
                losses["heatmap_loss"].update(
                    outputs["heatmap_loss"].item(), batch_size
                )
            # Normalize loss by accumulation_steps
            losses["loss"].update(loss.item(), batch_size)
            loss = loss / accumulation_steps

        if not DEBUG:
            # Scale the loss and perform backward pass
            scaler.scale(loss).backward()

            # Perform optimizer step and scheduler step every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                # scaler.unscale_(optimizer)
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                # Step optimizer and scaler
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients
                optimizer.zero_grad()

                # Step scheduler
                scheduler.step()
                if hasattr(model, "_ascension_callback"):
                    model._ascension_callback()

        # Logging
        if (batch_idx + 1) % (log_frequency * accumulation_steps) == 0 or (
            batch_idx + 1
        ) == len(train_loader):
            current_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], "
                f"LR: {current_lr:.6f}, "
                f"Loss: {losses['loss'].avg:.4f}, "
            )
            for key in outputs.keys():
                if key in losses:
                    msg += f"{key.capitalize()}: {losses[key].avg:.4f}, "
            logger.info(msg)
            if DEBUG:
                break

    # End of epoch logging
    epoch_loss = losses["loss"].avg
    msg = (
        f"Epoch [{epoch + 1}/{num_epochs}] completed. "
        f"Average Loss: {epoch_loss:.4f}, "
    )
    for key in outputs.keys():
        if key in losses:
            msg += f"Average {key.capitalize()}: {losses[key].avg:.4f}, "

    logger.info(msg)

    if val_loader is not None:
        # Validation looptrain
        model.eval()
        val_outputs = {}
        with torch.amp.autocast(device):
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_loader):
                    image = val_batch["image"].to(torch_device)
                    D = val_batch["D"].to(torch_device)
                    heatmap_target = val_batch.get("heatmap", None)
                    z_target = val_batch.get("z", None)
                    xy_target = val_batch.get("xy", None)
                    grade_target = val_batch["grade"].to(torch_device)

                    val_batch_input = {
                        "image": image,
                        "D": D,
                        "heatmap": heatmap_target,
                        "z": z_target,
                        "xy": xy_target,
                        "grade": grade_target,
                    }

                    val_outputs = model(val_batch_input)

                    val_loss = torch.tensor(0.0, device=torch_device)
                    if "z_loss" in val_outputs:
                        val_loss += val_outputs["z_loss"]
                        losses["val_z_loss"].update(
                            val_outputs["z_loss"].item(), batch_size
                        )
                    if "xy_loss" in val_outputs:
                        val_loss += val_outputs["xy_loss"]
                        losses["val_xy_loss"].update(
                            val_outputs["xy_loss"].item(), batch_size
                        )
                    if "grade_loss" in val_outputs:
                        val_loss += val_outputs["grade_loss"]
                        losses["val_grade_loss"].update(
                            val_outputs["grade_loss"].item(), batch_size
                        )
                    if "heatmap_loss" in val_outputs:
                        val_loss += val_outputs["heatmap_loss"]
                        losses["val_heatmap_loss"].update(
                            val_outputs["heatmap_loss"].item(), batch_size
                        )
                    losses["val_loss"].update(val_loss.item(), batch_size)

                    # Validation logging
                    if (batch_idx + 1) % (log_frequency * accumulation_steps) == 0 or (
                        batch_idx + 1
                    ) == len(val_loader):
                        msg = (
                            f"Validation: Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(val_loader)}], "
                            f"Validation Loss: {losses['val_loss'].avg:.4f}, "
                        )
                        for key in val_outputs.keys():
                            val_key = f"val_{key}"
                            if val_key in losses:
                                msg += f"Validation {key.capitalize()}: {losses[val_key].avg:.4f}, "

                        logger.info(msg)
                        if DEBUG:
                            break

        avg_val_loss = losses["val_loss"].avg
        msg = f"Validation Loss: {avg_val_loss:.4f}, "
        for key in val_outputs:
            val_key = f"val_{key}"
            if val_key in losses:
                msg += f"Validation {key.capitalize()}: {losses[val_key].avg:.4f}, "
        logger.info(msg)
    else:
        avg_val_loss = epoch_loss

    if not DEBUG:
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),  # Save scaler state
                "loss": avg_val_loss,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved at {checkpoint_path}")

    if not DEBUG:
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"Saved new best model with validation loss {best_val_loss:.4f}"
            )
    if DEBUG:
        break

if not DEBUG:
    # Save the final model at the end of training
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}")

logger.info("Training finished.")
# %%
