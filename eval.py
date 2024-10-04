import logging
import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.amp
from torch.utils.data import DataLoader

from src.rsna.rsna import (
    FirstStageModel,
    LumbarSpineDataset,
    SecondStageModel,
    custom_collate_fn,
    dotdict,
    logger_setup,
)

logger = logging.getLogger()


logger_setup()
# Base path to your dataset
base_path = "data/rsna-2024-lumbar-spine-degenerative-classification"
base_model_path = "models"

args = dotdict(all=False, max_depth=50, train_on=["zxy", "grade"], stage=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device)
logger.info(f"Using device: {device}")


# Initialize the FirstStageModel
first_stage_model = FirstStageModel(train_on=args.train_on)
first_stage_state_dict = torch.load(
    f"{base_model_path}/first_stage_best_model.pth",
    map_location=torch_device,
    weights_only=True,
)
first_stage_model.load_state_dict(first_stage_state_dict)
first_stage_model = first_stage_model.to(torch_device)

# Initialize the SecondStageModel
second_stage_model = SecondStageModel(first_stage_model, pretrained=False)
second_stage_state_dict = torch.load(
    f"{base_model_path}/second_stage_best_model.pth",
    map_location=torch_device,
    weights_only=True,
)
second_stage_model.load_state_dict(second_stage_state_dict)
second_stage_model = second_stage_model.to(torch_device)

# Set models to evaluation mode
first_stage_model.eval()
second_stage_model.eval()

# Set output_type to ["infer"] for inference
first_stage_model.output_type = ["infer"]
second_stage_model.output_type = ["infer"]

# Prepare the test dataset and data loader
logger.info("Loading test data...")
test_series_df = pd.read_csv(os.path.join(base_path, "test_series_descriptions.csv"))

# %%


# Modify LumbarSpineDataset to handle test data
class TestLumbarSpineDataset(LumbarSpineDataset):
    def __init__(self, *args, **kwargs):
        super(TestLumbarSpineDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx) -> Union[dict, torch.Tensor]:
        return self._get_all(idx)

    def _get_all(self, idx, count=0) -> Union[dict, torch.Tensor]:
        sample_info = self.samples[idx]
        try:
            instance_numbers_needed = set()  # No labels in test data

            # Read the series and process the volume
            volume, dicom_df, error_code = self._read_series(
                sample_info["study_id"],
                sample_info["series_id"],
                sample_info["series_description"],
                instance_numbers_needed,
            )

            # Prepare the image tensor
            # The volume is in shape (D, H, W), we need to convert it to (C, H, W)
            image = np.ascontiguousarray(volume.transpose(1, 2, 0))  # Shape: (H, W, D)

            # Resize and center the image
            image, scale_param = self.do_resize_and_center(image, reference_size=320)
            image = np.ascontiguousarray(image.transpose(2, 0, 1))  # Shape: (D, H, W)

            # Convert to torch tensor
            image = torch.from_numpy(image).float().half()

            if self.transform:
                image = self.transform(image)

            out = {
                "image": image,
                "D": torch.tensor(volume.shape[0], dtype=torch.int32),
                "study_id": sample_info["study_id"],
                "series_description": sample_info["series_description"],
                "error_code": error_code,
            }

            return out

        except Exception as e:
            return {
                "image": torch.zeros((1, 1, 1), dtype=torch.float32),
                "D": torch.tensor(0, dtype=torch.int32),
                "study_id": sample_info["study_id"],
                "series_description": sample_info["series_description"],
                "error_code": 1,
            }


batch_size = 1

# Create the test dataset and data loader
test_dataset = TestLumbarSpineDataset(
    image_dir=os.path.join(base_path, "test_images"),
    series=test_series_df,
    label=None,
    coords=None,
    transform=None,
    train_on=args.train_on,
    max_depth=50 if args.max_depth is None else args.max_depth,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn,
)

# Pre-populate results df
logger.info("Preparing results dataframe...")
study_ids = test_series_df["study_id"].unique()

ALL_CONDITIONS = [
    "Spinal Canal Stenosis",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
]

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

results_df = []
for study_id in study_ids:
    for condition in ALL_CONDITIONS:
        for level in LEVELS:
            row_id = f"{study_id}_{condition}_{level}"
            results_df.append({
                "row_id": row_id,
                "normal_mild": 1 / 3,
                "moderate": 1 / 3,
                "severe": 1 / 3,
            })
results_df = pd.DataFrame(results_df)

# %%

logger.info("Starting inference...")
# Inference loop
for batch in test_loader:
    batch = {
        k: v.to(torch_device) if torch.is_tensor(v) else v for k, v in batch.items()
    }
    study_ids_batch = batch.pop("study_id")  # List of study_ids
    series_descriptions_batch = batch.pop(
        "series_description"
    )  # List of series_descriptions
    error_codes = batch.pop("error_code")
    # remove error codes
    mask = [not e for e in error_codes]
    grades = [[4 / 7, 2 / 7, 1 / 7]] * batch_size
    if any(mask):
        D_cumsum = [0] + batch["D"].cumsum().cpu().tolist()
        new_batch = {}
        for i, m in enumerate(mask):
            if not m:
                continue
            left, right = D_cumsum.pop(0), D_cumsum[0]
            new_batch["image"] = batch["image"][left:right]
            new_batch["D"] = batch["D"][i]
        with torch.no_grad():
            outputs = second_stage_model(**new_batch)
            # outputs["grade"] of shape (batch_size, num_points, num_grades)

        out_idx = 0
        out_grades = outputs["grade"].cpu().numpy()
        for i in range(batch_size):
            if not mask[i]:
                continue
            grades[i] = out_grades[out_idx]
            out_idx += 1

    for i in range(batch_size):
        study_id = study_ids_batch[i]
        series_description = series_descriptions_batch[i].lower()
        grade = grades[i]  # Shape: (num_points, num_grades)

        CONDITIONS = {
            "sagittal t2/stir": [
                "spinal_canal_stenosis",
                "spinal_canal_stenosis",
            ],  # repeat twice to match other series
            "axial t2": [
                "left_subarticular_stenosis",
                "right_subarticular_stenosis",
            ],
            "sagittal t1": [
                "left_neural_foraminal_narrowing",
                "right_neural_foraminal_narrowing",
            ],
        }
        LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

        conditions = CONDITIONS.get(series_description, [])
        if not conditions:
            logger.warning(f"Unknown series description: {series_description}")
            continue  # Skip unknown series descriptions

        for c_idx, condition in enumerate(conditions):
            for l_idx, level in enumerate(LEVELS):
                row_id = f"{study_id}_{condition}_{level}"
                # grades are in order of conditions and levels
                point_idx = c_idx * len(LEVELS) + l_idx
                grade_probs = grade[point_idx]  # Shape: (num_grades,)
                # Ensure the row exists in results_df
                if row_id in results_df["row_id"].values:
                    results_df.loc[results_df.row_id == row_id, "normal_mild"] = (
                        grade_probs[0]
                    )
                    results_df.loc[results_df.row_id == row_id, "moderate"] = (
                        grade_probs[1]
                    )
                    results_df.loc[results_df.row_id == row_id, "severe"] = grade_probs[
                        2
                    ]
                else:
                    # If the row doesn't exist, append it
                    results_df = results_df.append(
                        {
                            "row_id": row_id,
                            "normal_mild": grade_probs[0],
                            "moderate": grade_probs[1],
                            "severe": grade_probs[2],
                        },
                        ignore_index=True,
                    )

# Save the results
submission_file = "submission.csv"
results_df.to_csv(submission_file, index=False)
print(results_df)
logger.info(f"Results saved to {submission_file}")
# %%
