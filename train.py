# %%
import argparse
import datetime
import glob
import inspect
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
_default_handler = None


def free_gpu_memory():
    # Delete all tensors that are no longer needed
    for obj in list(locals().values()):
        if torch.is_tensor(obj) and obj.is_cuda:
            del obj

    # Release all unreferenced memory held by PyTorch's caching allocator
    torch.cuda.empty_cache()

    # Forcing PyTorch to free any memory that it can currently release
    torch.cuda.ipc_collect()

    # Optional: Synchronize to make sure all pending operations are done
    torch.cuda.synchronize()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channel + skip_channel,
                out_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UNetDecoder3D(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [
            in_channel,
        ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            DecoderBlock3D(i, s, o) for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


# helper function to encode the input image
def encode(x, e):
    encode = []
    if hasattr(e, "stem"):
        x = e.stem(x)
    elif hasattr(e, "patch_embed"):
        x = e.patch_embed(x)
    for stage in e.stages:
        x = stage(x)
        encode.append(x)
    return encode


class Encoder2DUNet3D(nn.Module):
    def __init__(
        self,
        model_size: str,
        strategy: int,
        dynamic_matching: bool = False,
        train_on=["zxy", "grade"],
    ):
        super().__init__()
        self.output_type = ["infer", "loss"]
        self.register_buffer("D", torch.tensor(0))
        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(0.5))

        decoder_dim = None
        self.train_on = train_on
        self.dynamic_matching = dynamic_matching
        arch = "pvt_v2_b4"
        # strategy 1
        encoder_dim = {
            "pvt_v2_b2": [64, 128, 320, 512],
            "pvt_v2_b4": [64, 128, 320, 512],
        }.get(arch, [768])

        decoder_dim = [384, 192, 96]

        self.encoder = timm.create_model(
            model_name=arch,
            pretrained=False,
            in_chans=3,
            num_classes=0,
            global_pool="",
        )
        self.decoder = UNetDecoder3D(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1],
            out_channel=decoder_dim,
        )
        self.n_stages = 4
        if "zxy" in self.train_on and "grade" in self.train_on:
            self.heatmap = nn.Conv3d(decoder_dim[-1], 30, kernel_size=1)
        elif "zxy" in self.train_on:
            self.zxy_mask = nn.Conv3d(decoder_dim[-1], 10, kernel_size=1)
        elif "grade" in self.train_on:
            self.grade_mask = nn.Conv3d(decoder_dim[-1], 128, kernel_size=1)
            self.grade = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )

    def forward(self, batch):
        device = self.D.device
        image = batch["image"].to(device)
        D = batch["D"].cpu().tolist()
        num_image = len(D)

        B, H, W = image.shape
        image = image.reshape(B, 1, H, W)

        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.expand(-1, 3, -1, -1)

        # ---
        encode = pvtv2_encode(x, self.encoder)
        ##[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
        encode = [torch.split_with_sizes(e, D, 0) for e in encode]

        zxy_mask = []  # prob heatmap
        grade_mask = []
        heatmap = []
        for i in range(num_image):
            e = [encode[s][i].transpose(1, 0).unsqueeze(0) for s in range(4)]
            decoded, _ = self.decoder(feature=e[-1], skip=e[:-1][::-1])
            if "zxy" in self.train_on and "grade" in self.train_on:
                num_point, num_grade = 10, 3
                all = self.heatmap(decoded).squeeze(0)
                _, d, h, w = all.shape
                all = all.reshape(num_point, num_grade, d, h, w)
                all = all.flatten(1).softmax(-1).reshape(num_point, num_grade, d, h, w)
                heatmap.append(all)
            elif "grade" in self.train_on:
                g = self.grade_mask(decoded).squeeze(0)
                grade_mask.append(g)

            elif "zxy" in self.train_on:
                zxy = self.zxy_mask(decoded).squeeze(0)
                _, d, h, w = zxy.shape
                zxy = zxy.flatten(1).softmax(-1).reshape(-1, d, h, w)
                zxy_mask.append(zxy)

        if "zxy" in self.train_on and "grade" in self.train_on:
            xy, z = heatmap_to_coord(heatmap)
            grade = heatmap_to_grade(heatmap)
            heatmap = torch.cat([all.permute(2, 0, 1, 3, 4) for all in heatmap])
        elif "grade" in self.train_on:
            num_point = xy.shape[1]
            grade = masks_to_grade(zxy_mask, grade_mask)
            # print('grade', grade.shape)
            grade = grade.reshape(num_image * num_point, -1)
            grade = self.grade(grade)
            grade = grade.reshape(num_image, num_point, 3)
        elif "zxy" in self.train_on:
            xy, z = zxy_mask_to_coord(zxy_mask)
            # ---
            zxy_mask = torch.cat(zxy_mask, 1).transpose(1, 0)

        output = {}
        if "loss" in self.output_type:
            if "zxy" in self.train_on:
                if "grade" in self.train_on:
                    output["heatmap_loss"] = F_heatmap_loss(
                        heatmap, batch["heatmap"].to(device), D
                    )
                else:
                    output["heatmap_loss"] = F_xyz_mask_loss(
                        zxy_mask, batch["heatmap"].to(device), D
                    )

                mask = batch["z"].to(device) != -1
                output["z_loss"] = F_z_loss(z, batch["z"].to(device), mask)
                output["xy_loss"] = F_xy_loss(xy, batch["xy"].to(device), mask)

            if "grade" in self.train_on:
                index, valid = do_dynamic_match_truth(xy, batch["xy"].to(device))
                truth = batch["grade"].to(device)
                truth_matched = []
                for i in range(num_image):
                    truth_matched.append(truth[i][index[i]])
                truth_matched = torch.stack(truth_matched)
                output["grade_loss"] = F_grade_loss(grade[valid], truth_matched[valid])

        if "infer" in self.output_type:
            if "zxy" in self.train_on:
                output["zxy_mask"] = zxy_mask
                output["xy"] = xy
                output["z"] = z
            if "grade" in self.train_on:
                output["grade"] = F.softmax(grade, -1)

        return output


def heatmap_to_coord(heatmap):
    num_image = len(heatmap)
    device = heatmap[0].device
    num_point, num_grade, D, H, W = heatmap[0].shape
    D = max([h.shape[2] for h in heatmap])

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    z = torch.linspace(0, D - 1, D, device=device)

    point_xy = []
    point_z = []
    for i in range(num_image):
        num_point, num_grade, D, H, W = heatmap[i].shape
        pos_x = x.reshape(1, 1, 1, 1, W)
        pos_y = y.reshape(1, 1, 1, H, 1)
        pos_z = z[:D].reshape(1, 1, D, 1, 1)

        py = torch.sum(pos_y * heatmap[i], dim=(1, 2, 3, 4))
        px = torch.sum(pos_x * heatmap[i], dim=(1, 2, 3, 4))
        pz = torch.sum(pos_z * heatmap[i], dim=(1, 2, 3, 4))

        point_xy.append(torch.stack([px, py]).T)
        point_z.append(pz)

    xy = torch.stack(point_xy)
    z = torch.stack(point_z)
    return xy, z


def heatmap_to_grade(heatmap):
    num_image = len(heatmap)
    grade = []
    for i in range(num_image):
        num_point, num_grade, D, H, W = heatmap[i].shape
        g = torch.sum(heatmap[i], dim=(2, 3, 4))
        grade.append(g)
    grade = torch.stack(grade)
    return grade


def zxy_mask_to_coord(heatmap):
    num_image = len(heatmap)
    device = heatmap[0].device
    _, _, H, W = heatmap[0].shape
    D = max([h.shape[1] for h in heatmap])

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    z = torch.linspace(0, D - 1, D, device=device)

    point_xy = []
    point_z = []
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        pos_x = x.reshape(1, 1, 1, W)
        pos_y = y.reshape(1, 1, H, 1)
        pos_z = z[:D].reshape(1, D, 1, 1)

        py = torch.sum(pos_y * heatmap[i], dim=(1, 2, 3))
        px = torch.sum(pos_x * heatmap[i], dim=(1, 2, 3))
        pz = torch.sum(pos_z * heatmap[i], dim=(1, 2, 3))

        point_xy.append(torch.stack([px, py]).T)
        point_z.append(pz)

    xy = torch.stack(point_xy)
    z = torch.stack(point_z)
    return xy, z


def masks_to_grade(heatmap, grade_mask):
    num_image = len(heatmap)
    grade = []
    for i in range(num_image):
        num_point, D, H, W = heatmap[i].shape
        C, D, H, W = grade_mask[i].shape
        g = grade_mask[i].reshape(1, C, D, H, W)  # .detach()
        h = heatmap[i].reshape(num_point, 1, D, H, W)  # .detach()
        g = (h * g).sum(dim=(2, 3, 4))
        grade.append(g)
    grade = torch.stack(grade)
    return grade


# 3.  loss function
# modeling: loss functions
# magic.2.: example shown here is for sagittal t1 neural foraminal narrowing points
# dynamic matching - becuase of ambiguous/confusion of ground truth labeling of L5 and S1,
# your predicted xy coordinates may be misaligned with ground truth xy. Hence you must
# modified the grade target values as well in loss backpropagation


def do_dynamic_match_truth(xy, truth_xy, threshold=3):
    num_image, num_point, _2_ = xy.shape
    t = truth_xy[:, :5, 1].reshape(num_image, 5, 1)
    p = xy[:, :5, 1].reshape(num_image, 1, 5)
    diff = torch.abs(p - t)
    left, left_i = diff.min(-1)
    left_t = left < threshold

    t = truth_xy[:, 5:, 1].reshape(num_image, 5, 1)
    p = xy[:, 5:, 1].reshape(num_image, 1, 5)
    diff = torch.abs(p - t)
    right, right_i = diff.min(-1)
    right_t = right < threshold

    index = torch.cat([left_i, right_i + 5], 1).detach()
    valid = torch.cat([left_t, right_t], 1).detach()
    return index, valid


def F_grade_loss(grade, truth):
    weight = torch.FloatTensor([1, 2, 4]).to(grade.device)

    t = truth.reshape(-1)
    g = grade.reshape(-1, 3)

    loss = F.cross_entropy(g, t, weight=weight, ignore_index=-1)
    return loss


def F_z_loss(z, z_truth, mask):
    z_truth = z_truth.float()
    loss = F.mse_loss(z[mask], z_truth[mask])
    return loss


def F_xy_loss(xy, xy_truth, mask):
    xy_truth = xy_truth.float()
    loss = F.mse_loss(xy[mask], xy_truth[mask])
    return loss


def F_heatmap_loss(heatmap, truth, D):
    heatmap = torch.split_with_sizes(heatmap, D, 0)
    truth = torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    loss = 0
    for i in range(num_image):
        p, q = truth[i], heatmap[i]
        D, _, _, _, _ = p.shape

        eps = 1e-6
        p = torch.clamp(p.transpose(1, 0).flatten(1), eps, 1 - eps)
        q = torch.clamp(q.transpose(1, 0).flatten(1), eps, 1 - eps)
        m = (0.5 * (p + q)).log()

        def kl(x, t):
            return F.kl_div(x, t, reduction="batchmean", log_target=True)

        loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
    loss = loss / num_image
    return loss


# https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/11
# Jensen-Shannon divergence
def F_xyz_mask_loss(heatmap, truth, D):
    heatmap = torch.split_with_sizes(heatmap, D, 0)
    truth = torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    loss = 0
    for i in range(num_image):
        p, q = truth[i], heatmap[i]
        D, _, _, _, _ = p.shape
        eps = 1e-6
        p = torch.clamp(p.transpose(1, 0).flatten(1), eps, 1 - eps)
        q = torch.clamp(q.transpose(1, 0).flatten(1), eps, 1 - eps)
        m = (0.5 * (p + q)).log()

        def kl(x, t):
            return F.kl_div(x, t, reduction="batchmean", log_target=True)

        loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
        print(loss)
    loss = loss / num_image
    return loss


class LumbarSpineDataset(Dataset):
    def __init__(
        self,
        image_dir,
        series,
        label=None,
        coords=None,
        output_dim=80,
        transform=None,
        max_depth=50,
        train_on=["zxy", "grade"],
    ):
        """
        Args:
            image_dir (str): Directory with all the images.
            label_csv (str): Path to the csv file with labels.
            series_csv (str): Path to the csv file with series descriptions.
            transform (callable, optional): Optional transform to be applied on a sample.
            stage (str): 'train' or 'test' to specify the dataset stage.
            max_depth (int): Maximum number of slices to read to avoid OOM errors.
        """
        self.image_dir = image_dir
        if label is not None:
            if isinstance(label, str):
                self.labels = pd.read_csv(label)
            else:
                self.labels = label
        if coords is not None:
            if isinstance(coords, str):
                self.coords = pd.read_csv(coords)
            else:
                self.coords = coords
        if isinstance(series, str):
            self.series_descriptions = pd.read_csv(series)
        else:
            self.series_descriptions = series
        self.transform = transform
        self.train_on = train_on
        self.max_depth = max_depth
        self.samples = self._prepare_samples()
        self.output_dim = output_dim
        self.valid_indices = set(range(len(self.samples)))

    def _prepare_samples(self):
        samples = []

        # For each study, collect series information
        for _, row in self.series_descriptions.iterrows():
            study_id = row["study_id"]
            series_id = row["series_id"]
            series_description = row["series_description"].lower()

            # Build the image directory
            image_dir = os.path.join(self.image_dir, str(study_id), str(series_id))

            # Collect all DICOM file paths
            dicom_files = glob.glob(f"{image_dir}/*.dcm")
            if not dicom_files:
                continue  # Skip if no DICOM files found

            # Prepare sample entry
            samples.append({
                "study_id": study_id,
                "series_id": series_id,
                "series_description": series_description,
                "image_dir": image_dir,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_new_index(self, idx):
        # fetch the next valid index
        new_index = (idx + 1) % len(self.samples)
        while new_index not in self.valid_indices:
            new_index = (new_index + 1) % len(self.samples)
        return new_index

    def _get_all(self, idx, count=0) -> Union[dict, torch.Tensor]:
        try:
            sample_info = self.samples[idx]

            # Read the series and process the volume
            volume, dicom_df, error_code = self._read_series(
                sample_info["study_id"],
                sample_info["series_id"],
                sample_info["series_description"],
            )

            # Handle any errors in reading the series
            if error_code:
                self.valid_indices.remove(idx)
                if count > 10:
                    raise ValueError(f"The dataset is corrupted: {error_code}")
                return self._get_all(self._get_new_index(idx), count + 1)

            # Prepare the image tensor
            # The volume is in shape (D, H, W), we need to convert it to (C, H, W)
            # For 3D CNNs, you might keep it as (1, D, H, W)
            image = np.ascontiguousarray(volume.transpose(1, 2, 0))  # Shape: (H, W, D)

            # Resize and center the image
            image, scale_param = self.do_resize_and_center(image, reference_size=320)
            image = np.ascontiguousarray(image.transpose(2, 0, 1))  # Shape: (D, H, W)

            # Convert to torch tensor
            image = torch.from_numpy(image).float().half()

            if self.transform:
                image = self.transform(image)

            # Extract labels for the study
            # Prepare a label tensor
            # Assuming you have a method to map labels to indices or one-hot vectors
            xy, z = self._prepare_coords(
                sample_info,
                dicom_df,
                scale_param,
                image.shape,
            )
            grade = None
            heatmap = None
            if "grade" in self.train_on:
                grade = self._prepare_grade(sample_info)
                if "zxy" in self.train_on:
                    heatmap = self.generate_heatmap(
                        z,
                        xy,
                        # coords_mask,
                        grade.long(),
                        # label_mask,
                        image_shape=image.shape,
                    ).half()
            elif "zxy" in self.train_on:
                heatmap = self.generate_zxy_mask(
                    z,
                    xy,
                    image_shape=image.shape,
                ).half()

            out = {
                "image": image,
                "D": torch.tensor(volume.shape[0], dtype=torch.int32),
            }

            if "zxy" in self.train_on:
                out.update({
                    "z": z.half(),
                    "xy": xy.half(),
                    "heatmap": heatmap,
                })

            if "grade" in self.train_on and grade is not None:
                out["grade"] = grade.long()

            return out

        except Exception as e:
            self.valid_indices.remove(idx)
            if count > 10:
                raise e
            return self._get_all(self._get_new_index(idx), count + 1)

    def __getitem__(self, idx) -> Union[dict, torch.Tensor]:
        return self._get_all(idx)

    def _read_series(
        self,
        study_id,
        series_id,
        series_description,
    ):
        error_code = ""

        data_kaggle_dir = self.image_dir
        dicom_dir = f"{data_kaggle_dir}/{study_id}/{series_id}"

        # Read DICOM files
        dicom_files = self.sort_files(glob.glob(f"{dicom_dir}/*.dcm"))
        if not dicom_files:
            return None, None, "[no-files]"

        instance_numbers = [int(Path(f).stem) for f in dicom_files]
        depth = len(instance_numbers)
        if self.max_depth is not None and depth > self.max_depth:
            half_max_depth = self.max_depth // 2
            center = depth // 2
            left_i = max(0, center - half_max_depth)
            right_i = min(center + half_max_depth, depth)
            instance_numbers = instance_numbers[left_i:right_i]
            dicom_files = dicom_files[left_i:right_i]

        dicoms = [pydicom.dcmread(f) for f in dicom_files]

        # Make DICOM header DataFrame
        dicom_df = []
        for i, d in zip(instance_numbers, dicoms):
            dicom_df.append({
                "study_id": study_id,
                "series_id": series_id,
                "series_description": series_description,
                "instance_number": i,
                "ImagePositionPatient": [float(v) for v in d.ImagePositionPatient],
                "ImageOrientationPatient": [
                    float(v) for v in d.ImageOrientationPatient
                ],
                "PixelSpacing": [float(v) for v in d.PixelSpacing],
                "SpacingBetweenSlices": float(getattr(d, "SpacingBetweenSlices", 1.0)),
                "SliceThickness": float(getattr(d, "SliceThickness", 1.0)),
                "grouping": str([
                    round(float(v), 3) for v in d.ImageOrientationPatient
                ]),
                "H": d.pixel_array.shape[0],
                "W": d.pixel_array.shape[1],
            })
        dicom_df = pd.DataFrame(dicom_df)

        # Handle multi-shape images
        if (dicom_df.W.nunique() != 1) or (dicom_df.H.nunique() != 1):
            error_code = "[multi-shape]"
        Wmax = dicom_df.W.max()
        Hmax = dicom_df.H.max()

        # Sort slices
        groups = dicom_df.groupby("grouping")
        data = []
        sort_data_by_group = []

        for _, df_group in groups:
            position = np.array(df_group["ImagePositionPatient"].values.tolist())
            orientation = np.array(df_group["ImageOrientationPatient"].values.tolist())
            normal = np.cross(orientation[:, :3], orientation[:, 3:])
            projection = self.np_dot(normal, position)
            df_group.loc[:, "projection"] = projection
            df_group = df_group.sort_values("projection")

            # Ensure slices are continuous
            if len(df_group.SliceThickness.unique()) != 1:
                error_code += "[slice-thickness-variation]"

            volume = []
            for i in df_group.instance_number:
                idx = instance_numbers.index(i)
                v = dicoms[idx].pixel_array
                if error_code.find("multi-shape") != -1:
                    H, W = v.shape
                    v = np.pad(v, [(0, Hmax - H), (0, Wmax - W)], "reflect")
                volume.append(v)

            volume = np.stack(volume)
            volume = self.normalise_to_8bit(volume)

            data.append({
                "df": df_group,
                "volume": volume,
            })

            # Sort data by group
            if "sagittal" in series_description.lower():
                sort_data_by_group.append(position[0, 0])  # x
            if "axial" in series_description.lower():
                sort_data_by_group.append(position[0, 2])  # z

        # Sort the data
        data = [r for _, r in sorted(zip(sort_data_by_group, data), key=lambda x: x[0])]
        for i, r in enumerate(data):
            r["df"].loc[:, "group"] = i

        df_combined = pd.concat([r["df"] for r in data])
        df_combined.loc[:, "z"] = np.arange(len(df_combined))
        volume_combined = np.concatenate([r["volume"] for r in data])

        return volume_combined, df_combined, error_code

    def np_dot(self, a, b):
        return np.sum(a * b, 1)

    def normalise_to_8bit(self, x, lower=0.1, upper=99.9):
        lower, upper = np.percentile(x, (lower, upper))
        x = np.clip(x, lower, upper)
        x = x - np.min(x)
        x = x / np.max(x)
        return (x * 255).astype(np.uint8)

    def sort_files(self, files):
        file_ids = np.array([int(Path(fp).stem) for fp in files])
        sorted_indices = np.argsort(file_ids)
        files = [files[i] for i in sorted_indices]
        return files

    def do_resize_and_center(self, image, reference_size):
        H, W = image.shape[:2]
        if (W == reference_size) and (H == reference_size):
            return image, (1, 0, 0)

        s = reference_size / max(H, W)
        m = cv2.resize(image, dsize=None, fx=s, fy=s)
        h, w = m.shape[:2]
        padx0 = (reference_size - w) // 2
        padx1 = reference_size - w - padx0
        pady0 = (reference_size - h) // 2
        pady1 = reference_size - h - pady0

        m = np.pad(
            m,
            [[pady0, pady1], [padx0, padx1], [0, 0]],
            mode="constant",
            constant_values=0,
        )
        scale_param = (s, padx0, pady0)
        return m, scale_param

    def _prepare_grade(self, sample_info):
        labels_df = self.labels[self.labels["study_id"] == sample_info["study_id"]]
        series_description = sample_info["series_description"]
        label_tensor = torch.zeros(10)
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
        # mask_list = []
        for i, condition in enumerate(CONDITIONS[series_description]):
            for j, level in enumerate(LEVELS):
                label_col = f"{condition}_{level}"
                if label_col in labels_df.columns:
                    label_value = labels_df[label_col].values[0]
                    if label_value in ["Normal", "Mild", "Normal/Mild"]:
                        label_idx = 0
                    elif label_value == "Moderate":
                        label_idx = 1
                    elif label_value == "Severe":
                        label_idx = 2
                    else:
                        label_tensor[i * len(LEVELS) + j] = -1
                        # mask_list.append([0, 0, 0])
                        continue  # Skip if label is missing or invalid
                    label_tensor[i * len(LEVELS) + j] = label_idx
                    # mask_list.append([1, 1, 1])
                else:
                    # mask_list.append([0, 0, 0])
                    pass
        # mask_tensor = torch.tensor(mask_list, dtype=torch.int64)
        return label_tensor.long()  # , mask_tensor

    def _prepare_coords(self, sample_info, df_dicom, scale_param, image_shape):
        """
        Prepare the 3D coordinates.

        Args:
            sample_info (dict): Information about the current sample.
            df_dicom (DataFrame): DataFrame containing DICOM metadata for the current volume.

        Returns:
            coords_tensor (torch.Tensor): Tensor of shape (num_points, 3) containing (x, y, z) coordinates.
        """

        CONDITIONS = {
            "sagittal t2/stir": [
                "Spinal Canal Stenosis",
                "Spinal Canal Stenosis",
            ],  # repeat twice to match other series
            "axial t2": ["Left Subarticular Stenosis", "Right Subarticular Stenosis"],
            "sagittal t1": [
                "Left Neural Foraminal Narrowing",
                "Right Neural Foraminal Narrowing",
            ],
        }

        LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

        s1, padx0, pady0 = scale_param
        s2 = self.output_dim / max(image_shape[1], image_shape[2])

        # Filter coords and labels for this study
        study_id = sample_info["study_id"]
        series_description = sample_info["series_description"]
        coords_df = self.coords[self.coords["study_id"] == study_id]

        # Mapping from condition and level to index

        conditions = CONDITIONS.get(series_description, [])

        z_list = []
        xy_list = []
        # mask_list = []
        for condition in conditions:
            for level in LEVELS:
                # Get coordinates for this condition and level
                condition_coords = coords_df[
                    (coords_df["condition"] == condition)
                    & (coords_df["level"] == level)
                ]
                if condition_coords.empty:
                    xy_list.append([0.0, 0.0])
                    z_list.append(-1.0)
                    # mask_list.append([0.0])
                    continue

                # Get the instance number (slice index)
                instance_number = condition_coords["instance_number"].values[0]
                slice_index = df_dicom[df_dicom["instance_number"] == instance_number]

                if slice_index.empty:
                    xy_list.append([0.0, 0.0])
                    z_list.append(-1.0)
                    # mask_list.append([0.0])
                    continue
                else:
                    slice_index = slice_index.index[0]

                # Original coordinates
                x = condition_coords["x"].values[0]
                y = condition_coords["y"].values[0]
                z = slice_index

                # Adjust coordinates based on scaling and padding
                x = (x * s1 + padx0) * s2
                y = (y * s1 + pady0) * s2
                z = z  # Assuming no scaling in depth dimension

                xy_list.append([x, y])
                z_list.append(z)
                # mask_list.append([1.0])

        if xy_list:
            xy_tensor = torch.tensor(
                xy_list, dtype=torch.float32
            )  # Shape: (num_points, 2)
            z_tensor = torch.tensor(z_list, dtype=torch.float32)
            # mask_tensor = torch.tensor(mask_list, dtype=torch.float32)
        else:
            xy_tensor = torch.zeros((0, 2))
            z_tensor = torch.zeros((0, 1))
            # mask_tensor = torch.zeros((0, 1), dtype=torch.bool)

        return xy_tensor, z_tensor  # , mask_tensor

    def generate_heatmap(
        self,
        z_tensor,
        xy_tensor,
        # coords_mask,
        grades_tensor,
        # label_mask,
        image_shape,
    ):
        """
        Generate a heatmap tensor of shape [D, num_point, num_grade, H, W].

        Args:
            coords_tensor (Tensor): Tensor of shape [num_points, 3] containing (x, y, z) coordinates.
            grades_tensor (Tensor): Tensor of shape [num_points] containing grade indices.
            volume_shape (tuple): Shape of the volume as (D, H, W).

        Returns:
            heatmap (Tensor): Heatmap tensor.
        """
        num_point = 10
        num_grade = 3  # Number of severity grades
        D, W, H = image_shape
        scale = self.output_dim / max(H, W)
        H, W = int(H * scale), int(W * scale)

        # Initialize heatmap with zeros
        heatmap = torch.zeros((D, num_point, num_grade, H, W), dtype=torch.float32)

        sigma = 1
        tmp_size = sigma * 3

        # mask_list = []
        for idx in range(num_point):
            # if (coords_mask[idx].sum() == 0) or (label_mask[idx].sum() == 0):
            #     mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))

            x, y = xy_tensor[idx][0], xy_tensor[idx][1]
            z = z_tensor[idx]
            grade = grades_tensor[idx]

            if z == -1:
                # mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))
                continue

            if grade == -1:
                # mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))
                continue  # Skip if grade is missing

            x = x.round().int()
            y = y.round().int()
            z = z.round().int()

            # Ensure coordinates are within bounds
            if not (0 <= x < W and 0 <= y < H and 0 <= z < D):
                # mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))
                continue

            # Create meshgrid for the Gaussian
            x_min = max(0, x - tmp_size)
            x_max = min(W, x + tmp_size + 1)
            y_min = max(0, y - tmp_size)
            y_max = min(H, y + tmp_size + 1)
            z_min = max(0, z - tmp_size)
            z_max = min(D, z + tmp_size + 1)

            x_min, x_max, y_min, y_max, z_min, z_max = (
                torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])
                .round()
                .int()
                .tolist()
            )

            grid_x = torch.arange(x_min, x_max, dtype=torch.int32)
            grid_y = torch.arange(y_min, y_max, dtype=torch.int32)
            grid_z = torch.arange(z_min, z_max, dtype=torch.int32)

            yy, xx, zz = torch.meshgrid(grid_y, grid_x, grid_z, indexing="ij")

            # Compute the Gaussian
            gaussian = torch.exp(
                -((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * sigma**2)
            )

            # Add the Gaussian to the heatmap
            heatmap[z_min:z_max, idx, grade, y_min:y_max, x_min:x_max] = torch.maximum(
                heatmap[z_min:z_max, idx, grade, y_min:y_max, x_min:x_max],
                gaussian.permute(2, 0, 1),
            )
            # mask_list.append(torch.ones((D, num_grade, H, W), dtype=torch.float32))

        # Normalize the heatmap
        heatmap = heatmap / (heatmap.sum(dim=(3, 4), keepdim=True) + 1e-8)
        # mask_tensor = torch.stack(mask_list).permute(
        #     1, 0, 2, 3, 4
        # )  # Shape: (num_point, D, num_grade, H, W) -> (D, num_point, num_grade, H, W)

        return heatmap  # , mask_tensor

    def generate_zxy_mask(
        self,
        z_tensor,
        xy_tensor,
        image_shape,
    ):
        """
        Generate a heatmap tensor of shape [D, num_point, num_grade, H, W].

        Args:
            coords_tensor (Tensor): Tensor of shape [num_points, 3] containing (x, y, z) coordinates.
            grades_tensor (Tensor): Tensor of shape [num_points] containing grade indices.
            volume_shape (tuple): Shape of the volume as (D, H, W).

        Returns:
            heatmap (Tensor): Heatmap tensor.
        """
        num_point = 10
        D, W, H = image_shape
        scale = self.output_dim / max(H, W)
        H, W = int(H * scale), int(W * scale)

        heatmap = torch.zeros((D, num_point, H, W), dtype=torch.float32)

        sigma = 1
        tmp_size = sigma * 3

        # mask_list = []
        for idx in range(num_point):
            # if (coords_mask[idx].sum() == 0) or (label_mask[idx].sum() == 0):
            #     mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))

            x, y = xy_tensor[idx][0], xy_tensor[idx][1]
            z = z_tensor[idx]

            if z == -1:
                # mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))
                continue

            x = x.round().int()
            y = y.round().int()
            z = z.round().int()

            if not (0 <= x < W and 0 <= y < H and 0 <= z < D):
                # mask_list.append(torch.zeros((D, num_grade, H, W), dtype=torch.float32))
                continue

            # create the meshgrid for the Gaussian
            x_min = max(0, x - tmp_size)
            x_max = min(W, x + tmp_size + 1)
            y_min = max(0, y - tmp_size)
            y_max = min(H, y + tmp_size + 1)
            z_min = max(0, z - tmp_size)
            z_max = min(D, z + tmp_size + 1)

            x_min, x_max, y_min, y_max, z_min, z_max = (
                torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])
                .round()
                .int()
                .tolist()
            )

            grid_x = torch.arange(x_min, x_max, dtype=torch.int32)
            grid_y = torch.arange(y_min, y_max, dtype=torch.int32)
            grid_z = torch.arange(z_min, z_max, dtype=torch.int32)

            yy, xx, zz = torch.meshgrid(grid_y, grid_x, grid_z, indexing="ij")

            gaussian = torch.exp(
                -((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * sigma**2)
            )

            heatmap[z_min:z_max, idx, y_min:y_max, x_min:x_max] = torch.maximum(
                heatmap[z_min:z_max, idx, y_min:y_max, x_min:x_max],
                gaussian.permute(2, 0, 1),
            )
            # mask_list.append(torch.ones((D, num_grade```````, H, W), dtype=torch.float32))

        heatmap = heatmap / (heatmap.sum(dim=(2, 3), keepdim=True) + 1e-8)
        return heatmap  # , mask_tensor


def custom_collate_fn(batch):
    """
    Custom collate function to handle dictionaries in batch.
    """
    new_batch = {}
    # filter out batch items that are None
    batch = [sample for sample in batch if sample is not None]
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    for key in batch[0].keys():
        if key == "image":
            # concat intead of stack
            new_batch["image"] = torch.cat([sample["image"] for sample in batch])
        elif key == "heatmap":
            new_batch["heatmap"] = torch.cat([sample["heatmap"] for sample in batch])
        else:
            new_batch[key] = torch.stack([sample[key] for sample in batch])
    return new_batch


def split_study_ids(
    series_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    # Get unique study IDs
    study_ids = series_df["study_id"].unique()

    # Split study IDs into training and validation sets
    train_study_ids, val_study_ids = train_test_split(
        study_ids, test_size=test_size, random_state=random_state
    )

    return train_study_ids, val_study_ids


def filter_by_study_ids(df: pd.DataFrame, study_ids: List[str]):
    return df[df["study_id"].isin(study_ids)]


def save_combined_heatmap_as_png(
    heatmap: torch.Tensor,
    xy_coords: torch.Tensor,
    z_coords: torch.Tensor,
    image: torch.Tensor,
    point=None,
):
    """
    Combine heatmaps for each point based on their corresponding grades and save as PNG images for selected depths.

    Args:
        heatmap (Tensor): Heatmap tensor of shape [D, num_point, H, W].
        selected_depths (List[int], optional): List of depth indices to visualize. Defaults to all depths.

    Raises:
        ValueError: If the dimensions of heatmap and grade do not match.
        ValueError: If any selected depth index is out of bounds.
    """
    # Ensure the heatmap tensor is on CPU
    if heatmap.is_cuda:
        heatmap = heatmap.cpu()
    if not isinstance(xy_coords, torch.Tensor):
        xy_coords = torch.tensor(xy_coords)
    else:
        xy_coords = xy_coords.clone()

    if not isinstance(z_coords, torch.Tensor):
        z_coords = torch.tensor(z_coords)
    else:
        z_coords = z_coords.clone()

    if xy_coords.is_cuda:
        xy_coords = xy_coords.cpu()

    if z_coords.is_cuda:
        z_coords = z_coords.cpu()

    if image.is_cuda:
        image = image.cpu()

    D, num_point, H, W = heatmap.shape

    if image is not None:
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"background_image must be a torch.Tensor, got {type(image)}"
            )
        if image.ndim != 3:
            raise ValueError(
                f"background_image must have 3 dimensions [C, H, W], got {image.ndim} dimensions"
            )
        C_bg, H_bg, W_bg = image.shape
        if C_bg not in [1, 3, 4]:
            raise ValueError(
                f"background_image must have 1, 3, or 4 channels, got {C_bg}"
            )
        # Resize background image to match heatmap size if necessary
        if H_bg != H or W_bg != W:
            # Use torch to resize (assuming background_image is float tensor)
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)
        # Convert to NumPy array and transpose to [H, W, C]
        background_image_np = image.permute(1, 2, 0).numpy()
        # If grayscale, convert to RGB by repeating channels
        if C_bg == 1:
            background_image_np = np.repeat(background_image_np, 3, axis=2)
        elif C_bg == 4:
            # If RGBA, discard the alpha channel
            background_image_np = background_image_np[:, :, :3]
        # Normalize if necessary (assuming background_image is in [0, 1] or [0, 255]
        if background_image_np.max() > 1.0:
            background_image_np = background_image_np / 255.0
    else:
        background_image_np = None  # Use black background

    if (
        xy_coords.ndim != 2
        or xy_coords.shape[0] != num_point
        or xy_coords.shape[1] != 2
    ):
        raise ValueError(
            f"xy_coords must be 2D with shape [num_point, 2], got shape {xy_coords.shape}"
        )

    if point is None:
        point = range(num_point)
    # Iterate through the selected depths and save each combined heatmap
    for d in z_coords.unique():
        # Initialize a cumulative heatmap for depth d
        cumulative_heatmap = torch.zeros((H, W), dtype=torch.float32)
        for p in point:
            cumulative_heatmap += heatmap[d.int().item(), p, :, :]

        # Normalize the cumulative heatmap to [0, 1]
        heatmap_min = cumulative_heatmap.min()
        heatmap_max = cumulative_heatmap.max()
        if heatmap_max > heatmap_min:
            normalized_heatmap = (cumulative_heatmap - heatmap_min) / (
                heatmap_max - heatmap_min
            )
        else:
            normalized_heatmap = torch.zeros_like(cumulative_heatmap)

        heatmap_np = normalized_heatmap.numpy()

        # Plotting
        plt.figure(figsize=(6, 6))

        if background_image_np is not None:
            # Display background image
            plt.imshow(background_image_np, extent=(0, W, H, 0))
        else:
            # Use black background
            plt.imshow(np.zeros((H, W, 3)), extent=(0, W, H, 0))

        # Overlay the heatmap with transparency
        plt.imshow(heatmap_np, cmap="hot", alpha=0.5, extent=(0, W, H, 0))
        plt.colorbar(label="Normalized Intensity")
        plt.title(f"Combined Heatmap for Depth {d}")
        plt.axis("off")  # Hide axes

        # Overlay labeled circles
        for p in point:
            x, y = xy_coords[p].tolist()
            # Validate coordinates are within image bounds
            if not (0 <= x < W and 0 <= y < H):
                print(
                    f"Warning: Point {p} with coordinates ({x}, {y}) is out of bounds for image size ({W}, {H}). Skipping label."
                )
                continue

            # Plot a circle
            plt.scatter(x, y, s=100, facecolors="none", edgecolors="cyan", linewidth=2)

            # Add a label (e.g., point index)
            plt.text(x, y, str(p), color="cyan", fontsize=12, ha="center", va="center")

        # Save the image
        plt.show()


# encoder helper
def pvtv2_encode(x, e):
    encode = []
    x = e.patch_embed(x)
    for stage in e.stages:
        x = stage(x)
        encode.append(x)
    return encode


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


# %%

logger = logging.getLogger()


def logger_setup():
    global _default_handler
    if _default_handler is not None:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Log to console
    _default_handler = logging.StreamHandler()
    _default_handler.setLevel(logging.INFO)
    _default_handler.setFormatter(formatter)
    logger.addHandler(_default_handler)

    # Log to file
    file_handler = logging.FileHandler("train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


logger_setup()

# %%


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


args = dotdict(all=False, max_depth=30, train_on=["zxy", "grade"])

today_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = f"checkpoints_{today_str}"
os.makedirs(checkpoint_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.device(device)
logger.info(f"Using device: {device}")

base_path = "data/rsna-2024-lumbar-spine-degenerative-classification"

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
model = Encoder2DUNet3D(
    model_size="medium",
    strategy=1,
    train_on=args.train_on,
)
state_dict = torch.load(
    "data/00034142.pth",
    map_location=lambda storage, loc: storage,
    weights_only=True,
)["state_dict"]

print(model.load_state_dict(state_dict, strict=False))  # True
model = model.to(torch_device)
# zxy_detector = None
# if "zxy" not in args.train_on:
#     zxy_detector = model
#     model = None

# %%
DEBUG = False
log_frequency = 1
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 10

# Calculate the total number of steps for OneCycleLR considering gradient accumulation
steps_per_epoch = len(train_loader) // accumulation_steps
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
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
            # "heatmap_mask": heatmap_mask,
            "z": z_target,
            "xy": xy_target,
            # "coords_mask": coords_mask,
            "grade": grade_target,
            # "label_mask": label_mask,
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
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Step optimizer and scaler
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients
                optimizer.zero_grad()

                # Step scheduler
                scheduler.step()

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
        # Validation loop
        model.eval()
        val_outputs = {}
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_loader):
                    image = val_batch["image"].to(torch_device)
                    D = val_batch["D"].to(torch_device)
                    heatmap_target = val_batch.get("heatmap", None)
                    # heatmap_mask = val_batch.get("heatmap_mask", None)
                    z_target = val_batch.get("z", None)
                    xy_target = val_batch.get("xy", None)
                    # coords_mask = val_batch.get("coords_mask", None)
                    grade_target = val_batch["grade"].to(torch_device)
                    # label_mask = val_batch["label_mask"].to(device)

                    val_batch_input = {
                        "image": image,
                        "D": D,
                        "heatmap": heatmap_target,
                        # "heatmap_mask": heatmap_mask,
                        "z": z_target,
                        "xy": xy_target,
                        # "coords_mask": coords_mask,
                        "grade": grade_target,
                        # "label_mask": label_mask,
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
                            if key in losses:
                                msg += f"Validation {key.capitalize()}: {losses[key].avg:.4f}, "

                        logger.info(msg)
                        if DEBUG:
                            break

        avg_val_loss = losses["val_loss"].avg
        msg = f"Validation Loss: {avg_val_loss:.4f}, "
        for key in val_outputs:
            if key in losses:
                msg += f"Validation {key.capitalize()}: {losses[key].avg:.4f}, "
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
