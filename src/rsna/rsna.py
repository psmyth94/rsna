import glob
import logging
import os
from pathlib import Path
from typing import List, Optional, Set, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import timm
import timm_3d
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from spacecutter.callbacks import AscensionCallback, LogisticCumulativeLink
from torch.utils.data import Dataset

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


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


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


class FirstStageModel(nn.Module):
    def __init__(
        self,
        dynamic_matching: bool = False,
        pretrained: bool = True,
        train_on=["zxy", "grade"],
        num_points=5,
        num_grades=3,
        series: str = "sagittal t1",
    ):
        if series not in ["sagittal t1", "sagittal t2/stir", "axial t2"]:
            raise ValueError("Invalid series")
        super().__init__()
        self.output_type = ["infer", "loss"]
        self.register_buffer("D", torch.tensor(0))
        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(0.5))

        decoder_dim = None
        self.train_on = train_on
        self.dynamic_matching = dynamic_matching
        arch = "pvt_v2_b4"
        encoder_dim = {
            "pvt_v2_b2": [64, 128, 320, 512],
            "pvt_v2_b4": [64, 128, 320, 512],
        }.get(arch, [768])

        decoder_dim = [384, 192, 96]

        self.encoder = timm.create_model(
            model_name=arch,
            pretrained=pretrained,
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
        self.num_points = num_points
        self.num_grades = num_grades
        if "zxy" in self.train_on and "grade" in self.train_on:
            self.heatmap = nn.Conv3d(
                decoder_dim[-1], num_points * num_grades, kernel_size=1
            )
        elif "zxy" in self.train_on:
            self.zxy_mask = nn.Conv3d(decoder_dim[-1], num_points, kernel_size=1)
        elif "grade" in self.train_on:
            self.grade_mask = nn.Conv3d(decoder_dim[-1], 128, kernel_size=1)
            self.grade = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )
        self.xy_max = 80

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
                all = self.heatmap(decoded).squeeze(0)
                _, d, h, w = all.shape
                all = all.reshape(self.num_points, self.num_grades, d, h, w)
                all = (
                    all.flatten(1)
                    .softmax(-1)
                    .reshape(self.num_points, self.num_grades, d, h, w)
                )
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
            zxy_mask = batch["zxy_mask"].to(device)
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
                output["z_loss"] = F_z_loss(
                    z, batch["z"].to(device), mask, batch["D"].to(device)
                )
                output["xy_loss"] = F_xy_loss(
                    xy, batch["xy"].to(device), mask, self.xy_max
                )

            if "grade" in self.train_on:
                if self.dynamic_matching:
                    index, valid = do_dynamic_match_truth10(xy, batch["xy"].to(device))
                    truth = batch["grade"].to(device)
                    truth_matched = []
                    for i in range(num_image):
                        truth_matched.append(truth[i][index[i]])
                    truth_matched = torch.stack(truth_matched)
                    if torch.all(truth_matched[valid] == -1):
                        F_grade_loss(grade, truth_matched)
                    else:
                        output["grade_loss"] = F_grade_loss(
                            grade[valid], truth_matched[valid]
                        )
                else:
                    output["grade_loss"] = F_grade_loss(
                        grade, batch["grade"].to(device)
                    )

        if "infer" in self.output_type:
            if "zxy" in self.train_on:
                output["zxy_mask"] = zxy_mask
                output["xy"] = xy
                output["z"] = z
            if "grade" in self.train_on:
                output["grade"] = F.softmax(grade, -1)
            output["image"] = x

        return output


class SecondStageModel(nn.Module):
    def __init__(
        self,
        zxy_predictor,
        crop_size=32,
        num_grades=3,
        xy_max=80,
        output_type=["loss", "infer"],
        pretrained=True,
    ):
        super(SecondStageModel, self).__init__()
        self.crop_size = crop_size
        self.num_grades = num_grades
        self.xy_max = xy_max
        self.zxy_predictor = zxy_predictor
        self.zxy_predictor.output_type = ["infer"]
        self.output_type = output_type

        # Freeze the weights of the first stage model
        for param in self.zxy_predictor.parameters():
            param.requires_grad = False

        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features

        # Add a classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_grades),
        )

    def forward(self, batch):
        output = self.zxy_predictor(batch)
        z_pred = output["z"]
        xy_pred = output["xy"]
        device = z_pred.device
        images = output["image"].to(device)
        D_list = batch["D"].cpu().tolist()

        images = images[:, 0, :, :]
        num_images = len(D_list)
        D_cumsum = [0] + np.cumsum(D_list).tolist()
        crops = []
        for i in range(num_images):
            image_start = int(D_cumsum[i])
            image_end = int(D_cumsum[i + 1])
            image = images[image_start:image_end]  # Shape: (D_i, H, W)
            z_i = z_pred[i].round().long()  # Shape: (num_points,)
            xy_i = xy_pred[i]  # Shape: (num_points, 2)
            num_points = z_i.shape[0]
            for p in range(num_points):
                z_p = z_i[p]
                x_p, y_p = xy_i[p]
                crop = self.extract_crop(image, z_p, x_p, y_p)
                crops.append(crop)
        # Stack crops
        crops = torch.stack(
            crops
        )  # Shape: (batch_size * num_points, 3, H_crop, W_crop)
        # Pass through the backbone
        features = self.backbone(crops)
        # Pass through the classifier
        logits = self.classifier(features)
        # Reshape to (num_images, num_points, num_grades)
        logits = logits.view(num_images, num_points, -1)
        output = {}
        if "loss" in self.output_type:
            output["grade_loss"] = F_grade_loss(logits, batch["grade"].to(device))
        if "infer" in self.output_type:
            output["grade"] = F.softmax(logits, -1)
        return output

    def extract_crop(self, image, z, x, y):
        # image: tensor of shape (D, H, W)
        # z: scalar
        # x, y: scalars
        D, H, W = image.shape
        z = int(round(z.item() - 1))
        x = int(round(W * (x.item() / self.xy_max)))
        y = int(round(H * (y.item() / self.xy_max)))
        # Ensure coordinates are within bounds
        z = max(0, min(z, D - 1))
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        # Get the slice at depth z
        slice_img = image[z]  # Shape: (H, W)
        slice_img = slice_img.unsqueeze(0).repeat(3, 1, 1)
        # Define the crop boundaries
        half_size = self.crop_size // 2
        x_min = max(0, x - half_size)
        x_max = min(W, x + half_size)
        y_min = max(0, y - half_size)
        y_max = min(H, y + half_size)
        # Crop the image
        crop = slice_img[:, y_min:y_max, x_min:x_max]
        # Pad the crop if necessary to get the desired size
        pad_left = max(0, half_size - x)
        pad_right = max(0, (x + half_size) - W)
        pad_top = max(0, half_size - y)
        pad_bottom = max(0, (y + half_size) - H)
        crop = F.pad(
            crop, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )
        # Resize crop to the expected input size of the backbone
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return crop  # Shape: (3, H_crop, W_crop)


class SecondStageModelV2(nn.Module):
    def __init__(
        self,
        zxy_predictor,
        crop_size=80,
        depth_size=6,
        num_grades=3,
        xy_max=80,
        output_type=["loss", "infer"],
        backbone="efficientnet_b0",
        pretrained=True,
        in_chans=3,
        cutpoint_margin=0.15,
    ):
        super(SecondStageModelV2, self).__init__()
        self.crop_size = crop_size
        self.depth_size = depth_size
        self.num_grades = num_grades
        self.xy_max = xy_max
        self.zxy_predictor = zxy_predictor
        self.zxy_predictor.output_type = ["infer"]
        self.output_type = output_type
        self.in_chans = in_chans
        # Freeze the weights of the first stage model
        for param in self.zxy_predictor.parameters():
            param.requires_grad = False

        self.backbone = timm_3d.create_model(
            backbone,
            features_only=False,
            drop_rate=0,
            drop_path_rate=0,
            pretrained=pretrained,
            in_chans=in_chans,
            global_pool="max",
        )

        if "efficientnet" in backbone:
            head_in_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.LayerNorm(head_in_dim),
                nn.Dropout(0),
            )

        elif "vit" in backbone:
            self.backbone.head.drop = nn.Dropout(0)
            head_in_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        self.head = nn.Sequential(nn.Linear(head_in_dim, 1), LogisticCumulativeLink(3))
        self.ascension_callback = AscensionCallback(margin=cutpoint_margin)

    def forward(self, batch):
        output = self.zxy_predictor(batch)
        z_pred = output["z"]
        xy_pred = output["xy"]
        device = z_pred.device
        images = batch["image"].to(device)
        D_list = batch["D"].cpu().tolist()

        num_images = len(D_list)
        D_cumsum = [0] + np.cumsum(D_list).tolist()
        crops = []
        for i in range(num_images):
            image_start = int(D_cumsum[i])
            image_end = int(D_cumsum[i + 1])
            image = images[image_start:image_end]  # Shape: (D_i, H, W)
            z_i = z_pred[i].round().long()  # Shape: (num_points,)
            xy_i = xy_pred[i]  # Shape: (num_points, 2)
            num_points = z_i.shape[0]
            for p in range(num_points):
                z_p = z_i[p]
                x_p, y_p = xy_i[p]
                crop = self.extract_crop_3d(image, z_p, x_p, y_p)
                crops.append(crop)
        # Stack crops
        crops = torch.stack(
            crops
        )  # Shape: (batch_size * num_points, in_chans, D_crop, H_crop, W_crop)
        # Pass through the backbone
        features = self.backbone(crops)
        # Pass through the classifier
        logits = self.head(features)
        # Reshape to (num_images, num_points, num_grades)
        logits = logits.view(num_images, num_points, -1)
        grade_probs = F.softmax(logits, -1)
        output = {}
        if "loss" in self.output_type:
            output["grade_loss"] = F_grade_loss(grade_probs, batch["grade"].to(device))
        if "infer" in self.output_type:
            output["grade"] = F.softmax(logits, -1)
        return output

    def extract_crop_3d(self, image, z, x, y):
        # image: tensor of shape (D, H, W)
        # z, x, y: scalars
        D, H, W = image.shape
        z = int(round(z.item()))
        x = int(round(W * (x.item() / self.xy_max)))
        y = int(round(H * (y.item() / self.xy_max)))
        # Ensure coordinates are within bounds
        z = max(0, min(z, D - 1))
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        # Define the crop boundaries
        half_size = self.crop_size // 2
        half_depth_size = self.depth_size // 2
        z_min = max(0, z - half_depth_size)

        z_max = min(D, z + half_depth_size)
        y_min = max(0, y - half_size)
        y_max = min(H, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(W, x + half_size)
        # Crop the volume
        crop = image[
            z_min:z_max, y_min:y_max, x_min:x_max
        ]  # Shape: (D_crop, H_crop, W_crop)
        # Pad the crop if necessary to get the desired size
        pad_z_left = max(0, half_depth_size - z)
        pad_z_right = max(0, (z + half_depth_size) - D)
        pad_y_left = max(0, half_size - y)
        pad_y_right = max(0, (y + half_size) - H)
        pad_x_left = max(0, half_size - x)
        pad_x_right = max(0, (x + half_size) - W)
        # Note: pad order is (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right)

        pad = (
            pad_x_left,
            pad_x_right,
            pad_y_left,
            pad_y_right,
            pad_z_left,
            pad_z_right,
        )
        crop = F.pad(crop, pad, mode="constant", value=0)
        # Now crop shape should be (D_crop_padded, H_crop_padded, W_crop_padded)
        # Add channel dimension
        if self.in_chans == 1:
            crop = crop.unsqueeze(0)
        else:
            crop = crop.unsqueeze(0).repeat(self.in_chans, 1, 1, 1)
        # Resize crop to expected input size if necessary
        crop = F.interpolate(
            crop.unsqueeze(
                0
            ),  # Shape: (1, in_chans, D_crop_padded, H_crop_padded, W_crop_padded)
            size=(self.depth_size, self.crop_size, self.crop_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # Shape: (in_chans, D_crop_padded, H_crop_padded, W_crop_padded)
        return crop

    def _ascension_callback(self):
        self.ascension_callback.clip(self.head[-1])


class SecondStageModelV3(nn.Module):
    def __init__(
        self,
        zxy_predictor,
        crop_size=80,
        depth_size=6,
        num_grades=3,
        xy_max=80,
        output_type=["loss", "infer"],
        pretrained=True,
        in_chans=1,
    ):
        super(SecondStageModelV3, self).__init__()
        self.crop_size = crop_size
        self.depth_size = depth_size
        self.num_grades = num_grades
        self.xy_max = xy_max
        self.zxy_predictor = zxy_predictor
        self.zxy_predictor.output_type = ["infer"]
        self.output_type = output_type
        self.in_chans = in_chans
        self.encoder_decoder = FirstStageModel(
            dynamic_matching=True,
            pretrained=pretrained,
            train_on=["grade"],
        )
        # Freeze the weights of the first stage model
        for param in self.zxy_predictor.parameters():
            param.requires_grad = False

    def forward(self, batch):
        output = self.zxy_predictor(batch)
        z_pred = output["z"]
        xy_pred = output["xy"]
        device = z_pred.device
        images = batch["image"].to(device)
        D_list = batch["D"].cpu().tolist()

        num_images = len(D_list)
        D_cumsum = [0] + np.cumsum(D_list).tolist()
        crops = []
        for i in range(num_images):
            image_start = int(D_cumsum[i])
            image_end = int(D_cumsum[i + 1])
            image = images[image_start:image_end]  # Shape: (D_i, H, W)
            z_i = z_pred[i].round().long()  # Shape: (num_points,)
            xy_i = xy_pred[i]  # Shape: (num_points, 2)
            num_points = z_i.shape[0]
            for p in range(num_points):
                z_p = z_i[p]
                x_p, y_p = xy_i[p]
                crop = self.extract_crop_3d(image, z_p, x_p, y_p)
                crops.append(crop)
        # Stack crops
        crops = torch.stack(
            crops
        )  # Shape: (batch_size * num_points, in_chans, D_crop, H_crop, W_crop)
        # Pass through the backbone
        crops = crops[:, 0, :, :]
        new_D = torch.tensor([self.depth_size] * num_images).to(device)
        crops = crops.reshape(-1, crops.shape[2], crops.shape[3])
        if "loss" in self.output_type:
            new_batch = {
                "image": crops,
                "D": new_D,
                "zxy_mask": output["zxy_mask"],
                "grade": batch["grade"],
            }
        else:
            new_batch = {"image": crops, "D": new_D, "zxy_mask": output["zxy_mask"]}

        return self.encoder_decoder(new_batch)

    def extract_crop_3d(self, image, z, x, y):
        # image: tensor of shape (D, H, W)
        # z, x, y: scalars
        D, H, W = image.shape
        z = int(round(z.item()))
        x = int(round(W * (x.item() / self.xy_max)))
        y = int(round(H * (y.item() / self.xy_max)))
        # Ensure coordinates are within bounds
        z = max(0, min(z, D - 1))
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        # Define the crop boundaries
        half_size = self.crop_size // 2
        half_depth_size = self.depth_size // 2
        z_min = max(0, z - half_depth_size)
        z_max = min(D, z + half_depth_size)
        y_min = max(0, y - half_size)
        y_max = min(H, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(W, x + half_size)
        # Crop the volume
        crop = image[
            z_min:z_max, y_min:y_max, x_min:x_max
        ]  # Shape: (D_crop, H_crop, W_crop)
        # Pad the crop if necessary to get the desired size
        pad_z_left = max(0, half_depth_size - z)
        pad_z_right = max(0, (z + half_depth_size) - D)
        pad_y_left = max(0, half_size - y)
        pad_y_right = max(0, (y + half_size) - H)
        pad_x_left = max(0, half_size - x)
        pad_x_right = max(0, (x + half_size) - W)
        # Note: pad order is (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right)
        pad = (
            pad_x_left,
            pad_x_right,
            pad_y_left,
            pad_y_right,
            pad_z_left,
            pad_z_right,
        )
        crop = F.pad(crop, pad, mode="constant", value=0)
        # Now crop shape should be (D_crop_padded, H_crop_padded, W_crop_padded)
        # Add channel dimension
        if self.in_chans == 1:
            crop = crop.unsqueeze(0)
        else:
            crop = crop.unsqueeze(0).repeat(self.in_chans, 1, 1, 1)
        # Resize crop to expected input size if necessary
        crop = F.interpolate(
            crop.unsqueeze(
                0
            ),  # Shape: (1, in_chans, D_crop_padded, H_crop_padded, W_crop_padded)
            size=(self.depth_size, self.crop_size, self.crop_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # Shape: (in_chans, D_crop_padded, H_crop_padded, W_crop_padded)
        return crop


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


def do_dynamic_match_truth10(xy, truth_xy, threshold=3):
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
    assert torch.any(truth != -1)  # need at least one valid label

    t = truth.reshape(-1)
    g = grade.reshape(-1, 3)

    eps = 1e-5
    loss = F.nll_loss(torch.clamp(g, eps, 1 - eps).log(), t, weight=weight)
    # loss = F.cross_entropy(g, t, weight=weight, ignore_index=-1)
    return loss


def F_z_loss(z, z_truth, mask, z_max=None):
    z_truth = z_truth.float()
    assert torch.any(z_truth != -1)  # need at least one valid label
    z_max = z_max.float().unsqueeze(1)
    if z_max is not None:
        z = z / z_max
        z_truth = z_truth / z_max
    loss = F.mse_loss(z[mask], z_truth[mask])
    return loss


def F_xy_loss(xy, xy_truth, mask, xy_max=None):
    xy_truth = xy_truth.float()
    if xy_max is not None:
        xy = xy / xy_max
        xy_truth = xy_truth / xy_max
    loss = F.mse_loss(xy[mask], xy_truth[mask])
    return loss


def F_heatmap_loss(heatmap, truth, D):
    heatmap = torch.split_with_sizes(heatmap, D, 0)
    truth = torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    loss = 0
    for i in range(num_image):
        p, q = truth[i], heatmap[i]
        D, _, _, _, _ = p.shape  # D, num_point, num_grade, H, W

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
        series_type: str = "sagittal t1",
        direction: str = "left",
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

        self.series_type = series_type
        if isinstance(series_type, str):
            self.series_type = [series_type]
        if any(
            st not in ["sagittal t1", "sagittal t2/stir", "axial t2"]
            for st in self.series_type
        ):
            raise ValueError("Invalid series")

        if direction not in ["left", "right"]:
            raise ValueError("Invalid direction")
        self.direction = direction
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
            if series_description not in self.series_type:
                continue

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

    def _get_instance_numbers_needed(self, sample_info) -> Set[int]:
        """
        Returns a set of instance_numbers needed for the annotations of this sample.
        """
        CONDITIONS = {
            "sagittal t2/stir": [
                "Spinal Canal Stenosis",
                "Spinal Canal Stenosis",
            ],
            "axial t2": ["Left Subarticular Stenosis", "Right Subarticular Stenosis"],
            "sagittal t1": [
                "Left Neural Foraminal Narrowing",
                "Right Neural Foraminal Narrowing",
            ],
        }

        LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

        study_id = sample_info["study_id"]
        series_description = sample_info["series_description"]

        coords_df = self.coords[self.coords["study_id"] == study_id]

        conditions_list = CONDITIONS.get(series_description, [])

        instance_numbers_needed = set()

        for condition in conditions_list:
            for level in LEVELS:
                condition_coords = coords_df[
                    (coords_df["condition"] == condition)
                    & (coords_df["level"] == level)
                ]
                if not condition_coords.empty:
                    instance_number = condition_coords["instance_number"].values[0]
                    instance_numbers_needed.add(instance_number)

        return instance_numbers_needed

    def _get_all(self, idx, count=0) -> Union[dict, torch.Tensor]:
        sample_info = self.samples[idx]
        try:
            instance_numbers_needed = self._get_instance_numbers_needed(sample_info)

            # Read the series and process the volume
            volume, dicom_df, error_code = self._read_series(
                sample_info["study_id"],
                sample_info["series_id"],
                sample_info["series_description"],
                instance_numbers_needed,
            )

            # Handle any errors in reading the series
            if error_code:
                if idx in self.valid_indices:
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

            if torch.all(z == -1):
                if idx in self.valid_indices:
                    self.valid_indices.remove(idx)
                if count > 10:
                    raise ValueError("No valid coordinates found.")
                return self._get_all(self._get_new_index(idx), count + 1)

            grade = None
            heatmap = None
            if "grade" in self.train_on:
                grade = self._prepare_grade(sample_info)

                if torch.all(grade == -1):
                    if idx in self.valid_indices:
                        self.valid_indices.remove(idx)
                    if count > 10:
                        raise ValueError("No valid grade found.")
                    return self._get_all(self._get_new_index(idx), count + 1)

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

            out["study_id"] = sample_info["study_id"]
            out["series_description"] = sample_info["series_description"]

            if "sagittal t2/stir" in sample_info["series_description"].lower():
                if "heatmap" in out:
                    out["heatmap"] = out["heatmap"][:, :5, ...]
                if "grade" in out:
                    out["grade"] = out["grade"][:5]
                if "xy" in out:
                    out["xy"] = out["xy"][:5, :]
                if "z" in out:
                    out["z"] = out["z"][:5]
            elif "sagittal t1" in sample_info["series_description"].lower():
                if self.direction == "left":
                    if "heatmap" in out:
                        out["heatmap"] = out["heatmap"][:, :5, ...]
                    if "grade" in out:
                        out["grade"] = out["grade"][:5]
                    if "xy" in out:
                        out["xy"] = out["xy"][:5, :]
                    if "z" in out:
                        out["z"] = out["z"][:5]
                elif self.direction == "right":
                    if "heatmap" in out:
                        out["heatmap"] = out["heatmap"][:, 5:, ...]
                    if "grade" in out:
                        out["grade"] = out["grade"][5:]
                    if "xy" in out:
                        out["xy"] = out["xy"][5:, :]
                    if "z" in out:
                        out["z"] = out["z"][5:]

            return out

        except Exception as e:
            if idx in self.valid_indices:
                self.valid_indices.remove(idx)
            if count > 10:
                raise e
            return self._get_all(self._get_new_index(idx), count + 1)

    def __getitem__(self, idx) -> Union[dict, torch.Tensor]:
        return self._get_all(idx)

    def crop_volume(instance_numbers, instance_numbers_needed, depth, max_depth):
        required_indices = [
            i
            for i, num in enumerate(instance_numbers)
            if num in instance_numbers_needed
        ]

        selected_indices = set(required_indices)

        # how many to include
        remaining_depth = max_depth - len(selected_indices)

        if remaining_depth > 0:
            # Determine optional slice indices (excluding required ones)
            optional_indices = [i for i in range(depth) if i not in selected_indices]

            # Calculate stride to evenly sample the remaining slices
            stride = max(1, len(optional_indices) / remaining_depth)
            sampled_optional = [
                optional_indices[int(i * stride)]
                for i in range(remaining_depth)
                if int(i * stride) < len(optional_indices)
            ]
            selected_indices.update(sampled_optional)

        selected_indices = sorted(selected_indices)

        if len(selected_indices) > max_depth:
            selected_indices = selected_indices[:max_depth]

        return selected_indices

    def _read_series(
        self,
        study_id,
        series_id,
        series_description,
        instance_numbers_needed: Optional[Set[int]] = None,
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
            selected_indices = self.crop_volume(
                instance_numbers, instance_numbers_needed, depth, self.max_depth
            )
            instance_numbers = [instance_numbers[i] for i in selected_indices]
            dicom_files = [dicom_files[i] for i in selected_indices]

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

    def generate_smoothed_heatmap(
        self, z_tensor, xy_tensor, grades_tensor, image_shape
    ):
        num_point = xy_tensor.shape[0]
        num_grade = 3  # Assuming grades are 0, 1, 2
        D, H, W = image_shape

        # Initialize heatmap with zeros
        heatmap = torch.zeros((D, num_point, num_grade, H, W), dtype=torch.float32)

        # Standard deviations for the Gaussian kernel
        sigma_spatial = 1.0  # Spatial dimensions (x, y, z)
        sigma_grade = 1.0  # Grade dimension
        tmp_size_spatial = int(sigma_spatial * 3)
        tmp_size_grade = int(sigma_grade * 3)

        for idx in range(num_point):
            x = xy_tensor[idx, 0]
            y = xy_tensor[idx, 1]
            z = z_tensor[idx]
            grade = grades_tensor[idx]

            if z == -1 or grade == -1:
                continue  # Skip if z or grade is invalid

            # Convert to float for precise calculations
            x = x.item()
            y = y.item()
            z = z.item()
            g = grade.item()

            # Define the ranges for x, y, z, and grade
            x_min = max(0, int(x - tmp_size_spatial))
            x_max = min(W, int(x + tmp_size_spatial + 1))
            y_min = max(0, int(y - tmp_size_spatial))
            y_max = min(H, int(y + tmp_size_spatial + 1))
            z_min = max(0, int(z - tmp_size_spatial))
            z_max = min(D, int(z + tmp_size_spatial + 1))
            g_min = max(0, int(g - tmp_size_grade))
            g_max = min(num_grade, int(g + tmp_size_grade + 1))

            # Create coordinate grids
            grid_x = torch.arange(x_min, x_max, dtype=torch.float32)
            grid_y = torch.arange(y_min, y_max, dtype=torch.float32)
            grid_z = torch.arange(z_min, z_max, dtype=torch.float32)
            grid_g = torch.arange(g_min, g_max, dtype=torch.float32)

            # Create meshgrid for the Gaussian kernel
            zz, yy, xx, gg = torch.meshgrid(
                grid_z, grid_y, grid_x, grid_g, indexing="ij"
            )

            # Compute the Gaussian kernel
            gaussian = torch.exp(
                -(
                    ((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2)
                    / (2 * sigma_spatial**2)
                    + ((gg - g) ** 2) / (2 * sigma_grade**2)
                )
            )

            # Convert indices to long tensors for indexing
            x_indices = torch.arange(x_min, x_max, dtype=torch.long)
            y_indices = torch.arange(y_min, y_max, dtype=torch.long)
            z_indices = torch.arange(z_min, z_max, dtype=torch.long)
            g_indices = torch.arange(g_min, g_max, dtype=torch.long)

            # Ensure that the Gaussian and heatmap indices align
            # Gaussian shape: [len(grid_z), len(grid_y), len(grid_x), len(grid_g)]
            # Heatmap slice shape: [len(z_indices), len(y_indices), len(x_indices), len(g_indices)]

            # Assign the Gaussian values to the heatmap using advanced indexing
            heatmap[
                z_indices[:, None, None, None],
                idx,
                g_indices[None, None, None, :],
                y_indices[None, :, None, None],
                x_indices[None, None, :, None],
            ] = torch.maximum(
                heatmap[
                    z_indices[:, None, None, None],
                    idx,
                    g_indices[None, None, None, :],
                    y_indices[None, :, None, None],
                    x_indices[None, None, :, None],
                ],
                gaussian,
            )

        # Normalize the heatmap over spatial dimensions (D, H, W)
        heatmap = heatmap / (heatmap.sum(dim=(0, 3, 4), keepdim=True) + 1e-8)

        return heatmap

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
        elif isinstance(batch[0][key], torch.Tensor):
            new_batch[key] = torch.stack([sample[key] for sample in batch])
        else:
            new_batch[key] = [sample[key] for sample in batch]
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
    Combine heatmaps for each point based on their corresponding grades and save as PNG
    images for selected depths.

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


def visualize_predictions_and_crop(model, batch, output_dir="plots"):
    """
    Takes in an instance of SecondStageModelV2 and a batch of data.
    Uses the zxy_predictor to get z, x, y predictions.
    Plots the predicted x,y positions on the original image.
    Extracts the crop for one of the predicted points and visualizes it.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Get device
    device = next(model.parameters()).device

    # Use the zxy_predictor to get predictions
    with torch.no_grad():
        output = model.zxy_predictor(batch)
        z_pred = output["z"]  # List of tensors, each tensor is (num_points,)
        xy_pred = output["xy"]  # List of tensors, each tensor is (num_points, 2)
        images = batch["image"].to(device)  # Shape: (sum D_i, H, W)
        D_list = batch["D"]  # Tensor of depths per image

    num_images = len(D_list)
    D_cumsum = [0] + torch.cumsum(D_list, dim=0).tolist()

    series_description = batch["series_description"]
    # For each image in the batch
    for i in range(num_images):
        image_start = int(D_cumsum[i])
        image_end = int(D_cumsum[i + 1])
        image = images[image_start:image_end]  # Shape: (D_i, H, W)
        z_i = z_pred[i].round().long()  # Shape: (num_points,)
        xy_i = xy_pred[i]  # Shape: (num_points, 2)
        num_points = z_i.shape[0]

        W = image.shape[2]
        H = image.shape[1]
        # Plot the image slice with predicted points
        # We'll plot the slice at the predicted z position of the first point
        os.makedirs(output_dir, exist_ok=True)
        if "sagittal" in series_description[i].lower():
            for idx in range(2):
                if idx == 0:
                    z_slice = z_i[0].item()
                else:
                    z_slice = z_i[-1].item()
                if z_slice < 0 or z_slice >= image.shape[0]:
                    z_slice = (
                        image.shape[0] // 2
                    )  # Default to middle slice if out of bounds

                image_slice = image[z_slice].cpu().numpy()  # Shape: (H, W)

                plt.figure(figsize=(6, 6))
                plt.imshow(image_slice, cmap="gray")

                # Transform predicted x, y coordinates to image pixel coordinates
                if idx == 0:
                    x_coords = xy_i[:5, 0].cpu().numpy() * (W / model.xy_max)
                    y_coords = xy_i[:5, 1].cpu().numpy() * (H / model.xy_max)
                else:
                    x_coords = xy_i[-5:, 0].cpu().numpy() * (W / model.xy_max)
                    y_coords = xy_i[-5:, 1].cpu().numpy() * (H / model.xy_max)

                plt.scatter(x_coords, y_coords, c="r", marker="x")
                plt.title(f"Image {i}, Slice {z_slice}")
                plt.axis("off")
                # save the plot
                plt.savefig(f"{output_dir}/image_{i}_slice_{z_slice}.png")
                plt.close()
        elif "axial" in series_description[i].lower():
            for idx in range(num_points):
                z_slice = z_i[idx].item()
                if z_slice < 0 or z_slice >= image.shape[0]:
                    z_slice = image.shape[0] // 2
                x_coords = xy_i[idx, 0].cpu().numpy() * (W / model.xy_max)
                y_coords = xy_i[idx, 1].cpu().numpy() * (H / model.xy_max)

                plt.figure(figsize=(6, 6))
                plt.imshow(image[z_slice].cpu().numpy(), cmap="gray")
                plt.scatter(x_coords, y_coords, c="r", marker="x")
                plt.title(f"Image {i}, Slice {z_slice}")
                plt.axis("off")
                # save the plot
                plt.savefig(f"{output_dir}/image_{i}_slice_{z_slice}.png")
                plt.close()

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
        for idx1, cond in enumerate(CONDITIONS[series_description[i]]):
            for idx2, level in enumerate(LEVELS):
                idx = idx1 * len(LEVELS) + idx2
                z_p = z_i[idx]
                x_p = xy_i[idx, 0]
                y_p = xy_i[idx, 1]
                crop = model.extract_crop_3d(image, z_p, x_p, y_p)

                D, H, W = image.shape
                z = int(round(z_p.item()))
                # Ensure coordinates are within bounds
                z = max(0, min(z_p, D - 1))
                # Define the cro boundaries
                half_depth_size = model.depth_size // 2
                z_min = z - half_depth_size
                z_max = z + half_depth_size
                z_slices = list(range(z_min, z_max))

                # Visualize the crop
                def visualize_crop(crop):
                    # crop: tensor of shape (1, D, H, W)
                    crop = crop[0, :, :, :]  # Shape: (D, H, W)
                    D, H, W = crop.shape

                    # Plot each slice
                    fig, axes = plt.subplots(1, D, figsize=(D * 2, 2))
                    # create title
                    title = f"Image {i}, {cond}, {level}"
                    fig.suptitle(title)
                    for idx in range(len(z_slices)):
                        slice_idx = z_slices[idx]
                        axes[idx].imshow(crop[idx].cpu().numpy(), cmap="gray")
                        axes[idx].axis("off")
                        axes[idx].set_title(f"Slice {slice_idx}")
                    fig.savefig(f"{output_dir}/crop_image_{i}_{cond}_{level}.png")
                    plt.close()

                visualize_crop(crop)
