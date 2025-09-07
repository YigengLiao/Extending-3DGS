#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import math
import copy

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # self.original_image = gt_image.clamp(0.0, 1.0).to("cuda")
        self.original_image = gt_image.clamp(0.0, 1.0).to("cpu")
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.crop_box = (0, 0)
        self.tile_side = 0

def split_camera_into_tiles(cam, tile_side, enable, is_test_dataset):
    if not enable or is_test_dataset:
        if is_test_dataset:
            cam.original_image.to("cpu")
        return [cam]


    H, W = cam.image_height, cam.image_width
    n_cols = math.ceil(W / tile_side)
    n_rows = math.ceil(H / tile_side)
    Wp, Hp = n_cols * tile_side, n_rows * tile_side
    pad_right, pad_bottom = Wp - W, Hp - H

    cam.tile_side = tile_side

    def pad_last2(t):
        if t is None or (pad_right == 0 and pad_bottom == 0):
            return t

        return nn.functional.pad(t, (0, pad_right, 0, pad_bottom), mode="constant", value=-1)

    img_p = pad_last2(getattr(cam, "original_image", None))
    mask_p = pad_last2(getattr(cam, "alpha_mask", None))
    invd_p = pad_last2(getattr(cam, "invdepthmap", None))
    dmsk_p = pad_last2(getattr(cam, "depth_mask", None))

    tiles: List[type(cam)] = []
    for j in range(n_rows):
        for i in range(n_cols):
            u0, v0 = i * tile_side, j * tile_side
            u1, v1 = u0 + tile_side, v0 + tile_side

            tile_cam = copy.copy(cam)
            tile_cam.uid = f"{cam.uid}_tile{j}_{i}"
            tile_cam.crop_box = ( u0, v0)

            if img_p is not None:
                tile_cam.original_image = img_p[..., v0:v1, u0:u1]
            if mask_p is not None:
                tile_cam.alpha_mask = mask_p[..., v0:v1, u0:u1]
            if invd_p is not None:
                tile_cam.invdepthmap = invd_p[..., v0:v1, u0:u1]
            if dmsk_p is not None:
                tile_cam.depth_mask = dmsk_p[..., v0:v1, u0:u1]

            tiles.append(tile_cam)

    return tiles
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

