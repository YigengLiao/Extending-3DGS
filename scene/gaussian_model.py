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
# NOTE — Modifications:
# Added support for constructing implicit MLP layers and hash encodings.
# Each Gaussian no longer stores SH (spherical harmonics) coefficients; instead,
# two per-Gaussian latent vectors are introduced for color and transmittance
# estimation (e.g., u_gauss and material). To enable sparse gradients,
# the activations for scaling, rotation, and opacity are moved into the rendering
# pipeline (`diff_gaussian_rasterization`).

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
from simple_knn._C import distCUDA2
import numpy as np
import torch
import torch.nn as nn

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from networks import ColorMLP, HashGridEncoder, DensityMLP
from utils.graphics_utils import BasicPointCloud

class GaussianModel:

    def setup_functions(self):


        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.grid_activation = torch.sigmoid
        self.inverse_grid_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, optimizer_type="default", iso=True):
        self.optimizer_type = optimizer_type
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)
        self._u_gauss = torch.empty(0)
        self._material = torch.empty(0)
        self._mlp_normal = torch.empty(0)
        self._mlp_color = torch.empty(0)
        self._hashgrid_xyz = torch.empty(0)
        self._hashgrid_u = torch.empty(0)
        self._hashgrid_mat = torch.empty(0)
        self._mlp_density = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self._exposure = torch.empty(0)
        self.spatial_lr_scale = 0
        self.iso = iso
        self._rotation = torch.empty(0)
        self.setup_functions()

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scaling
        # return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation
        # return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self._opacity
        # return self.opacity_activation(self._opacity)

    @property
    def get_u_gauss(self):
        return self._u_gauss

    @property
    def get_material(self):
        return self._material

    @property
    def get_mlp_color(self):
        return self._mlp_color

    @property
    def get_hashgrid_xyz(self):
        return self._hashgrid_xyz

    @property
    def get_hashgrid_density(self):
        return self._hashgrid_density

    @property
    def get_mlp_density(self):
        return self._mlp_density

    @property
    def get_max_radii2D(self):
        return self.max_radii2D

    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def create_from_pcd(self,
                        pcd: BasicPointCloud,
                        cam_infos: list,
                        spatial_lr_scale: float,
                        ):

        self.spatial_lr_scale = spatial_lr_scale

        pcdpp = np.repeat(pcd.points, repeats=2, axis=0)
        point_cloud1 = torch.tensor(np.asarray(pcdpp)).float().cuda()

        mean = point_cloud1.mean(dim=0)
        std = point_cloud1.std(dim=0)

        lower = mean - 3 * std
        upper = mean + 3 * std
        mask = ((point_cloud1 >= lower) & (point_cloud1 <= upper)).all(dim=1)
        point_cloud = point_cloud1[mask]

        self.point_weiyi = torch.min(point_cloud, dim=0)[0].unsqueeze(0)
        self.point_scale = torch.max(point_cloud, dim=0)[0].unsqueeze(0)
        point_cloud2 = torch.randn(0, 3, dtype=torch.float, device="cuda") * (self.point_scale - self.point_weiyi) + self.point_weiyi
        self.point_scale = 0.9 / (self.point_scale - self.point_weiyi)
        fused_point_cloud = torch.cat((point_cloud1, point_cloud2), dim=0)

        N = fused_point_cloud.shape[0]
        print("Number of points at initialisation : ", N)

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        u_gauss = torch.randn(N, 2, dtype=torch.float, device="cuda")
        material = torch.randn(N, 32, dtype=torch.float, device="cuda")

        if not self.iso:
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            self._rotation = nn.Parameter(rots.requires_grad_(True))
        else:
            scales = torch.log(torch.sqrt(dist2))[..., None]

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._u_gauss = nn.Parameter(u_gauss.requires_grad_(True))
        self._material = nn.Parameter(material.requires_grad_(True))

        self._mlp_color = ColorMLP(n_dir=3, n_extra=64)

        self._hashgrid_xyz = HashGridEncoder(n_levels=16)
        self._hashgrid_density = HashGridEncoder(n_levels=16)

        self._mlp_density = DensityMLP(input_dim=34)

        self.max_radii2D = torch.zeros(N, device="cuda")

        # self.exposure_mapping = {ci.image_name: i for i, ci in enumerate(cam_infos)}
        self.pretrained_exposures = None

        # E_init = torch.eye(3, 4, device="cuda").repeat(len(cam_infos), 1, 1)
        # self._exposure = nn.Parameter(E_init.requires_grad_(True))

        print(f"[Init] Points N = {N}")

    def training_setup(self, training_args):

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_gauss = [
            # 1 PATCH
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._u_gauss], 'lr': 2e-2, "name": "u_gauss"},
            {'params': [self._material], 'lr': 2e-2, "name": "material"}

            # PATCHES
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._u_gauss], 'lr': 1e-2, "name": "u_gauss"},
            # {'params': [self._material], 'lr': 1e-2, "name": "material"}
        ]
        l_mlp = [
            # 1 PATCH
            {'params': self._mlp_color.parameters(), 'lr': 1e-3, "name": "mlp_color"},
            {'params': self._mlp_density.parameters(), 'lr': 1e-3, "name": "mlp_density"},
            {'params': self._hashgrid_xyz.parameters(), 'lr': 2e-2, "name": "hashgrid_xyz"},
            {'params': self._hashgrid_density.parameters(), 'lr': 2e-2, "name": "hashgrid_xyz"},

            # PATCHES
            # {'params': self._mlp_color.parameters(), 'lr': 3e-4, "name": "mlp_color"},
            # {'params': self._mlp_density.parameters(), 'lr': 3e-4, "name": "mlp_density"},
            # {'params': self._hashgrid_xyz.parameters(), 'lr': 5e-3, "name": "hashgrid_xyz"},
            # {'params': self._hashgrid_density.parameters(), 'lr': 5e-3, "name": "hashgrid_xyz"},
        ]

        if not self.iso:
            l_gauss.append({'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"})

        if self.optimizer_type == "default":

            self.optimizer_gauss = torch.optim.SparseAdam(l_gauss, eps=1e-15)
            self.optimizer_mlp = torch.optim.Adam(l_mlp, eps=1e-15)

        # self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

        # self.exposure_scheduler_args = get_expon_lr_func(
        #     training_args.exposure_lr_init,
        #     training_args.exposure_lr_final,
        #     lr_delay_steps=training_args.exposure_lr_delay_steps,
        #     lr_delay_mult=training_args.exposure_lr_delay_mult,
        #     max_steps=training_args.iterations
        # )

    def update_learning_rate(self, iteration):
        # ''' Learning rate scheduling per step '''
        # if self.pretrained_exposures is None:
        #     for param_group in self.exposure_optimizer.param_groups:
        #         param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer_gauss.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

