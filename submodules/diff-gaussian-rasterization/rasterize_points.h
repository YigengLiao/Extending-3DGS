/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 PreprocessCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const std::vector<torch::Tensor>& viewmatrix,
    const std::vector<torch::Tensor>& projmatrix,
    const std::vector<float>& tan_fovx,
    const std::vector<float>& tan_fovy,
    const int image_height,
    const int image_width,
    const int tile_side,
    const torch::Tensor& sh,
    const int degree,
    const std::vector<torch::Tensor>& campos,
    const std::vector<torch::Tensor>& crop_box,
    const bool prefiltered,
    const bool debug);

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RenderCUDA(
    const int P,
    const torch::Tensor& point_offsets,
    const torch::Tensor& geomBuffer,
    const torch::Tensor& background,
    const torch::Tensor& colors,
    const torch::Tensor& transmission,
    const int image_height,
    const int image_width,
    const int tile_side,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& point_offsets,
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const std::vector<torch::Tensor>& viewmatrix,
    const std::vector<torch::Tensor>& projmatrix,
    const std::vector<float>& tan_fovx,
    const std::vector<float>& tan_fovy,
    const int image_height,
    const int image_width,
    const int tile_side,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& sh,
    const int degree,
    const std::vector<torch::Tensor>& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& out_color,
    const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);