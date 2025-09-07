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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include <fstream>
#include <string>
#include <functional>

#pragma message(__FILE__ " : reached")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

template<class T>
inline void d2d_elems(T* dst, const T* src, size_t count) {
    cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice);
}

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
    const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int patches = campos.size();
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  torch::Tensor geomBuffer0 = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc0 = resizeFunctional(geomBuffer0);
  size_t chunk_size0 = CudaRasterizer::required<CudaRasterizer::GeometryState>(P * patches);
  char* chunkptr0 = geomFunc0(chunk_size0);

  CudaRasterizer::GeometryState geomState0 = CudaRasterizer::GeometryState::fromChunk(chunkptr0, P * patches);

  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);

  torch::Tensor point_id;
  torch::Tensor v_dir;

  torch::Tensor point_offsets = torch::zeros({patches + 1},
                                                  torch::TensorOptions().dtype(torch::kInt32).device(device));
  if(P != 0)
  {
      int M = 0;
      if(sh.size(0) != 0)
      {
        M = sh.size(1);
      }

      int *P_counter = nullptr;
      cudaMalloc(&P_counter, sizeof(int));
      cudaMemset(P_counter, 0, sizeof(int));
      for (size_t i = 0; i < patches; ++i) {
          CudaRasterizer::Rasterizer::preprocess(
            P_counter,
            reinterpret_cast<char*>(geomBuffer0.contiguous().data_ptr()),
            P, degree, M,
            W, H, tile_side, patches,
            means3D.contiguous().data<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data<float>(),
            opacity.contiguous().data<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(),
            viewmatrix[i].contiguous().data<float>(),
            projmatrix[i].contiguous().data<float>(),
            campos[i].contiguous().data<float>(),
            crop_box[i].contiguous().data<float>(),
            tan_fovx[i],
            tan_fovy[i],
            prefiltered,
            debug);
          cudaDeviceSynchronize();
          cudaMemcpy(point_offsets.data_ptr<int>() + (i + 1), P_counter, sizeof(int), cudaMemcpyDeviceToDevice);
      }

      int newP;
      cudaMemcpy(&newP, P_counter, sizeof(int), cudaMemcpyDeviceToHost);

      size_t chunk_size = CudaRasterizer::required<CudaRasterizer::GeometryState>(newP);
      char* chunkptr = geomFunc(chunk_size);
      CudaRasterizer::GeometryState geomState = CudaRasterizer::GeometryState::fromChunk(chunkptr, newP);

      d2d_elems(geomState.depths,             geomState0.depths,             newP);
      d2d_elems(geomState.clamped,            geomState0.clamped,            newP * 3);
      d2d_elems(geomState.internal_radii,     geomState0.internal_radii,     newP);
      d2d_elems(geomState.means2D,            geomState0.means2D,            newP);
      d2d_elems(geomState.cov3D,              geomState0.cov3D,              newP * 6);
      d2d_elems(geomState.conic_opacity,      geomState0.conic_opacity,      newP);
      d2d_elems(geomState.conic0_correction,  geomState0.conic0_correction,  newP);
      d2d_elems(geomState.rgb,                geomState0.rgb,                newP * 3);
      d2d_elems(geomState.tiles_touched,  	  geomState0.tiles_touched,    	 newP);
      d2d_elems(geomState.point_id,           geomState0.point_id,           newP);
      d2d_elems(geomState.v_dir,              geomState0.v_dir,              newP * 3);

      cudaFree(P_counter);
      point_id = torch::from_blob(
        geomState.point_id, {newP}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)).clone();
      v_dir = torch::from_blob(
        geomState.v_dir, {newP * 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).clone();

  }

  return std::make_tuple(point_id, point_offsets, v_dir, geomBuffer);
}

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
    const bool debug)
{
  const int H = image_height;
  const int W = image_width;
  const int patches = point_offsets.size(0) - 1;

  if (tile_side == 0 && patches != 1) {
    AT_ERROR("When patches=false, geomBuffer list must contain exactly one element");
  }
  auto int_opts = point_offsets.options().dtype(torch::kInt32);
  auto float_opts = point_offsets.options().dtype(torch::kFloat32);
  torch::Tensor out_color;
  if (tile_side == 0)
      out_color = torch::full({patches, NUM_CHANNELS, H, W}, 0.0, float_opts);
  else
      out_color = torch::full({patches, NUM_CHANNELS, tile_side, tile_side}, 0.0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
      rendered = CudaRasterizer::Rasterizer::render(
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        binningFunc,
        imgFunc,
        P,
        point_offsets.contiguous().data<int>(),
        background.contiguous().data<float>(),
        W, H, tile_side, patches,
        colors.contiguous().data<float>(),
        transmission.contiguous().data<float>(),
        out_color.contiguous().data<float>(),
        debug);
  }
  return std::make_tuple(rendered, out_color, geomBuffer, binningBuffer, imgBuffer);
}


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
    const bool debug)
{
  const int P = means3D.size(0);
  const int patches = campos.size();
  const int H = image_height;
  const int W = image_width;

  int M = 0;
  if(sh.size(0) != 0)
  {
    M = sh.size(1);
  }
  std::vector<float*> viewmatrix0(patches);
  std::vector<float*> projmatrix0(patches);
  std::vector<float> tan_fovx0(patches);
  std::vector<float> tan_fovy0(patches);
  std::vector<float*> campos0(patches);
  for (size_t i = 0; i < patches; ++i) {
    viewmatrix0[i] = viewmatrix[i].contiguous().data<float>();
    projmatrix0[i] = projmatrix[i].contiguous().data<float>();
    tan_fovx0[i] = tan_fovx[i];
    tan_fovy0[i] = tan_fovy[i];
    campos0[i] = campos[i].contiguous().data<float>();
  }



  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if(P != 0)
  {

      CudaRasterizer::Rasterizer::backward(P, degree, M, R,
      point_offsets.contiguous().data<int>(),
      background.contiguous().data<float>(),
      W, H, tile_side, patches,
      means3D.contiguous().data<float>(),
      sh.contiguous().data<float>(),
      colors.contiguous().data<float>(),
      scales.data_ptr<float>(),
      scale_modifier,
      rotations.data_ptr<float>(),
      cov3D_precomp.contiguous().data<float>(),
      viewmatrix0,
      projmatrix0,
      campos0,
      tan_fovx0,
      tan_fovy0,
      reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
      out_color.contiguous().data<float>(),
      dL_dout_color.contiguous().data<float>(),
      dL_dmeans2D.contiguous().data<float>(),
      dL_dconic.contiguous().data<float>(),
      dL_dopacity.contiguous().data<float>(),
      dL_dcolors.contiguous().data<float>(),
      dL_dmeans3D.contiguous().data<float>(),
      dL_dcov3D.contiguous().data<float>(),
      dL_dsh.contiguous().data<float>(),
      dL_dscales.contiguous().data<float>(),
      dL_drotations.contiguous().data<float>(),
      debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
        torch::Tensor& means3D,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix)
{
  const int P = means3D.size(0);

  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

  if(P != 0)
  {
    CudaRasterizer::Rasterizer::markVisible(P,
        means3D.contiguous().data<float>(),
        viewmatrix.contiguous().data<float>(),
        projmatrix.contiguous().data<float>(),
        present.contiguous().data<bool>());
  }

  return present;
}
