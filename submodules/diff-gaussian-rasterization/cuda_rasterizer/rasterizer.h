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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED
#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

        static void preprocess(
            int* P_counter,
            char* geom_buffer,
            const int P,
            int D,
            int M,
            const int width,
            int height,
            int tile_side,
            int patches,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* viewmatrix,
            const float* projmatrix,
            const float* cam_pos,
            const float* crop_box,
            const float tan_fovx,
            float tan_fovy,
            const bool prefiltered,
            bool debug);

		static int render(
            char* geom_buffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            const int P,
            const int* point_offsets,
            const float* background,
            const int width,
            int height,
            int tile_side,
            int patches,
            const float* colors_precomp,
            const float* transmission,
            float* out_color,
            bool debug);

		static void backward(
            const int P, int D, int M, int R,
            const int* point_offsets,
            const float* background,
            const int width, int height, int tile_side, int patches,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const std::vector<float*> viewmatrix,
            const std::vector<float*> projmatrix,
            const std::vector<float*> campos,
            const std::vector<float> tan_fovx, std::vector<float> tan_fovy,
            char* geom_buffer,
            char* binning_buffer,
            char* img_buffer,
            const float* out_color,
            const float* dL_dpix,
            float* dL_dmean2D,
            float* dL_dconic,
            float* dL_dopacity,
            float* dL_dcolor,
            float* dL_dmean3D,
            float* dL_dcov3D,
            float* dL_dsh,
            float* dL_dscale,
            float* dL_drot,
            bool debug);
	};
};

#endif