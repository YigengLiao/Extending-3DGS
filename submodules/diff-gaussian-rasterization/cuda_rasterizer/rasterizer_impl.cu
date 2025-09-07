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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

#include <stdio.h>
__global__ void countTileHits(
    int P,
    const float2* points_xy,
    const int* radii,
    dim3 grid,
    uint32_t* tile_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P || !(radii[idx] > 0)) return;

    uint2 mn, mx;  getRect(points_xy[idx], radii[idx], mn, mx, grid);
    for (int y = mn.y; y < mx.y; ++y)
        for (int x = mn.x; x < mx.x; ++x)
            atomicAdd(&tile_counts[y * grid.x + x], 1u);
}

__global__ void initPointIdKernel(uint32_t* point_id, size_t P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < P) {
        point_id[idx] = static_cast<int>(idx);
    }
}

__global__ void updateOpacity(float4* conic_opacity, const float* opacity, size_t P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < P) {
        conic_opacity[idx].w *= opacity[idx];
    }
}

__global__ void fillTileLists(
    int P,
    const float2* points_xy,
    const int*  radii,
    dim3 grid,
    uint32_t* tile_offsets,
    uint32_t* point_list)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P || !(radii[idx] > 0))  return;

    uint2 mn, mx;  getRect(points_xy[idx], radii[idx], mn, mx, grid);
    for (int ty = mn.y; ty < mx.y; ++ty)
        for (int tx = mn.x; tx < mx.x; ++tx)
        {
            uint32_t tile = ty * grid.x + tx;
            uint32_t at   = atomicAdd(&tile_offsets[tile], 1u);
            point_list[at] = idx;
        }
}

__global__ void finalizeRanges(
    int tiles,
    const uint32_t* offsets,
    const uint32_t* counts,
    uint2* ranges)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < tiles) {
        uint32_t end = offsets[t];
        ranges[t]    = make_uint2(end - counts[t], end);
    }
}


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{

    auto idx = cg::this_grid().thread_rank();

    if (idx >= P)
        return;


    if (radii[idx] > 0)
    {

        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

        uint2 rect_min, rect_max;

        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {

                uint64_t key = (uint64_t)(y * grid.x + x);
                key <<= 32;

                key |= *((uint32_t*)&depths[idx]);

                gaussian_keys_unsorted[off]   = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

namespace CudaRasterizer {
GeometryState GeometryState::fromChunk(char*& chunk, size_t P) {
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.conic0_correction, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.point_id, P, 128);
	obtain(chunk, geom.v_dir, P * 3, 128);

	return geom;
}
}
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_weight, N, 128);

	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P, dim3 tile_grid)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);

    const int tiles = tile_grid.x * tile_grid.y;
    obtain(chunk, binning.tile_counts,     tiles, 128);
    obtain(chunk, binning.tile_offsets,    tiles, 128);
	obtain(chunk, binning.ranges, 		   tiles, 128);
    cub::DeviceScan::ExclusiveSum(nullptr, binning.sorting_size,
                                  binning.tile_counts, binning.tile_offsets,
                                  tiles);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

void CudaRasterizer::Rasterizer::preprocess(
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
    bool debug
)
{

    GeometryState geomState = GeometryState::fromChunk(geom_buffer, patches * P);
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width  / (2.0f * tan_fovx);
    dim3 tile_grid;
    if (tile_side == 0) {
        tile_grid = dim3(
            (width + BLOCK_X - 1) / BLOCK_X,
            (height + BLOCK_Y - 1) / BLOCK_Y,
            1
        );
    } else {
        tile_grid = dim3(
            (tile_side + BLOCK_X - 1) / BLOCK_X,
            (tile_side + BLOCK_Y - 1) / BLOCK_Y,
            1
        );
    }

	const int tiles = tile_grid.x * tile_grid.y;
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
    {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, projmatrix,
        (glm::vec3*)cam_pos,
        crop_box,
        width, height, tile_side,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        geomState.internal_radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        geomState.conic0_correction,
        geomState.point_id,
        geomState.v_dir,
        tile_grid,
        geomState.tiles_touched,
        prefiltered,
        P_counter
    ), debug)
}


// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::render(
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
    bool debug
)
{
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
        {
            throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
        }

    updateOpacity<<<(P+255)/256,256>>>(
            geomState.conic_opacity,
            transmission,
            P);
    dim3 tile_grid;
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    const float* feature_ptr;
    int num_rendered;

    size_t img_chunk_size;
    char* img_chunkptr;
    ImageState imgState;
    size_t binning_chunk_size;
    char* binning_chunkptr;
    BinningState binningState;


    if (tile_side == 0) {

        tile_grid = dim3((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
        const int tiles = tile_grid.x * tile_grid.y;
        img_chunk_size = required<ImageState>(width * height);
        img_chunkptr = imageBuffer(img_chunk_size);
        imgState = ImageState::fromChunk(img_chunkptr, width * height);

        CHECK_CUDA(cub::DeviceScan::InclusiveSum(
            geomState.scanning_space,
            geomState.scan_size,
            geomState.tiles_touched,
            geomState.point_offsets,
            P
        ), debug)

        CHECK_CUDA(cudaMemcpy(
            &num_rendered,
            geomState.point_offsets + P - 1,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ), debug)

        binning_chunk_size = required<BinningState>(num_rendered, tile_grid);

        binning_chunkptr = binningBuffer(binning_chunk_size);

        binningState = BinningState::fromChunk(binning_chunkptr,
                                    num_rendered,
                                    tile_grid);

        CHECK_CUDA(cudaMemset(binningState.tile_counts,
                            0, tiles * sizeof(uint32_t)), debug);

        countTileHits<<< (P + 255) / 256, 256 >>>(
                P,
                geomState.means2D,
                geomState.internal_radii,
                tile_grid,
                binningState.tile_counts);
        CHECK_CUDA( , debug);

        CHECK_CUDA(cub::DeviceScan::ExclusiveSum(
                binningState.list_sorting_space,
                binningState.sorting_size,
                binningState.tile_counts,
                binningState.tile_offsets,
                tiles),
                debug);

        if (num_rendered > 0)
            fillTileLists<<< (P + 255) / 256, 256 >>>(
                P,
                geomState.means2D,
                geomState.internal_radii,
                tile_grid,
                binningState.tile_offsets,
                binningState.point_list);
            CHECK_CUDA( , debug);

        finalizeRanges<<<(tiles+255)/256,256>>>(
            tiles,
            binningState.tile_offsets,
            binningState.tile_counts,
            binningState.ranges);
        CHECK_CUDA( , debug);
        feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    } else {
        tile_grid = dim3((tile_side + BLOCK_X - 1) / BLOCK_X, (tile_side + BLOCK_Y - 1) / BLOCK_Y, patches);
        const int tiles = tile_grid.x * tile_grid.y;
        dim3 block(BLOCK_X, BLOCK_Y, 1);
        num_rendered = P;
        img_chunk_size = required<ImageState>(tile_side * tile_side * patches);
        img_chunkptr = imageBuffer(img_chunk_size);
        imgState = ImageState::fromChunk(img_chunkptr, tile_side * tile_side * patches);
        binning_chunk_size = required<BinningState>(P, dim3(patches, 1, 1));
        binning_chunkptr = binningBuffer(binning_chunk_size);
        binningState = BinningState::fromChunk(binning_chunkptr, P, dim3(patches, 1, 1));
        initPointIdKernel<<<(P + 255) / 256, 256>>>(binningState.point_list, P);
        std::vector<uint2> host_ranges(patches);
        std::vector<int> h_offsets(patches + 1);
        CHECK_CUDA(cudaMemcpy(h_offsets.data(), point_offsets,
                              (patches + 1) * sizeof(int), cudaMemcpyDeviceToHost), debug);
        for (int i = 0; i < patches; ++i)
            host_ranges[i] = make_uint2(h_offsets[i], h_offsets[i + 1]);
        CHECK_CUDA(cudaMemcpy(binningState.ranges,
                              host_ranges.data(),
                              patches * sizeof(uint2),
                              cudaMemcpyHostToDevice), debug);
        feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    }

    const int width0 = (tile_side == 0) ? width : tile_side;
	const int height0 = (tile_side == 0) ? height : tile_side;

    CHECK_CUDA(FORWARD::render(
            tile_grid, block,
            binningState.ranges,
            binningState.point_list,
            width0, height0, tile_side,
            geomState.means2D,
            feature_ptr,
            geomState.conic_opacity,
            geomState.conic0_correction,
            imgState.accum_weight,
            background,
            out_color
            ), debug)

    return num_rendered;
}

void CudaRasterizer::Rasterizer::backward(
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
	bool debug)
{
    dim3 tile_grid;

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	ImageState imgState;
	BinningState binningState;
	if (tile_side == 0) {
	    tile_grid = dim3((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	    imgState = ImageState::fromChunk(img_buffer, width * height);
	    binningState = BinningState::fromChunk(binning_buffer, R, tile_grid);
	} else {
	    tile_grid = dim3((tile_side + BLOCK_X - 1) / BLOCK_X, (tile_side + BLOCK_Y - 1) / BLOCK_Y, patches);
	    imgState = ImageState::fromChunk(img_buffer, tile_side * tile_side * patches);
	    binningState = BinningState::fromChunk(binning_buffer, R, dim3(patches, 1, 1));
	}

	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;

	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		binningState.ranges,
		binningState.point_list,
		width, height, tile_side,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.conic0_correction,
		color_ptr,
		imgState.accum_weight,
		out_color,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor),
		debug)
    const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    for (size_t i = 0; i < patches; ++i) {

        float focal_y = height / (2.0f * tan_fovy[i]);
        float focal_x = width  / (2.0f * tan_fovx[i]);

        int h_offsets[2];
        cudaMemcpy(&h_offsets[0], point_offsets + i, 2 * sizeof(int), cudaMemcpyDeviceToHost);

        int offset0 = h_offsets[0];
        int offset1 = h_offsets[1];


        CHECK_CUDA(BACKWARD::preprocess(P, D, M, offset0, offset1,
            (float3*)means3D,
            geomState.internal_radii,
            shs,
            geomState.clamped,
            (glm::vec3*)scales,
            (glm::vec4*)rotations,
            scale_modifier,
            cov3D_ptr,
            viewmatrix[i],
            projmatrix[i],
            focal_x, focal_y,
            tan_fovx[i], tan_fovy[i],
            (glm::vec3*)campos[i],
            (float3*)dL_dmean2D,
            dL_dconic,
            (glm::vec3*)dL_dmean3D,
            dL_dcolor,
            dL_dcov3D,
            dL_dsh,
            (glm::vec3*)dL_dscale,
            (glm::vec4*)dL_drot),
            debug)
    }
}
