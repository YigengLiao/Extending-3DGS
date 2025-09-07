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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }

    result += 0.5f;

    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}

__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float cov3D[6], const float* viewmatrix)
{

    float3 t = transformPoint4x3(mean, viewmatrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float cov3D[6])
{

    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    glm::vec4 q = rot;
    float r = q.x;
    float x = q.y, y = q.z, z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    glm::mat3 Sigma = glm::transpose(M) * M;

    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const float* crop_box,
    const int W, int H, int tile_side,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* points_xy_image,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    float4* conic0_correction,
    int* point_id,
    float* v_dir,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered,
    int* P_counter)
{

    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);

    float p_w = 1.0f / (p_hom.w + 1e-7f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    float cov3D[6];
    if (cov3D_precomp != nullptr) {
        for (int i = 0; i < 6; ++i)
            cov3D[i] = cov3D_precomp[idx * 6 + i];
    } else {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D);
    }

    float3 cov0 = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    float3 cov = cov0;
    cov.x += 0.1f;
    cov.z += 0.1f;
    float det0 = (cov0.x * cov0.z - cov0.y * cov0.y);
    float det = (cov.x * cov.z - cov.y * cov.y);

    if (det0 == 0.0f || det == 0.0f) return;

    float det_inv = 1.f / det;
    float det_inv0 = 1.f / det0;
    float correction = sqrt(max(0.000025f, det0 * det_inv));
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
    float3 conic0 = { cov0.z * det_inv0, -cov0.y * det_inv0, cov0.x * det_inv0 };

    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    float2 point_image = { ndc2Pix(p_proj.x, W) - crop_box[0], ndc2Pix(p_proj.y, H) - crop_box[1] };

    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

    unsigned mask    = __activemask();
    int leader       = __ffs(mask) - 1;
    int warp_count   = __popc(mask);

    int linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    int lane = linear_tid & 31;

    int warp_base = (lane == leader) ? atomicAdd(P_counter, warp_count) : 0;
    warp_base = __shfl_sync(mask, warp_base, leader);

    unsigned lanemask_lt = (1u << lane) - 1u;
    int local_offset = __popc(mask & lanemask_lt);

    int out_idx = warp_base + local_offset;

    if (colors_precomp == nullptr)
    {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[out_idx * C + 0] = result.x;
        rgb[out_idx * C + 1] = result.y;
        rgb[out_idx * C + 2] = result.z;
    }

    cov3Ds[out_idx * 6 + 0] = cov3D[0];
    cov3Ds[out_idx * 6 + 1] = cov3D[1];
    cov3Ds[out_idx * 6 + 2] = cov3D[2];
    cov3Ds[out_idx * 6 + 3] = cov3D[3];
    cov3Ds[out_idx * 6 + 4] = cov3D[4];
    cov3Ds[out_idx * 6 + 5] = cov3D[5];

    depths[out_idx] = p_view.z;
    radii[out_idx] = my_radius;
    points_xy_image[out_idx] = point_image;
    conic_opacity[out_idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    conic0_correction[out_idx] = { conic0.x, conic0.y, conic0.z, correction };
    tiles_touched[out_idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    point_id[out_idx] = idx;

    v_dir[out_idx * 3 + 0] = cam_pos->x - p_orig.x;
    v_dir[out_idx * 3 + 1] = cam_pos->y - p_orig.y;
    v_dir[out_idx * 3 + 2] = cam_pos->z - p_orig.z;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H, int tile_side,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    const float4* __restrict__ conic0_correction,
    float* __restrict__ final_W,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color)
{
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, (uint32_t)W), min(pix_min.y + BLOCK_Y , (uint32_t)H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };
    bool inside = pix.x < (uint32_t)W && pix.y < (uint32_t)H;
    bool done = !inside;
    uint2 range;
    if (tile_side == 0){
        range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    } else {
        range = ranges[block.group_index().z];
    }

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int   collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float4 collected_conic0_correction[BLOCK_SIZE];

    float sum_w = 0.0f;
    float C[CHANNELS] = { 0 };

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_conic0_correction[block.thread_rank()] = conic0_correction[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {

            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float4 con_o = collected_conic_opacity[j];
            float correction = collected_conic0_correction[j].w;

            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) continue;
            float alpha = con_o.w * exp(power) * correction;
            //  float alpha = min(0.99f, con_o.w * exp(power) * correction);
            if (alpha < 1.0f / 255.0f) continue;
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha;

            sum_w += alpha;
        }
    }
    const int pixel_offset = (tile_side == 0) ? 0 : (block.group_index().z * CHANNELS * H * W);
    const int pixel_offset_W = (tile_side == 0) ? 0 : (block.group_index().z * H * W);
    if (inside)
    {
        if (sum_w > 1.f) {
            const float invw = 1.f / sum_w;
            for (int ch = 0; ch < CHANNELS; ch++)
                out_color[pixel_offset + ch * H * W + pix_id] = C[ch] * invw;
        } else {
            const float fac = 1.f - sum_w;
            for (int ch = 0; ch < CHANNELS; ch++)
                out_color[pixel_offset + ch * H * W + pix_id] = C[ch] + fac * bg_color[ch];
        }
        final_W[pixel_offset_W + pix_id] = sum_w;
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H, int tile_side,
    const float2* means2D,
    const float* colors,
    const float4* conic_opacity,
    const float4* conic0_correction,
    float* final_W,
    const float* bg_color,
    float* out_color)
{
    renderCUDA<NUM_CHANNELS> <<<grid, block>>>(
        ranges,
        point_list,
        W, H, tile_side,
        means2D,
        colors,
        conic_opacity,
        conic0_correction,
        final_W,
        bg_color,
        out_color);
}

void FORWARD::preprocess(int P, int D, int M,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const float* crop_box,
    const int W, int H, int tile_side,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    float4* conic0_correction,
    int* point_id,
    float* v_dir,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered,
    int* P_counter)
{
    preprocessCUDA<NUM_CHANNELS> <<< (P + 255) / 256, 256 >>> (
        P, D, M,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        crop_box,
        W, H, tile_side,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        radii,
        means2D,
        depths,
        cov3Ds,
        rgb,
        conic_opacity,
        conic0_correction,
      	point_id,
      	v_dir,
        grid,
        tiles_touched,
        prefiltered,
      	P_counter
    );
}