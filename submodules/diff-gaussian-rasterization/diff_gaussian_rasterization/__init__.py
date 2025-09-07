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

from typing import NamedTuple, List
import torch.nn as nn
import torch
from cumm.dtypes import float32
from openpyxl.styles.builtins import output
from torch import dtype

from . import _C

def cpu_deep_copy_tuple(input_tuple):

    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)

def rasterize_gaussians_preprocess(
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        u_gauss,
        material
):

    return _RasterizeGaussians_preprocess.apply(
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        u_gauss,
        material
    )

def rasterize_gaussians_render(
        means3D,
        point_offsets,
        geomBuffer,
        colors_precomp,
        transmission,
        raster_settings,
        point_id,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        sh,
    ):

    return _RasterizeGaussians_render.apply(
        means3D,
        point_offsets,
        geomBuffer,
        colors_precomp,
        transmission,
        raster_settings,
        point_id,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        sh,
    )

class _RasterizeGaussians_preprocess(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
            u_gauss,
            material
    ):
        opacities = torch.sigmoid(opacities)
        scales = torch.exp(scales)
        rotations = torch.nn.functional.normalize(rotations)
        # 将参数按照 C++ 库的调用约定重新组织
        args = (
            means3D,  # 3D 均值
            colors_precomp,  # 预计算颜色
            opacities,  # 不透明度
            scales,  # 缩放因子
            rotations,  # 旋转矩阵
            raster_settings.scale_modifier,  # 缩放修正
            cov3Ds_precomp,  # 预计算 3D 协方差
            raster_settings.viewmatrix,  # 视图矩阵
            raster_settings.projmatrix,  # 投影矩阵
            raster_settings.tanfovx,  # X 方向 FOV 的切线
            raster_settings.tanfovy,  # Y 方向 FOV 的切线
            raster_settings.image_height,  # 图像高度
            raster_settings.image_width,  # 图像宽度
            raster_settings.tile_side,
            sh,  # 球面谐波
            raster_settings.sh_degree,  # SH 展开度
            raster_settings.campos,  # 相机位置
            raster_settings.crop_box,  # 是否开启多图像块渲染训练阶段
            raster_settings.prefiltered,  # 是否预滤波
            raster_settings.debug  # 调试标志
        )
        # 调用 C++/CUDA 光栅化函数
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # 在调试模式下先复制参数以防止被破坏
            try:
                point_id, point_offsets, v_dir, geomBuffer = _C.rasterize_gaussians_preprocess(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")  # 保存快照供调试
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            point_id, point_offsets, v_dir, geomBuffer = _C.rasterize_gaussians_preprocess(*args)

        means3D_out = means3D[point_id]
        u_gauss_out = u_gauss[point_id]
        material_out = material[point_id]

        # 保存用于反向传播的张量和设置
        ctx.save_for_backward(point_id)
        ctx.u_gauss_shape = u_gauss.shape
        ctx.material_shape = material.shape

        return u_gauss_out, material_out, point_id, means3D_out, v_dir.view(-1,3), point_offsets, geomBuffer

    @staticmethod
    def backward(ctx, grad_u_gauss_out, grad_material_out, *unused):

        point_id, = ctx.saved_tensors
        u_gauss_shape = ctx.u_gauss_shape
        material_shape = ctx.material_shape
        point_id = point_id.unsqueeze(0)
        grad_u_gauss = torch.sparse_coo_tensor(point_id, grad_u_gauss_out, size=u_gauss_shape).coalesce()
        grad_material = torch.sparse_coo_tensor(point_id, grad_material_out, size=material_shape).coalesce()

        return None, None, None, None, None, None, None, None, grad_u_gauss, grad_material

class _RasterizeGaussians_render(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            point_offsets,
            geomBuffer,
            colors_precomp,
            transmission,
            raster_settings,
            point_id,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            sh,
    ):


        ctx.means3D_shape = means3D.shape
        ctx.opacities_shape = opacities.shape
        ctx.scales_shape = scales.shape
        ctx.rotations_shape = rotations.shape

        means3D = means3D[point_id]
        opacities = torch.sigmoid(opacities[point_id])
        scales = torch.exp(scales[point_id])
        rotations = torch.nn.functional.normalize(rotations[point_id])
        pt_count = means3D.shape[0]
        args = (
            pt_count,
            point_offsets,
            geomBuffer,
            raster_settings.bg,  # 背景色
            colors_precomp,  # 预计算颜色
            transmission,
            raster_settings.image_height,  # 图像高度
            raster_settings.image_width,  # 图像宽度
            raster_settings.tile_side,
            raster_settings.debug  # 调试标志
        )

        # 调用 C++/CUDA 光栅化函数
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # 在调试模式下先复制参数以防止被破坏
            try:
                num_rendered, color, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_render(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")  # 保存快照供调试
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, out_color, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians_render(*args)

        # 保存用于反向传播的张量和设置
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp, means3D, opacities, transmission, scales, rotations, point_offsets, point_id,
            cov3Ds_precomp, sh,
            geomBuffer, binningBuffer, imgBuffer, out_color
        )
        return out_color

    @staticmethod
    def backward(ctx, grad_out_color):

        means3D_shape = ctx.means3D_shape
        opacities_shape = ctx.opacities_shape
        scales_shape = ctx.scales_shape
        rotations_shape = ctx.rotations_shape
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, opacities, transmission, scales, rotations, point_offsets, point_id, cov3Ds_precomp, sh,\
        geomBuffer, binningBuffer, imgBuffer, out_color = ctx.saved_tensors  # 恢复保存的张量

        # 按照 C++ 接口重新组织参数
        args = (
            point_offsets,
            raster_settings.bg,
            means3D,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,  # 图像高度
            raster_settings.image_width,  # 图像宽度
            raster_settings.tile_side,
            grad_out_color,  # 上游梯度
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            out_color,
            raster_settings.debug
        )

        # 调用 C++ 反向传播函数计算梯度
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # 复制参数
            try:
                grad_means2D, grad_colors_precomp, grad_opacities0, grad_means3D0,\
                grad_cov3Ds_precomp, grad_sh, grad_scales0, grad_rotations0 = \
                    _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")  # 保存快照
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities0, grad_means3D0,\
            grad_cov3Ds_precomp, grad_sh, grad_scales0, grad_rotations0 = \
                _C.rasterize_gaussians_backward(*args)
        grad_transmission = grad_opacities0 * opacities
        grad_opacities0 = grad_opacities0 * transmission

        grad_opacities0 = grad_opacities0 * opacities * (1 - opacities)
        grad_scales0 = grad_scales0 * scales

        rotations_norm = rotations.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        dot = (grad_rotations0 * rotations).sum(dim=-1, keepdim=True)
        grad_rotations0 = (grad_rotations0 - dot * rotations) / rotations_norm

        point_id = point_id.unsqueeze(0)
        grad_means3D = torch.sparse_coo_tensor(point_id, grad_means3D0, size=means3D_shape).coalesce()
        grad_opacities = torch.sparse_coo_tensor(point_id, grad_opacities0, size=opacities_shape).coalesce()
        grad_scales = torch.sparse_coo_tensor(point_id, grad_scales0, size=scales_shape).coalesce()
        grad_rotations = torch.sparse_coo_tensor(point_id, grad_rotations0, size=rotations_shape).coalesce()

        return grad_means3D, None, None, grad_colors_precomp, grad_transmission, None, None, grad_opacities, grad_scales, grad_rotations, \
            grad_cov3Ds_precomp, grad_sh

class GaussianRasterizationSettings(NamedTuple):
    image_height: int  # 图像高度
    image_width: int   # 图像宽度
    tile_side: int
    tanfovx : List[float]    # X 方向视场角切线值
    tanfovy : List[float]    # Y 方向视场角切线值
    bg : torch.Tensor  # 背景颜色张量
    scale_modifier : float  # 全局缩放修正因子
    viewmatrix : List[torch.Tensor]  # 视图矩阵张量
    projmatrix : List[torch.Tensor]  # 投影矩阵张量
    sh_degree : int    # 球面谐波展开度
    campos : List[torch.Tensor]  # 相机位置张量
    crop_box : torch.Tensor  # 图像块坐标
    prefiltered : bool  # 是否使用预滤波
    debug : bool        # 调试模式开关

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings  # 保存光栅化设置

    def forward(self, means3D, opacities, u_gauss, material, rotations = None,
                shs = None, colors_precomp = None, scales = None, cov3D_precomp = None):

        raster_settings = self.raster_settings  # 获取设置
        # 校验参数，SHs 与预计算颜色只能提供其一
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        # 校验 scale/rotation 与 3D 协方差之一
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
                ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 为空时初始化为零维张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # 调用核心光栅化函数，并返回渲染结果
        return rasterize_gaussians_preprocess(
            means3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            u_gauss,
            material
        )
    def render(self, means3D, point_offsets, geomBuffer, point_id, opacities, transmission, rotations = None,
                shs = None, colors_precomp = None, scales = None, cov3D_precomp = None):

        raster_settings = self.raster_settings  # 获取设置

        # 为空时初始化为零维张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # 调用核心光栅化函数，并返回渲染结果
        return rasterize_gaussians_render(
            means3D,
            point_offsets,
            geomBuffer,
            colors_precomp,
            transmission,
            raster_settings,
            point_id,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            shs,
        )
