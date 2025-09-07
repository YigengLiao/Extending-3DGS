import math
import torch
from scene.gauss_cam_throughput import gauss_cam_transmittance
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def render_frame(cam, model, bg_color, is_test_dataset):

    tanfovx = []
    tanfovy = []
    viewmatrix = []
    projmatrix = []
    campos = []
    crop_box = []

    for c in cam:

        tanfovx.append(math.tan(c.FoVx * 0.5))
        tanfovy.append(math.tan(c.FoVy * 0.5))
        viewmatrix.append(c.world_view_transform)
        projmatrix.append(c.full_proj_transform)
        campos.append(c.camera_center)
        crop_box.append(torch.Tensor(c.crop_box).cuda())

    if is_test_dataset:
        tile_side = 0
    else:
        tile_side = int(cam[0].tile_side)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam[0].image_height),
        image_width=int(cam[0].image_width),
        tile_side=tile_side,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        crop_box=crop_box,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    u_gauss, material, point_id, xyz, v_dir, point_offsets, geomBuffer = rasterizer(
        means3D = model.get_xyz.detach(),
        opacities = model.get_opacity.detach(),
        u_gauss = model.get_u_gauss,
        material = model.get_material,
        rotations = model.get_rotation.detach(),
        shs = None,
        colors_precomp = torch.Tensor([0.]),
        scales = model.get_scaling.detach(),
        cov3D_precomp = None)
    v_dir = v_dir * model.point_scale
    v_dir_norm = v_dir / (v_dir.norm(dim=-1, keepdim=True) + 1e-12)
    d_ref = (v_dir_norm + 1) / 2

    xyz_input = (xyz - model.point_weiyi) * model.point_scale + 0.05
    xyz_feat = model.get_hashgrid_xyz(xyz_input)

    in_feat = torch.cat([d_ref, xyz_feat, material], dim=-1)
    raw_rgb = model.get_mlp_color(in_feat)
    rgb = torch.sigmoid(raw_rgb).type_as(xyz)

    transmission = gauss_cam_transmittance(xyz_input, v_dir, u_gauss, sigma=0.03, net=model.get_mlp_density, hash_feature=model.get_hashgrid_density)

    rendered_image = rasterizer.render(
        means3D=model.get_xyz,
        point_offsets=point_offsets,
        geomBuffer=geomBuffer,
        point_id=point_id,
        opacities = model.get_opacity,
        transmission = transmission,
        rotations = model.get_rotation,
        shs=None,
        colors_precomp=rgb,
        scales = model.get_scaling,
        cov3D_precomp = None)

    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
    }

    return out, point_offsets