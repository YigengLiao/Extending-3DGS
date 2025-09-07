import torch

def clipped_point_to_camera_vectors(points, v_dir):

    inv_d = 1.0 / v_dir

    t1 = (.0 - points) * inv_d
    t2 = (1.0 - points) * inv_d

    t1 = torch.nan_to_num(t1, nan=-torch.inf)
    t2 = torch.nan_to_num(t2, nan=torch.inf)

    tmin = torch.minimum(t1, t2)
    tmax = torch.maximum(t1, t2)

    t_enter = tmin.max(dim=1).values
    t_exit = tmax.min(dim=1).values

    t0 = t_enter.clamp(0.0, 1.0)
    t1 = t_exit.clamp(0.0, 1.0)

    lam = (t1 - t0).clamp(min=0.0)

    keep_mask = lam > 0
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    d_kept = v_dir[keep_mask]
    lam_kept = lam[keep_mask]
    vectors = d_kept * lam_kept.unsqueeze(1)

    t0_kept = t0[keep_mask]
    start_pts = points[keep_mask] + t0_kept.unsqueeze(1) * d_kept

    return start_pts, vectors, keep_idx

def gauss_cam_transmittance(gauss_xyz, v_dir, u_gauss, sigma, net, hash_feature):

    device = gauss_xyz.device
    start_pts, v_dir, keep_idx = clipped_point_to_camera_vectors(gauss_xyz, v_dir)

    v_dir_length = v_dir.norm(dim=-1)
    m = v_dir_length / sigma
    n = torch.floor(m).long()
    segments = n + 1
    s_factor = m / segments

    offsets = torch.empty_like(segments)
    torch.cumsum(segments, dim=0, out=offsets)
    offsets = torch.cat([segments.new_zeros(1), offsets[:-1]])

    total_pts = int(segments.sum())

    row_id = torch.repeat_interleave(
        torch.arange(segments.size(0), device=device), segments
    )

    local_idx = torch.arange(total_pts, device=device) - offsets[row_id]

    mid_t = (local_idx.float() + 0.5) / segments[row_id].float()

    gauss_rep = start_pts[row_id]
    v_dir_rep = v_dir[row_id]
    points = gauss_rep + mid_t.unsqueeze(1) * v_dir_rep

    u_rep = u_gauss[keep_idx[row_id]]
    points_feature = hash_feature(points)
    inputs = torch.cat((points_feature, u_rep), dim=-1)

    a = net(inputs).squeeze(-1)

    rho = torch.nn.functional.sigmoid(a.type_as(gauss_xyz)) ** s_factor[row_id]

    rho_sum = torch.ones_like(gauss_xyz[:,0])
    rho_sum.scatter_reduce_(dim=0,index=keep_idx[row_id],src=rho,reduce='prod')

    return rho_sum.unsqueeze(-1)