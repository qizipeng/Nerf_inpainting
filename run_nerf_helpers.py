import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image



# Misc
def img2mse(x,y,mask = None):
    if mask==None:
        return torch.mean((x - y) ** 2)
    else:
        # print(torch.sum(x * mask), torch.sum(y * mask))
        # print(torch.sum((x * mask - y * mask) ** 2) / torch.sum(mask) *3.)
        # mask = mask[...,0].unsqueeze(-1)
        # print('loss2:')
        # print(torch.sum(x * mask), torch.sum(y * mask))
        # print(torch.sum((x * mask - y * mask) ** 2) / torch.sum(mask))

        return torch.sum((x * mask - y * mask) ** 2) / torch.sum(mask)

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


light_h = 16
light_w = 2 * light_h

def _warn_degree(angles):
    if (np.abs(angles) > 2 * np.pi).any():
        print(
            "Some input value falls outside [-2pi, 2pi]. You sure inputs are "
            "in radians")


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Degrees?
    _warn_degree(pts_sph[:, 1:])

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas






# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model

class Render(nn.Module):
    def __init__(self, light_h):
        super(Render, self).__init__()
        self.light_h = light_h
        self.lights = self.gen_light(self.light_h)

        self.autoexpo_params = nn.Parameter(torch.Tensor([0.5, 0.]))

        # print('light sum:', torch.sum(self.lights))


        # lighs = self.lights.reshape(16,32,3)
        # lighs = lighs.detach().cpu().numpy() *255
        # lighs = Image.fromarray(np.uint8(lighs))
        # lighs.save("light4.png")

        # self.lights = self.lights.reshape(-1, 3)
        # self.lights = torch.tile(self.lights.unsqueeze(0), (1024, 1, 1))

    def _clip_0to1_warn(self, tensor_0to1):
        """Enforces [0, 1] on a tensor/array that should be already [0, 1].
        """
        msg = "Some values outside [0, 1], so clipping happened"
        if isinstance(tensor_0to1, torch.Tensor):
            if torch.min(tensor_0to1) < 0 or torch.max(tensor_0to1) > 1:
                print(msg)
                tensor_0to1 = torch.clip(
                    tensor_0to1, min=0, max=1)
        else:
            if tensor_0to1.min() < 0 or tensor_0to1.max() > 1:
                print(msg)
                tensor_0to1 = np.clip(tensor_0to1, 0, 1)
        return tensor_0to1

    def linear2srgb(self, tensor_0to1):
        if isinstance(tensor_0to1, torch.Tensor):
            pow_func = torch.pow
            where_func = torch.where
        else:
            pow_func = np.power
            where_func = np.where

        srgb_linear_thres = 0.0031308
        srgb_linear_coeff = 12.92
        srgb_exponential_coeff = 1.055
        srgb_exponent = 2.4

        # tensor_0to1 = self._clip_0to1_warn(tensor_0to1)

        tensor_linear = tensor_0to1 * srgb_linear_coeff
        tensor_nonlinear = srgb_exponential_coeff * (
            pow_func(tensor_0to1 + 1e-5, 1 / srgb_exponent)
        ) - (srgb_exponential_coeff - 1)

        is_linear = tensor_0to1 <= srgb_linear_thres
        tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

        return tensor_srgb



    def gen_light(self, light_h):
        light_shape = [light_h, 2 * light_h, 3]
        maxv = 1
        light = maxv * torch.rand(light_h, 2 * light_h, 3)
        light = light.reshape(-1,3)  ###512 * 3
        # light = torch.tile(light.unsqueeze(0), (1024, 1, 1))
        # light = Variable(light, requires_grad=True)
        light = nn.Parameter(light)
        return light
        # No negative light
        # return torch.clip(light, 0., np.inf)  # 3D

    def forward(self, normal_map, albedo_pred, visit_pred, lights_xyz, areas, pts):
        # print('light sum:', torch.sum(self.lights))
        # print(self.lights.shape, self.lights.is_leaf, self.lights.requires_grad)
        # print(light_model.lights.shape)
        # print("lighting_dirs:",self.lights[0,:])
        # print("solid angle:", areas[0,:])
        # lighs = self.lights.reshape(16,32,3)
        # lighs = lighs.detach().cpu().numpy() *255
        # # light = 100 * torch.rand(light_h, 2 * light_h, 3)
        # # lighs = light.detach().cpu().numpy() * 255
        # lighs = Image.fromarray(np.uint8(lighs))
        # lighs.save("light6.png")

        new_lights = torch.zeros(16,32,3).cuda()
        new_lights[7:9,15:17,...] = 10
        # new_lights[0:3, 8:10, ...] = 10
        new_lights = new_lights.reshape(-1,3)






        ###pts N * 3
        ###lights_xyz = L *3
        env_lights = torch.clip(self.lights.data, 0, np.inf)

        env_lights = new_lights

        in_dir = torch.tile(torch.unsqueeze(lights_xyz, 0), (pts.shape[0],1,1)) - torch.tile(torch.unsqueeze(pts, 1),(1, lights_xyz.shape[0], 1))  ## N L 3
        cosin = torch.einsum('ijk,ik->ij', in_dir, normal_map)###n * m  1024* 512

        front_lit = torch.where(cosin>0, cosin, cosin * 0)

        light_visit = front_lit * visit_pred     ####1024 * 512

        lights = light_visit[:,:,None] * env_lights[None,:,:] ### N L 3


        # print(lights.shape)

        # print(albedo_pred.shape, lights.shape, areas.shape, cosin.shape)

        rgb_light = albedo_pred.unsqueeze(1) * lights * cosin.unsqueeze(-1) * areas.unsqueeze(0)

        # light = torch.tile(light.unsqueeze(0), (160000, 1, 1))
        # shade = torch.einsum('ij,jk->ijk', cosin_visit ,(self.lights  * areas.detach()))     ##512* 3 1024*512 512 *1
        rgb_light = torch.sum(rgb_light, dim = 1)  ###1024 * 512 * 3

        # rgb_light = shade * albedo_pred

        rgb_light = torch.clip(rgb_light, 0, 1)

        # print(torch.sum(torch.isnan(rgb_light)))
        # print(rgb_light)
        rgb_light = self.linear2srgb(rgb_light)
        # print(rgb_light)

        # print(torch.sum(torch.isnan(rgb_light)))

        # def linear2srgb(tensor_0to1):
        #     if isinstance(tensor_0to1, tf.Tensor):
        #         pow_func = tf.math.pow
        #         where_func = tf.where
        #     else:
        #         pow_func = np.power
        #         where_func = np.where
        #
        #     srgb_linear_thres = 0.0031308
        #     srgb_linear_coeff = 12.92
        #     srgb_exponential_coeff = 1.055
        #     srgb_exponent = 2.4
        #
        #     tensor_0to1 = _clip_0to1_warn(tensor_0to1)
        #
        #     tensor_linear = tensor_0to1 * srgb_linear_coeff
        #     tensor_nonlinear = srgb_exponential_coeff * (
        #         pow_func(tensor_0to1, 1 / srgb_exponent)
        #     ) - (srgb_exponential_coeff - 1)
        #
        #     is_linear = tensor_0to1 <= srgb_linear_thres
        #     tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)
        #
        #     return tensor_srgb


        # lights = torch.sum(lights, dim = 1)
        #
        # rgb_light = torch.cat([rgb_light, lights], dim = -1)

        autoexpo = self.autoexpo_params
        scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
        shift = autoexpo[1]

        rgb_light = (rgb_light - shift) / scale

        rgb_light = torch.clip(rgb_light, 0, 1)



        return rgb_light



class Visit(nn.Module):
    def __init__(self, D=8, W=256, input_ch=128, output_ch=3, skips=[4]):
        super(Visit, self).__init__()
        self.skips = skips
        self.visit_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        self.last_linear = nn.Linear(W, output_ch)
        self.sigmod = torch.nn.Sigmoid()
    def forward(self, features):
        h = features
        for i, l in enumerate(self.visit_linears):
            h = self.visit_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([features, h], -1)

        Visit = self.last_linear(h)
        Visit = self.sigmod(Visit)
        return Visit

class Normal(nn.Module):
    def __init__(self, D=8, W=256, input_ch=128, output_ch=3, skips=[4]):
        super(Normal, self).__init__()
        self.skips = skips
        self.normal_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        self.last_linear = nn.Linear(W, output_ch)
        # self.sigmod = torch.nn.Sigmoid()
    def forward(self, features):
        h = features
        for i, l in enumerate(self.normal_linears):
            h = self.normal_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([features, h], -1)

        normal = self.last_linear(h)
        # normal = self.sigmod(normal)
        return normal


class Albedo(nn.Module):
    def __init__(self, D=8, W=256, input_ch=128, output_ch=3, skips=[4]):
        super(Albedo, self).__init__()
        self.skips = skips
        self.albedo_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        self.last_linear = nn.Linear(W, output_ch)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, features):
        h = features
        for i, l in enumerate(self.albedo_linears):
            h = self.albedo_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([features, h], -1)

        albedo = self.last_linear(h)
        albedo = self.sigmod(albedo)
        return albedo



class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        # surface_features  = h
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            surface_features = feature
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)

            # input_pts.requires_grad_(True)
            # print(alpha.requires_grad, input_pts.requires_grad)
            #
            # normal_map = torch.autograd.grad(
            #     outputs=alpha,
            #     inputs=h,
            #     grad_outputs=torch.ones_like(alpha, requires_grad=False),
            #     retain_graph=True,
            #     create_graph=True,
            # )[0]
            # print('normal shape', normal_map.shape)

        else:
            outputs = self.output_linear(h)

        # return outputs
        return {'outputs':outputs,
                 'features': surface_features}

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


if __name__ =="__main__":
    light_h = 16
    light(light_h)
