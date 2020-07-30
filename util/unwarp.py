import torch
from util.projector import FlatProjector
import util.fisheye as fisheye
import util.image as image

class Unwarper:
    def __init__(self, opt):

        self.fisheye_type = opt.fisheye_type
        self.out_img_size = opt.out_size

        # 3d joint space predefined parameters
        self.near_distance = opt.near_distance
        self.far_distance = opt.far_distance

        #projection camera model parameters (camera faces along the z axis)
        self.max_depth = opt.max_depth
        self.projector = FlatProjector(opt.out_size, opt.space_to_img_ratio)

        # Thresholds
        self.min_depth_thrs = opt.min_depth_thrs
        self.initialized = False

    def initialize(self, input_img_size, is_cuda):

        theta_map, phi_map = fisheye.make_theta_phi_meshgrid(input_img_size, self.fisheye_type)
        theta_map.requires_grad = False
        phi_map.requires_grad = False
        fisheye_mask = image.get_center_circle_mask(input_img_size)
        fisheye_mask.requires_grad = False

        self.fisheye_mask = fisheye_mask
        self.theta_map = theta_map
        self.phi_map = phi_map

        if is_cuda:
            self.fisheye_mask = self.fisheye_mask.to(device='cuda')
            self.theta_map = self.theta_map.to(device='cuda')
            self.phi_map = self.phi_map.to(device='cuda')

        self.is_cuda = is_cuda

        self.initialized = True

    def unwarp(self, fish_depth_img):
        assert len(fish_depth_img.shape) == 4

        self._initialize_if_not(fish_depth_img)

        fish_depth_img = self._preprocess(fish_depth_img)

        n_sample = fish_depth_img.size(0)

        unwarped_imgs = []
        for i in range(n_sample):
            unwarped = self._unwarp_single_img(fish_depth_img[i])
            unwarped_imgs.append(unwarped)

        result = torch.stack(unwarped_imgs, 0)
        del unwarped_imgs # save memory

        return result

    def _initialize_if_not(self, depth_img):
        n_epoch, n_channel, height, width = depth_img.shape
        if not self.initialized:
            input_img_size = (height, width)
            self.initialize(input_img_size, depth_img.is_cuda)

    def _preprocess(self, depth_img):
        depth_img = depth_img * self.fisheye_mask
        return depth_img

    def _unwarp_single_img(self, depth_img):
        depth = depth_img.view(-1) # H, W
        theta = self.theta_map.view(-1)
        phi = self.phi_map.view(-1)

        try: # DEBUG: Error occurs here intermittenly
            valid_index = self._filter_valid_index(depth)

            if self._check_if_img_empty(depth):
                return self._get_empty_img()
            depth = depth[valid_index]
            theta = theta[valid_index]
            phi = phi[valid_index]

            valid_index = self._sort_index_depth_desc_order(depth)
            if self._check_if_img_empty(depth):
                return self._get_empty_img()
            depth = depth[valid_index]
            theta = theta[valid_index]
            phi = phi[valid_index]

        except Exception as e:
            print(valid_index)
            print(depth.shape)
            print(e)
            raise Exception()

        distance = self._scale_reversed_depth_to_distance(depth)

        x, y, z = self._convert_to_cartesian(distance, theta, phi)
        xyz = torch.stack((x,y,z), dim=1)

        uv = self.projector.convert_to_uv(xyz, data_format='NC')
        uv = uv.long()
        """loss should not backprop to uv. because uv is used as index"""
        uv = uv.detach()

        valid_index = self._filter_valid_uv_index(uv)
        uv = uv[valid_index]
        xyz = xyz[valid_index]

        converted_img = self._map_to_img(uv, xyz)
        return converted_img

    def _check_if_img_empty(self, depth_img):
        return len(depth_img.size()) == 0 or depth_img.nelement() == 0

    def _get_empty_img(self):
        out_img_shape = (1, self.out_img_size, self.out_img_size)
        out_img = torch.zeros(out_img_shape)
        if self.is_cuda:
            return out_img.to(device='cuda')

        return out_img

    def _filter_valid_index(self, depth):
        valid_index = (depth > self.min_depth_thrs).nonzero()
        return valid_index.squeeze()

    def _sort_index_depth_desc_order(self, depth):
        """
            Sort x, y, z in descending order by its distance from the camera.
            The result is used for implementing occlusion of point clouds.
            'projection' will only consider a point that is near (which has a larger index) the camera if multiple points are projected to a same point.
        """
        reversed_depth = depth
        sorted_idx = torch.argsort(reversed_depth, descending = True)
        return sorted_idx

    def _convert_to_cartesian(self, distance, theta, phi):
        distance = distance.double()
        theta = theta.double()
        phi = phi.double()

        z = distance * torch.cos(theta)
        x = distance * torch.sin(theta) * torch.cos(phi)
        y = distance * torch.sin(theta) * torch.sin(phi)

        return x, y, z

    def _filter_valid_uv_index(self, uv):
        # select indices that u and v is both in image (0 < u,v < img_size)
        mask_0 = uv >= 0
        mask_1 = uv <= (self.out_img_size - 1)

        mask = mask_0 * mask_1 # and operation
        mask = mask[:, 0] * mask[:, 1]

        valid_index = mask.nonzero()
        return valid_index.squeeze()

    def _map_to_img(self, uv, xyz):
        empty_img = self._get_empty_img()

        u = uv[:,0]
        v = uv[:,1]
        d = xyz[:,2] # z == d, orthogonal projection (Flat Projector)
        #d = torch.norm(xyz[:,:2], dim=1) # z == d, orthogonal projection (Flat Projector)

        d = self._scale_3d_to_grayscale(d)
        empty_img[0, v, u] = d.float()

        return empty_img

    def _scale_reversed_depth_to_distance(self, reversed_depth):
        # depth image follows the inverse depth scale. A nearer point has brighter color.
        depth_01 = (1.0 - reversed_depth)
        return depth_01 * (self.far_distance - self.near_distance) + self.near_distance

    def _scale_3d_to_grayscale(self, val):
        depth_01 = (val - self.near_distance) / (self.far_distance - self.near_distance)
        return (1.0 - depth_01) # convert to the inverse depth scale
