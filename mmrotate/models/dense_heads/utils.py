# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import convex_iou
import numpy as np
import cv2


def points_center_pts(RPoints, y_first=True):
    """Compute center point of Pointsets.

    Args:
        RPoints (torch.Tensor): the  lists of Pointsets, shape (k, 18).
        y_first (bool, optional): if True, the sequence of Pointsets is (y,x).

    Returns:
        center_pts (torch.Tensor): the mean_center coordination of Pointsets,
            shape (k, 18).
    """
    RPoints = RPoints.reshape(-1, 9, 2)

    if y_first:
        pts_dy = RPoints[:, :, 0::2]
        pts_dx = RPoints[:, :, 1::2]
    else:
        pts_dx = RPoints[:, :, 0::2]
        pts_dy = RPoints[:, :, 1::2]
    pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
    pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
    center_pts = torch.cat([pts_dx_mean, pts_dy_mean], dim=1).reshape(-1, 2)
    return center_pts


def convex_overlaps(gt_bboxes, points):
    """Compute overlaps between polygons and points.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
        points (torch.Tensor): Points to be assigned, shape(n, 18).

    Returns:
        overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
    """
    overlaps = convex_iou(points, gt_bboxes)
    overlaps = overlaps.transpose(1, 0)
    return overlaps


def levels_to_images(mlvl_tensor, flatten=False):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
        flatten (bool, optional): if shape of mlvl_tensor is (N, C, H, W)
            set False, if shape of mlvl_tensor is  (N, H, W, C) set True.

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    if flatten:
        channels = mlvl_tensor[0].size(-1)
    else:
        channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        if not flatten:
            t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


def get_num_level_anchors_inside(num_level_anchors, inside_flags):
    """Get number of every level anchors inside.

    Args:
        num_level_anchors (List[int]): List of number of every level's anchors.
        inside_flags (torch.Tensor): Flags of all anchors.

    Returns:
        List[int]: List of number of inside anchors.
    """
    split_inside_flags = torch.split(inside_flags, num_level_anchors)
    num_level_anchors_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_anchors_inside

class DistanceMap:
    """
    A class to generate and manage distance or Gaussian maps centered at arbitrary positions.
    Args:
        h (int): Height of the output map.
        w (int): Width of the output map.
        dist (str, optional): Type of distance metric to use. Either 'l2' for squared L2 distance or 'gaussian' for a Gaussian kernel. Default is 'l2'.
        sigma (float, optional): Standard deviation for the Gaussian kernel. Default is 4096.
        normalize (bool, optional): Whether to normalize the output map. For 'l2', applies sqrt and negation; for 'gaussian', normalizes by the Gaussian constant. Default is False.
        device (str or torch.device, optional): Device to store tensors on. Default is 'cpu'.
    Attributes:
        h (int): Height of the output map.
        w (int): Width of the output map.
        device (str or torch.device): Device used for computation.
        _sigma (float): Standard deviation for the Gaussian kernel.
        template_h (int): Height of the internal template (2 * h).
        template_w (int): Width of the internal template (2 * w).
        center_y (int): Y-coordinate of the template center.
        center_x (int): X-coordinate of the template center.
        template (torch.Tensor): Precomputed template map.
    Methods:
        get_gaussian_at(mu):
            Returns a cropped map centered at the given coordinates.
            Args:
                mu (tuple or list or torch.Tensor): (x, y) coordinates of the center.
            Returns:
                torch.Tensor: Cropped map of shape (h, w) centered at mu.
        get_gaussian_batch(mus):
            Returns a batch of cropped maps centered at the given coordinates.
            Args:
                mus (torch.Tensor): Tensor of shape (N, 2) containing (x, y) coordinates for N centers.
            Returns:
                torch.Tensor: Batch of cropped maps of shape (N, h, w).
    """
    
    # Cache for instances to avoid recomputing templates for the same parameters
    _instances = {}

    def __init__(self, size, dist='gaussian', sigma=4096, normalize=False, device='cuda'):
        assert dist in ['gaussian', 'l2'], "dist must be either 'gaussian' or 'l2'"
        self.h = size[0]
        self.w = size[1]
        self.device = device

        self._sigma = sigma
        
        self.template_h = (size[0] * 2 + 4) # to ensure the template is large enough to cover the area around the center
        self.template_w = (size[1] * 2 + 4) # to ensure the template is large enough to cover the area around the center
        self.center_y = self.template_h // 2
        self.center_x = self.template_w // 2
        
        self._compute_template(dist, normalize)

    @classmethod
    def get_instance(cls, size, dist='gaussian', sigma=4096, normalize=False, device='cuda'):
        key = (size, dist, sigma, normalize, str(device))
        if key not in cls._instances:
            cls._instances[key] = cls(size, dist, sigma, normalize, device)
        return cls._instances[key]
    
    def _compute_template(self, dist, normalize):
        y = torch.arange(self.template_h, dtype=torch.float32, device=self.device)
        x = torch.arange(self.template_w, dtype=torch.float32, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        xy = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
        mu = torch.tensor([self.center_y, self.center_x], dtype=torch.float32, device=self.device).reshape(1, 2)
        
        self.template = self._compute_distance(xy, mu, sigma=self._sigma, dist=dist, normalize=normalize)
        self.template = self.template.reshape(self.template_h, self.template_w)
    
    def _compute_distance(self, xy, mu, sigma, dist='l2', normalize=False):
        dxy = xy - mu
        if dist == 'l2':
            #d = -(dxy**2).sum(dim=-1)
            d = -(dxy**2).sum(dim=-1)
            if normalize:
                d = -d.abs().sqrt()
        elif dist == 'gaussian':
            #sigma = torch.tensor([[sigma, 0], [0, sigma]], dtype=torch.float32, device=self.device)
            #d = torch.exp(-0.5 * dxy.unsqueeze(-1).permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy.unsqueeze(-1))).squeeze(-1).squeeze(-1))
            d = torch.exp(-0.5 / sigma * (dxy**2).sum(dim=-1))
            if normalize:
                #d = d / (2 * torch.pi * sigma.det().clamp(1e-7).sqrt())
                d = d / (2 * torch.pi * sigma)
        else:
            raise ValueError("dist must be either 'l2' or 'gaussian'")
        return d
    
    
    def compute_map(self, mu):
        x, y = int(mu[0]), int(mu[1])
        
        template_y_start = self.center_y - y
        template_y_end = template_y_start + self.h
        template_x_start = self.center_x - x
        template_x_end = template_x_start + self.w
        
        gaussian = self.template[template_y_start:template_y_end, template_x_start:template_x_end]
        
        return gaussian
    
    def compute_map_batch(self, mus):
        N = len(mus)
        
        template_y_start = self.center_y - mus[:, 1].long()  # [N, ]
        template_x_start = self.center_x - mus[:, 0].long()  # [N, ]
        
        batch_idx = torch.arange(N, device=self.device).view(-1, 1, 1)         # [N, 1, 1]
        result_y_idx = torch.arange(self.h, device=self.device).view(1, -1, 1) # [1, h, 1]
        result_x_idx = torch.arange(self.w, device=self.device).view(1, 1, -1) # [1, 1, w]
        
        template_y_idx = template_y_start.view(-1, 1, 1) + result_y_idx  # [N, h, 1]
        template_x_idx = template_x_start.view(-1, 1, 1) + result_x_idx  # [N, 1, w]
        
        template_y_idx = template_y_idx.expand(-1, -1, self.w) # [N, h, w]
        template_x_idx = template_x_idx.expand(-1, self.h, -1) # [N, h, w]
        
        result = torch.zeros(N, self.h, self.w, device=self.device)
        result[batch_idx, result_y_idx, result_x_idx] = self.template[template_y_idx, template_x_idx]
        
        return result

def watershed_segmentation(image, markers, use_cuda_ops=False):
    """Perform watershed segmentation on an image using markers.
    Args:
        image (torch.Tensor or np.ndarray): Input image of shape (H, W, 3), uint8 dtype.
               tensor if use_cuda_ops is True, numpy array otherwise. 
        markers (torch.Tensor): Markers for watershed segmentation, shape (H, W),  int32 dtype
        use_cuda_ops (bool): Whether to use CUDA operations for segmentation.
    Returns:
        torch.Tensor: Segmentation map of shape (H, W), int64 dtype"""
    device = markers.device

    if use_cuda_ops:
        raise NotImplementedError( "CUDA operations for watershed segmentation are not implemented yet.")
        # TODO: Add CUDA implementation
        # the nppiSegmentWatershed_8u_C1IR function from NVIDIA's NPP library can not 
        # 1）OpenCV does not natively provide a CUDA-accelerated implementation of the watershed algorithm.
        # 2）The nppiSegmentWatershed_8u_C1IR function from NVIDIA's NPP library does not support marker-based segmentation. 
        #   It relies on the gradient information of the image itself for segmentation, making it currently infeasible to meet the requirements here.
    else:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy().astype(np.uint8)
        markers = markers.detach().cpu().numpy().astype(np.int32)
        segs = torch.tensor(cv2.watershed(image, markers), device=device, dtype=torch.int64)

    return segs