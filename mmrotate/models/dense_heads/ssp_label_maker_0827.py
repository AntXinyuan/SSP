# Copyright (c) OpenMMLab. All rights reserved.

import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from ..builder import ROTATED_HEADS
from .rotated_fcos_head import RotatedFCOSHead
from .utils import DistanceMap, watershed_segmentation, mask_rbox_iou, rbox_to_mid_points

from mmrotate.core.evaluation import eval_rbox_single_image
from mmrotate.core.visualization import (
    ssp_visualize_single,
    ssp_load_raw_image,
    ssp_generate_label_single,
    t2n,
)

INF = 1e8

@ROTATED_HEADS.register_module()
class SSPLabelMarkerHead(RotatedFCOSHead):
    """
    A specialized head for point-supervised pseudo-label generation in rotated object detection.
    Inherits from RotatedFCOSHead and extends functionality for handling spatial partitioning,
    region growing, and pseudo-label generation.

    Key Features:
    - Implements spatial partitioning and region growing algorithms for instance segmentation
    - Generates pseudo-labels with various bounding box representations (PCA, min area rectangle)
    - Supports class-specific handling (square objects, overlapping classes, merging classes)
    - Provides visualization capabilities for debugging and analysis

    See the paper(https://arxiv.org/pdf/2506.10601) for more details.
    Paper title: Semantic-decoupled Spatial Partition Guided Point-supervised Oriented Object Detection

    Args:
        cls_square (list): List of class IDs that should be treated as square objects 
            (angles will be set to 0). Default is [1, 9, 11] corresponding to:
            - 1: baseball-diamond
            - 9: storage-tank  
            - 11: roundabout

        cls_overlap (list of lists): Pairs of class IDs that can overlap spatially.
            Default is [[3, 10], [6, 12]] corresponding to:
            - [3-ground-track-field, 10-soccer-ball-field]
            - [6-ship, 12-harbor]

        cls_merge (list of lists): Groups of class IDs that should be merged during processing.
            First group contains classes that will use PCA boxes, second group uses min area rectangles.
            Default is:
            - Group 0: [0,4,5,6,9,13,14] (plane, small/large vehicles, ship, storage-tank, pool, helicopter)
            - Group 1: [1,2,3,7,8,10,11,12] (baseball, bridge, track-field, tennis/basketball courts, soccer, roundabout, harbor)

        cls_stable (list, optional): Class IDs that are considered stable. Default is None.
            If provided, bounding box stabilization will be applied to these classes.

        sp_thres (dict): Threshold configuration for spatial partitioning:
            - default: [pos_thres, neg_thres] (default [0.999, 0.005])
            - override: Tuple of (class_ids, [cls_pos_thres, cls_neg_thres]) for specific classes
            - confidence: Tuple of (low, mid, high) confidence thresholds (default 0.05, 0.6, 0.95)

        is_record_stage (bool): Whether current stage is for recording/visualization.
            If True, enables visualization and pseudo-label generation.

        use_single_scale (bool): Whether to use single scale feature maps instead of FPN.

        *args, **kwargs: Additional arguments passed to parent RotatedFCOSHead.

    Examples of Class IDs Reference in DOTA-v1.0:
        0: plane, 1: baseball-diamond, 2: bridge, 3: ground-track-field,
        4: small-vehicle, 5: large-vehicle, 6: ship, 7: tennis-court,
        8: basketball-court, 9: storage-tank, 10: soccer-ball-field,
        11: roundabout, 12: harbor, 13: swimming-pool, 14: helicopter
    """
    def __init__(self,
                 cls_square=[1, 9, 11],
                 cls_overlap=[[3, 10], [6, 12]], 
                 cls_merge=[[0, 4, 5, 6, 9, 13, 14], [1, 2, 3, 7, 8, 10, 11, 12]],
                 cls_stable=None,
                 sp_thres=dict(
                    default=[0.999, 0.005],
                    override=(([0, 1, 3, 7, 8, 10, 14], [0.995, 0.005]),),
                    confidence=(0.05, 0.6, 0.95)),
                 is_record_stage=False,
                 use_single_scale=False,
                 *args, **kwargs):
        self.cls_square = cls_square
        self.cls_overlap = cls_overlap
        self.cls_merge = cls_merge
        self.cls_stable = cls_stable
        self.use_single_scale = use_single_scale

        super().__init__(*args, **kwargs)

        self.runner_info = {} # assign value through RecordEpochIterHook, containg {'max_epochs': 7, 'max_iters': 22400, 'epoch': 6, 'iter': 38403, 'inner_iter': 3}
        self.store_dir = kwargs.get('train_cfg').get('store_dir', None)
        self.visualize_dir = kwargs.get('train_cfg').get('visualize_dir', None)
        self.pseudo_label_dir = kwargs.get('train_cfg').get('pseudo_label_dir', None)
        self.need_visualize = (self.visualize_dir is not None) and is_record_stage
        self.need_pseudo_label = (self.pseudo_label_dir is not None) and is_record_stage
        self.is_record_stage = is_record_stage

        conf_thres = sp_thres['confidence']
        pos_thres, neg_thres = sp_thres['default']
        pos_thres = [pos_thres,] * self.num_classes
        neg_thres = [neg_thres,] * self.num_classes
        if 'override' in sp_thres.keys():
            for cls_ids, (cls_pos, cls_neg) in sp_thres['override']:
                for cls_id in cls_ids:
                    pos_thres[cls_id] = cls_pos
                    neg_thres[cls_id] = cls_neg
        self.sp_thres = dict(pos=pos_thres, neg=neg_thres, conf=conf_thres)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        # Remove unused conv moudles
        self.reg_convs = self.conv_reg = self.conv_centerness = self.conv_angle = self.scale_angle = self.scales = None

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                angle_preds (list[Tensor]): Box angle for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        feats = list(feats)
        #@# if self.use_single_scale:
        #@#     feats, scales, strides = [feats[0],], [self.scales[0],], [self.strides[0],]
        #@# else:
        #@#     feats, scales, strides = feats, self.scales, self.strides
        #@# cls_score, = multi_apply(self.forward_single, feats, scales, strides)

        feats = [feats[0],] if self.use_single_scale else feats
        cls_score, = multi_apply(self.forward_single, feats)
        return cls_score,

    def forward_single(self, x):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions, angle predictions \
                and centerness predictions of input feature maps.
        """
        cls_feat = x
        #@# reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        #@# for reg_layer in self.reg_convs:
        #@#     reg_feat = reg_layer(reg_feat)
        #@# bbox_pred = self.conv_reg(reg_feat)
        #@# 
        #@# if self.centerness_on_reg:
        #@#     centerness = self.conv_centerness(reg_feat)
        #@# else:
        #@#     centerness = self.conv_centerness(cls_feat)
        #@# # scale the bbox_pred of different level
        #@# # float to avoid overflow when enabling FP16
        #@# bbox_pred = scale(bbox_pred).float()
        #@# if self.norm_on_bbox:
        #@#     # bbox_pred needed for gradient computation has been modified
        #@#     # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
        #@#     # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
        #@#     bbox_pred = bbox_pred.clamp(min=0)
        #@#     if not self.training:
        #@#         bbox_pred *= stride
        #@# else:
        #@#     bbox_pred = bbox_pred.exp()
        #@# angle_pred = self.conv_angle(reg_feat)
        #@# if self.is_scale_angle:
        #@#     angle_pred = self.scale_angle(angle_pred).float()

        return cls_score,

     
    @force_fp32(
        apply_to=('cls_scores',))
    def loss(self,
             cls_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        if self.use_single_scale:
            single_level_points = self.prior_generator.single_level_grid_priors(
                featmap_sizes[0],
                level_idx=0,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)
            all_level_points, cls_scores = [single_level_points,], [cls_scores[0],]
        else:
            all_level_points = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)
        
        loss_mask, extra_results_mask = self.loss_mask(cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas)
            
        return dict(**loss_mask,)

    def loss_mask(self, cls_scores, priors_all_lvl, gt_bboxes, gt_labels, img_metas):
        """ Compute the loss for point-supervised mask learning.
        Args:
            cls_scores (list[Tensor]): List of classification scores for each 
                feature level, each with shape (num_imgs, num_cls, h, w).
            prior_all_lvl (list[Tensor]): List of prior points of each fpn level, 
                each has shape (num_points, 4).
            gt_bboxes (list[Tensor]): Ground truth bounding boxes for each image, 
                each with shape (num_gts, 2/4/5).
            gt_labels (list[Tensor]): Ground truth labels for each image, 
                each with shape (num_gts,).
            img_metas (list[dict]): List of image meta information, each containing 
                'img_shape' key with shape (height, width, 3).
        Returns:
            dict: A dictionary of loss components.
        """
        cls_labels_target, extra_results_list = self.get_targets_mask(
            cls_scores, priors_all_lvl, gt_bboxes, gt_labels, img_metas)
        
        # 1. prepare loss_cls
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_labels = torch.cat(cls_labels_target)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=cls_scores[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        num_avail = torch.tensor(
            len(avail_inds), dtype=torch.float, device=cls_scores[0].device)
        num_avail = max(reduce_mean(num_avail), 1.0)
        
        loss_cls = self.loss_cls(
            flatten_cls_scores[avail_inds], flatten_labels[avail_inds], avg_factor=num_avail)
        
        # 2. collect loss_cls
        factor = 0.0 if self.is_record_stage else 1.0

        return dict(loss_cls=loss_cls*factor), extra_results_list
    
    def get_targets_mask(self, cls_scores, priors_all_lvl, gt_bboxes_list, gt_labels_list, img_metas):
        """Generate targets for point-supervised mask learning.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            priors_all_lvl (list[Tensor]): Prior points for each feature level,
                each has shape (num_points, 4).
            gt_bboxes_list (list[Tensor]): Ground truth bounding boxes for each
                image, each has shape (num_gt, 5).
            gt_labels_list (list[Tensor]): Ground truth labels for each image,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        Returns:
            concat_lvl_labels (list[Tensor]): Concatenated labels for each level, 
            extra_results_list (list[Tensor]): Extra results for each image, 
        """
        concat_points = torch.cat(priors_all_lvl, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in priors_all_lvl]

        def lvl2img(data_list):
            num_imgs = data_list[0].size(0)
            num_lvls = len(data_list)
            data_img_lvl_list = []
            for img_id in range(num_imgs):
                data_img_lvl_list.append([])
                for lvl_id in range(num_lvls):
                    data_img_lvl_list[img_id].append(data_list[lvl_id][img_id])
            return data_img_lvl_list
        
        cls_scores_img_lvl = lvl2img(cls_scores)
        
        # get labels and bbox_targets of each image
        labels_list, extra_results_list = multi_apply(
            self._get_target_mask_single,
            gt_bboxes_list,
            gt_labels_list,
            cls_scores_img_lvl,
            img_metas,
            points=concat_points,
            num_points_per_lvl=num_points)

        def img2lvl(data_list, num_data_per_lvl):
            num_levels = len(num_data_per_lvl)
            data_list = [data.split(num_data_per_lvl, 0) for data in data_list]

            concat_lvl_data = []       
            for i in range(num_levels):
                concat_lvl_data.append(
                    torch.cat([data[i] for data in data_list]))
            
            return concat_lvl_data
        
        concat_lvl_labels = img2lvl(labels_list, num_points)
        
        return concat_lvl_labels, extra_results_list

    @torch.no_grad()
    def _get_target_mask_single(self, gt_bboxes, gt_labels, cls_scores_lvl, img_metas, points, num_points_per_lvl):
        """Generate target masks for a single image, and visuzalize the results and generate the pseudo-labels.
        Args:
            gt_bboxes (Tensor): Ground truth bounding boxes with shape (num_gts, 5).
            gt_labels (Tensor): Ground truth labels with shape (num_gts,).
            cls_scores_lvl (list[Tensor]): Classification scores for all levels, each with shape (num_points_lvl, num_classes).
            img_metas (dict): Image meta information.
            points (Tensor): Points with shape (num_points, 2).
            num_points_per_lvl (list[int]): Number of points per level.
        Returns:
            tuple: A tuple containing:
                - labels (Tensor): Labels for each point.
                - extra_results (dict): Additional results for visualization and evaluation. Default is None.
        """
        num_gts = gt_labels.size(0)
        gt_points = gt_bboxes[:, :2] # just use the center points of gt_bboxes for point-supervised learning
        num_lvl = len(cls_scores_lvl)
        featmap_sizes = [cls_scores_lvl[i].shape[-2:] for i in range(num_lvl)]

        vis_results = dict()

        if self.need_pseudo_label or num_gts == 0:
            labels, vis_results = self.dummy_assign_label(points)
        else:
            labels, extra_results = self.ssp_assign_label(gt_points, gt_labels, points[:, :2], self.num_classes, img_metas)

        if self.is_record_stage:
            if self.need_visualize or self.need_pseudo_label:
                _ie_results = self.ssp_instance_extraction(cls_scores_lvl[0], gt_points*0.25, gt_labels, sp_thres=self.sp_thres, 
                                                  cls_square=self.cls_square, cls_merge=self.cls_merge, cls_overlap=self.cls_overlap, cls_stable=self.cls_stable)
                cpm_partition, cpm_growing, pse_boxes_pca, pse_boxes_rect, pse_boxes_hybrid, pse_boxes_auto =_ie_results
                pse_labels = torch.concat([gt_labels[gt_labels==cls_id] for cls_id in range(self.num_classes)])
                pse_gt_bboxes = torch.concat([gt_bboxes[gt_labels==cls_id] for cls_id in range(self.num_classes)])

            if self.need_pseudo_label:
                ssp_generate_label_single(pse_labels, pse_boxes_pca, img_metas, self.runner_info['dataset'], self.pseudo_label_dir+'/vor_pca', format='le90')
                ssp_generate_label_single(pse_labels, pse_boxes_rect, img_metas, self.runner_info['dataset'], self.pseudo_label_dir+'/vor_rect', format='le90')
                ssp_generate_label_single(pse_labels, pse_boxes_hybrid, img_metas, self.runner_info['dataset'], self.pseudo_label_dir+'/vor_mix', format='le90', need_print=True)
                ssp_generate_label_single(pse_labels, pse_boxes_auto, img_metas, self.runner_info['dataset'], self.pseudo_label_dir+'/vor_auto', format='le90')

            if self.need_visualize:
                vis_results['cpm_target'] = t2n(torch.split(labels, num_points_per_lvl, dim=0)[0].reshape(*featmap_sizes[0]))
                vis_results['cpm_pred'] = t2n(cls_scores_lvl[0].sigmoid())

                vis_results['cpm_partition'] = t2n(cpm_partition - 2)
                vis_results['cpm_growing'] = t2n(cpm_growing - 2)

                vis_results['img_partition'] = t2n(extra_results['img_partition'] - 2)
                vis_results['img_growing'] = t2n(extra_results['img_growing'] - 2)
                
                vis_results['pse_boxes_gt'] = t2n(pse_gt_bboxes)
                vis_results['pse_boxes_ssp_hybrid'] = t2n(pse_boxes_hybrid)
                vis_results['pse_labels'] = t2n(pse_labels)
                vis_results['ori_labels'] = t2n(gt_labels)

                metric, ious = eval_rbox_single_image(pse_boxes_hybrid, pse_gt_bboxes, pse_labels, self.num_classes, class_aware=False)
                vis_results['metric'] = metric
                vis_results['ious'] = ious

                visualize_dir = os.path.join(self.store_dir, self.visualize_dir)
                ssp_visualize_single(vis_results, img_metas, self.runner_info['dataset'], visualize_dir, scale=0.25, need_print=True)
        
        return labels, None
 
    def dummy_assign_label(self, points):
        """Assign dummy labels for the case when no ground truth is available."""
        num_points = points.size(0)
        labels = -1 * points.new_ones(num_points, dtype=torch.long)
        labels[0] = 0 # assign at least one point to class#bg to avoid error
        return labels, dict()

    def ssp_assign_label(self, gt_points, gt_labels, points, num_classes, img_metas):
        """Assign labels to points based on ground truth points and labels."""    
        default_gt_radius = (8, 128)
        
        device = gt_points.device
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_points
        
        dist_sample_and_gt = torch.cdist(points, center_point_gt) # [num_sample, num_gt]
        dist_gt_and_gt = torch.cdist(center_point_gt, center_point_gt) # [num_gt, num_gt]

        # 0. determine guessed_gt_radius based [class-compatible] gt_points
        label_idx = gt_labels[:, None].expand(num_gts, num_gts)
        cls_compat = torch.ones((self.num_classes, self.num_classes), dtype=torch.bool).to(device)
        for cls_group in self.cls_overlap:
            a, b = cls_group
            cls_compat[a, b] = cls_compat[b, a] = False
        compat_mask = cls_compat[label_idx, label_idx.T]
        self_mask = torch.eye(num_gts, device=device, dtype=torch.bool)
        dist_gt_and_gt[~compat_mask|self_mask] = INF

        guessed_gt_radius, _ = dist_gt_and_gt.min(dim=1)
        guessed_gt_radius[guessed_gt_radius == INF] = default_gt_radius[1]
        inner_mask = dist_sample_and_gt < guessed_gt_radius[None, :]
        
        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype, device=gt_labels.device)
        
        if num_gts == 1:
            inner_mask = dist_sample_and_gt < default_gt_radius[1]
            index_pos = (dist_sample_and_gt < default_gt_radius[0]).nonzero()
            index_neg = (dist_sample_and_gt > default_gt_radius[1]).nonzero()
            labels[index_pos[:, 0]] = gt_labels[0]
            labels[index_neg[:, 0]] = num_classes

        else:
            # 1.1. radius-based negtive label-assignment
            index_neg = (~inner_mask).all(dim=1).nonzero().squeeze(-1)
            if len(index_neg) > 0:
                labels[index_neg] = num_classes

            # 1.2 radius-based postive label-assignment
            final_gt_radius = torch.min(guessed_gt_radius/2, torch.full_like(guessed_gt_radius, default_gt_radius[0]))
            index_pos = (dist_sample_and_gt < final_gt_radius).nonzero()
            if len(index_pos) > 0:
                labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

        # 2. vor-based postive lable-assignment
        ## scale=0.25
        ## gt_points = gt_points * scale
        ## raw_img = ssp_load_raw_image(img_metas, scale=scale, normalize=True)
        ## 
        ## h, w = raw_img.shape[:2]
        ## sp_map, ridge_mask  = self.spatital_partition((h, w), gt_points, gt_labels, sp_thres=self.sp_thres)
        ## rg_map, rg_pro_map = self.region_growing(sp_map, raw_img, gt_labels, self.num_classes, need_filter=True)

        (h, w), sp_map, ridge_mask, rg_map, rg_pro_map = self.ssp_scaled_partition_growing(
            img_metas, gt_points, gt_labels, work_scale=0.5, base_scale=0.25)

        #rg_pro_map = rg_map

        # 2.1 Add some postive samples based on partition&growing regions
        pos_mask = (labels[:h*w] == -1) & (rg_pro_map.flatten() > 1)
        labels[:h*w][pos_mask] = gt_labels[rg_pro_map.flatten()[pos_mask]-2]

        # 2.2 Add some negtive samples based on partition&growingridges, but
        # exclude ridges in overlap regions of overlap-cls,
        # e.g. ship-ridges in both ship-harbor regions
        sample_gt_label = gt_labels[None, :].expand_as(dist_sample_and_gt).clone()
        sample_gt_label[~inner_mask] = -1
        overlap_mask = torch.zeros(num_points, device=device, dtype=torch.bool)
        for cls_group in self.cls_overlap:
            cls_overlap = torch.tensor(cls_group, device=device)
            overlap_mask |= torch.isin(cls_overlap[None, :], sample_gt_label).all(dim=1)
        
        bg_mask = ridge_mask.flatten() & ~overlap_mask[:h*w]
        labels[:h*w][bg_mask] = num_classes

        #labels[labels==-1] = num_classes # set for visualization

        #### extral visualize code ####
        if self.need_visualize:
            rg_map = torch.where(rg_map > 1, rg_map, sp_map)
            full_vor_map, _  = self.spatital_partition((h, w), gt_points*0.25, gt_labels, dict(pos=[-1.]*num_classes, neg=[-1.]*num_classes))
            empty_map = torch.zeros((h, w), dtype=torch.int32, device=device)
            for k in range(len(gt_points)):
                radius_mask = dist_sample_and_gt[:h*w, k].reshape((h, w)) < guessed_gt_radius[k]
                partition_mark = (full_vor_map == k + 2)
                gt_mask = radius_mask & partition_mark
                empty_map = torch.where(gt_mask, torch.full_like(empty_map, k+2), empty_map)
            empty_map = torch.where(empty_map == 0, torch.ones_like(empty_map), rg_map.int())
            for k in range(len(gt_points)):
                empty_map = torch.where(empty_map == k+2, torch.full_like(empty_map, gt_labels[k]), empty_map)
        else:
            empty_map = None
        #### extral visualize code ####
        
        return labels, dict(
            img_partition=sp_map,
            img_growing=rg_map,
            ideal_mask=empty_map,)

    def ssp_scaled_partition_growing(self, img_metas, gt_points, gt_labels, work_scale=0.25, base_scale=0.25):
        """Perform spatial partitioning and region growing on the image at a specified scale."""
        work_raw_img = ssp_load_raw_image(img_metas, scale=work_scale, normalize=True)
        
        scale_ratio = int(work_scale / base_scale)
        work_h, work_w = work_raw_img.shape[:2]
        base_h, base_w = work_h // scale_ratio,  work_h // scale_ratio

        sp_map, ridge_mask  = self.spatital_partition((base_h, base_w), gt_points*base_scale, gt_labels, sp_thres=self.sp_thres)

        # resize sp_map and ridge_mask to work_scale
        work_sp_map = nn.functional.interpolate(
            sp_map[None, None].float(), size=(work_h, work_w), mode='nearest').squeeze().long()
        
        rg_map, rg_pro_map = self.region_growing(work_sp_map, work_raw_img, gt_labels, self.num_classes, need_filter=True)

        # resize work_rg_map and work_rg_pro_map to base_scale
        rg_map = nn.functional.interpolate(
            rg_map[None, None].float(), size=(base_h, base_w), mode='nearest').squeeze().long()
        rg_pro_map = nn.functional.interpolate(
            rg_pro_map[None, None].float(), size=(base_h, base_w), mode='nearest').squeeze().long()

        return (base_h, base_w), sp_map, ridge_mask, rg_map, rg_pro_map

    def spatital_partition(self, img_size, gt_points, gt_labels, sp_thres, conf_map=None, other_ridge=None):
        """
        Performs spatial partitioning of an image based on ground truth points and labels, generating a partition map and ridge mask.
        Args:
            img_size (tuple): The size of the image as (height, width).
            gt_points (Tensor): Tensor of shape (J, 2) containing ground truth point coordinates.
            gt_labels (Tensor): Tensor of shape (J,) containing class labels for each ground truth point.
            thres_cfg (dict): Threshold configuration dictionary. Should contain 'default' thresholds and optionally 'override' and 'confidence' keys.
            conf_map (Tensor, optional): Confidence map of shape (h, w). If provided, used for further partition refinement. Default is None.
            other_ridge (Tensor, optional): Precomputed ridge mask of shape (h, w). If provided, used instead of computing ridge mask. Default is None.
        Returns:
            sp_map (Tensor): Spatial partition map of shape (h, w), where each pixel is assigned a partition id (0: ignore, 1: background, 2...N+2: class ids).
            ridge_mask (Tensor): Boolean mask of shape (h, w) indicating ridge (boundary) regions in the partition map.
        """
        device = gt_points.device
        J = len(gt_points)
        h, w = img_size
        if J == 0:
            sp_map = sp_lines = gt_labels.new_zeros((h, w))
            return sp_map, sp_lines

        #1. Execute basic spatial partition, which generates a map full of the obj_id
        x = torch.linspace(0, h, h, device=device)
        y = torch.linspace(0, w, w, device=device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)

        dm = DistanceMap.get_instance((h, w))
        sp_map = dm.compute_map_batch(gt_points)
        sp_prob_map, sp_map = torch.max(sp_map, 0)

        #2. Execute partition refinement, which extends the map with bg_id and ing_id
        #2.1 Prepare ridge_mask
        if other_ridge is not None:
            ridge_mask = other_ridge.bool()
        else:
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
            kernel[0, 0, 1, 1] = -8.0
            ridge_mask =(torch.conv2d(sp_map[None, None].float(), kernel, padding=1) != 0).squeeze()

        #2.2 Prepare core_mask, bg_mask
        pos_thres, neg_thres = sp_prob_map.new_tensor(sp_thres['pos']), sp_prob_map.new_tensor(sp_thres['neg'])
        sp_cls_map = gt_labels[sp_map]
        core_mask = sp_prob_map > pos_thres[sp_cls_map]
        bg_mask = sp_prob_map < neg_thres[sp_cls_map]

        #2.3 Assign refined partition id, i.e. 0 is unknow region, 1 is backgound region, 2...N+2 is cls 1...N
        # Don't move to the other position
        ign_id, bg_id = 0, 1
        sp_map += 2

        if conf_map is None:
            sp_map[~core_mask] = ign_id
            sp_map[bg_mask|ridge_mask] = bg_id
        else:
            low_thres, mid_thres, high_thres = sp_thres['conf'] # 0.05. 0.6, 0.95
            low_quality_mask = conf_map < conf_map.max() * low_thres
            high_quality_mask = conf_map > conf_map.max() * high_thres
            mid_quality_mask = conf_map > conf_map.max() * mid_thres

            many_ins_flag = J > 1

            dist_gt_with_gt = torch.cdist(gt_points, gt_points)/2. + INF * torch.eye(J, device=device)
            mid_gt = (gt_points+gt_points[dist_gt_with_gt.min(dim=1)[1]])/2
            mid_mask = (torch.cdist(xy.view(-1, 2), mid_gt) < 2).any(dim=1).reshape(h, w)

            guessed_gt_radius, _ = dist_gt_with_gt.min(dim=1)
            near_mask = (torch.cdist(xy.view(-1, 2), gt_points) < guessed_gt_radius).any(dim=1).reshape(h, w)

            sp_map[~(core_mask|(high_quality_mask&near_mask&many_ins_flag))] = ign_id
            sp_map[bg_mask|low_quality_mask|(ridge_mask&~mid_quality_mask)|(mid_mask&many_ins_flag)] = bg_id

        return sp_map, ridge_mask

    def region_growing(self, sp_map, img_u8c3, gt_labels, num_cls, need_filter=False):
        """
        Perform region growing segmentation using watershed algorithm and optionally filter outlier instances.
        Args:
            sp_map (torch.Tensor): Superpixel map or initial segmentation map (H, W).
            img_u8c3 (np.ndarray): Input image in uint8 format with 3 channels (H, W, 3).
            gt_labels (torch.Tensor): Ground truth labels for each instance (N,).
            num_cls (int): Number of classes.
            need_filter (bool, optional): Whether to filter outlier instances based on area. Defaults to False.
        Returns:
            tuple:
                - rg_map (torch.Tensor): Region map after watershed segmentation.
                - rg_pro_map (torch.Tensor or None): Processed region map with outlier instances set to ignore region (0), or None if need_filter is False.
        """

        #1. Execute region growing based on watershed
        rg_map = watershed_segmentation(img_u8c3, sp_map)

        if not need_filter:
            return rg_map, None

        #2 Drop outliter instance mask according to the area of each object
        #2.1 Calculate the area of all objects.
        obj_id, ins_area = torch.unique(rg_map, return_counts=True)
        obj_id -= 2 # recover original obj_id, i.e. [2...N] -> [0...N-2], and bg_id, ign_id < 0 will be dropped
        all_obj_id = torch.arange(len(gt_labels), device=sp_map.device)
        all_obj_area = sp_map.new_zeros(len(gt_labels))
        all_obj_area[torch.isin(all_obj_id, obj_id).nonzero().flatten()] = ins_area[torch.isin(obj_id, all_obj_id)]

        #2.2 Assign bad instance to ignore region
        outlier_obj_id = []
        for cls_id in range(num_cls):
            cls_obj_id = (gt_labels == cls_id).nonzero().squeeze(-1)
            cls_obj_area = all_obj_area[cls_obj_id]
            if len(cls_obj_id) != 0:
                outlier_obj_id.append(cls_obj_id if len(cls_obj_id) == 1 else cls_obj_id[cls_obj_area > cls_obj_area.median() * 3])
        outlier_obj_id = torch.concat(outlier_obj_id) + 2

        rg_pro_map = rg_map.clone()
        rg_pro_map[(rg_map.unsqueeze(-1) == outlier_obj_id).any(-1)] = 0 

        return rg_map, rg_pro_map

    def ssp_instance_extraction(self, scores, gt_points, gt_labels, sp_thres, scale=0.25, cls_square=None, cls_merge=None, cls_overlap=None, cls_stable=None):
        """
        Extracts instance-level segmentation and bounding boxes from class probability maps and ground truth points.

        Args:
            scores (torch.Tensor): Class probability maps of shape (num_cls, H, W).
            gt_points (torch.Tensor): Ground truth points for instances, shape (N, 2).
            gt_labels (torch.Tensor): Ground truth class labels for each point, shape (N,).
            sp_thres (dict): Configuration for thresholding and partitioning.
            scale (float, optional): Scaling factor for output bounding boxes. Default is 0.25.
            cls_square (list, optional): List of square cls, whose angle will be set as zero. Defaults to None.
            cls_merge (list or None, optional): If provided, specifies class merging mapping.
            cls_overlap (list or None, optional): If provided, specifies pairs of overlapping classes for compatibility.
            cls_stable (list or None, optional): If provided, specifies classes that are stable.
        Returns:
            tuple:
                cpm_partition (torch.Tensor): Partition maps for each class, shape (num_cls, H, W).
                cpm_growing (torch.Tensor): Region growing maps for each class, shape (num_cls, H, W).
                pse_boxes_pca (torch.Tensor): Bounding boxes (PCA minmax) for all instances, shape (M, 5).
                pse_boxes_rect (torch.Tensor): Bounding boxes (min area rectangle) for all instances, shape (M, 5).
                pse_boxes_hybrid (torch.Tensor): Hybrid bounding boxes for all instances, shape (M, 5).
        """
        num_cls, h, w = scores.shape
        device = gt_points.device

        # 1. normalize scores
        scores = scores.detach().sigmoid()
        _min = scores.view(num_cls, -1).min(1)[0].reshape(num_cls, 1, 1)
        _max = scores.view(num_cls, -1).max(1)[0].reshape(num_cls, 1, 1)
        scores = (scores - _min) / (_max - _min)

        #2. Generate global ridge mask based on compatible classes
        if cls_overlap is None:
            _, glo_ridge_mask = self.spatital_partition((h, w), gt_points, gt_labels, sp_thres)
        else:
            cls_compat = torch.ones((num_cls, num_cls), dtype=torch.bool).to(device)
            for cls_group in cls_overlap:
                a, b = cls_group
                cls_compat[a, b] = cls_compat[b, a] = False

            sp_cache = {}
            glo_ridge_masks = {}
            unique_cls_set = torch.unique(gt_labels).tolist()
            for cls_id in unique_cls_set:
                cls_friends = cls_compat[cls_id].nonzero().flatten()
                cls_select_mask = torch.isin(gt_labels, cls_friends)
                cls_friends = tuple(cls_friends.tolist())
                ridge_mask =  sp_cache.get(cls_friends, None)
                if ridge_mask is None:
                    friends_gt_points = gt_points[cls_select_mask]
                    friends_gt_labels = gt_labels[cls_select_mask]
                    _, ridge_mask = self.spatital_partition((h, w), friends_gt_points, friends_gt_labels, sp_thres)
                    sp_cache[cls_friends] = ridge_mask
                glo_ridge_masks[cls_id] = ridge_mask
            #print(len(sp_cache), sp_cache.keys())

        if cls_merge is not None:
            cls_mapping = torch.zeros(num_cls, dtype=torch.long).to(device)
            for big_id, small_id in enumerate(cls_merge):
                cls_mapping[small_id] = big_id

        #3. Execute region growing and box conversion for each class
        cpm_partition = []
        cpm_growing= []
        pse_boxes_pca = []
        pse_boxes_rect = []
        pse_boxes_hybrid = []
        pse_boxes_auto = []
        for cls_id in range(num_cls):
            cls_gt_points = gt_points[gt_labels == cls_id] 
            cls_gt_labels = gt_labels[gt_labels == cls_id]
            if len(cls_gt_labels) == 0:
                cpm_partition.append(gt_labels.new_zeros((h, w))) 
                cpm_growing.append(gt_labels.new_zeros((h, w)))
            else:
                glo_ridge_mask = glo_ridge_masks[cls_id] if cls_overlap is not None else glo_ridge_mask
                sp_map, _ = self.spatital_partition((h, w), cls_gt_points, cls_gt_labels, sp_thres, conf_map=scores[cls_id], other_ridge=glo_ridge_mask)

                cpm_img = ((1 - (scores[cls_id])) * 255).detach().cpu().numpy().astype(np.uint8)
                cpm_img = np.stack([cpm_img, cpm_img, cpm_img], -1)
                rg_map, _ = self.region_growing(sp_map, cpm_img, friends_gt_labels, num_cls)

                boxes_pca = self.box_conversion(rg_map, cls_gt_points, cls_id, use_gt_center=True, mode='pca_minmax', cls_square=cls_square, cls_stable=cls_stable)
                boxes_rect = self.box_conversion(rg_map, cls_gt_points, cls_id, use_gt_center=True, mode='minarea_rect', cls_square=cls_square, cls_stable=cls_stable)
                boxes_hybrid_auto = self.box_conversion(rg_map, cls_gt_points, cls_id, use_gt_center=True, mode='hybrid', cls_square=cls_square, cls_stable=cls_stable)

                boxes_hybrid = boxes_pca if cls_merge is not None and cls_mapping[cls_id] == 0 else boxes_rect

                cpm_growing.append(rg_map)
                cpm_partition.append(sp_map)
                pse_boxes_pca.append(boxes_pca)
                pse_boxes_rect.append(boxes_rect)
                pse_boxes_hybrid.append(boxes_hybrid)
                pse_boxes_auto.append(boxes_hybrid_auto)

        cpm_partition = torch.stack(cpm_partition, dim=0)
        cpm_growing = torch.stack(cpm_growing, dim=0)
        pse_boxes_pca = torch.concat(pse_boxes_pca, dim=0)
        pse_boxes_rect = torch.concat(pse_boxes_rect, dim=0)
        pse_boxes_hybrid = torch.concat(pse_boxes_hybrid, dim=0)
        pse_boxes_auto = torch.concat(pse_boxes_auto, dim=0)

        #4. Scales the resulting bounding boxes according to the provided scale factor.
        pse_boxes_pca[:, :4] /= scale
        pse_boxes_rect[:, :4] /= scale
        pse_boxes_hybrid[:, :4] /= scale
        pse_boxes_auto[:, :4] /= scale

        return cpm_partition, cpm_growing, pse_boxes_pca, pse_boxes_rect, pse_boxes_hybrid, pse_boxes_auto

    def box_conversion(self, ins_map, gt_points, cls_id, use_gt_center=False, mode='minarea_rect', cls_square=None, cls_stable=None):
        """
        Convert instance masks and ground truth points to bounding box representations.
        Args:
            ins_map (torch.Tensor): Instance map where each pixel value indicates the instance id.
            gt_points (torch.Tensor): Ground truth points for each instance, shape (N, 2).
            cls_id (int): Class id for the current set of instances.
            use_gt_center (bool, optional): If True, use ground truth points as box centers. Defaults to False.
            mode (str, optional): Method for box conversion. Options are 'minarea_rect' or 'pca_minmax'. Defaults to 'minarea_rect'.
            cls_square (list, optional): List of square cls, whose angle will be set as zero. Defaults to None.
            cls_stable (list or set, optional): List or set of class ids for which bounding box stabilization is applied. Defaults to None.
        Returns:
            torch.Tensor: Bounding boxes for each instance, shape (N, 5), where each box is (center_x, center_y, width, height, angle).
        """
        bboxes = []
        for ins_id in range(len(gt_points)):
            ins_mask_pts = (ins_map == ins_id + 2).nonzero().float()[:, [1, 0]] #i,j -> x,y
            area = len(ins_mask_pts)
            if area <= 1:
                bbox = np.array([*gt_points[ins_id].cpu(), 0, 0, 0])

            elif mode == 'pca_minmax':
                bbox = self._m2b_pca_minmax(ins_mask_pts, gt_point=None)
            elif mode == 'minarea_rect':
                bbox = self._m2b_minarea_rect(ins_mask_pts)
            else:
                bbox = self._m2b_hybrid_auto(ins_mask_pts, thres=0.8)
            bboxes.append(bbox)
        bboxes = torch.tensor(np.array(bboxes), device=ins_map.device)
        if use_gt_center:
            #bboxes[:, :2] = gt_points if use_gt_center else bboxes[:, :2]
            bboxes = self._rbox_clip(bboxes, gt_points, img_shape=ins_map.shape[:2])
        if cls_square is not None and cls_id in cls_square:
            bboxes[:, 4] = 0

        if cls_stable is not None and cls_id in cls_stable:
            areas = bboxes[:, 2] * bboxes[:, 3]
            zero_area_mask = areas == 0
            if len(bboxes) >= 5 & len(areas[~zero_area_mask]) > 0:
                ref_area = areas[~zero_area_mask].median()
                bad_ins_mask = zero_area_mask | (areas > ref_area * 3.) | (areas < ref_area / 3.)
                if len(bboxes[~bad_ins_mask]) > 0:
                    good_bbox = bboxes[~bad_ins_mask].reshape(-1, 5).median(0)[0]
                    bboxes[bad_ins_mask, 2:4] = good_bbox[None, 2:4]
        return bboxes

    def _m2b_pca_minmax(self, mask_pts, gt_point=None):
        U, S, V = torch.pca_lowrank(mask_pts)
        theta = torch.atan2(V[1, 0], V[0, 0])
        center = gt_point if gt_point is not None else mask_pts.mean(dim=0)
        mask_pts = torch.mm(mask_pts - center[None, :], V)
        (l, d), (r, t) = mask_pts.min(0)[0].tolist(), mask_pts.max(0)[0].tolist()
        w, h = r - l, t - d
        #w, h = 2 * max(abs(l), abs(r)), 2 * max(abs(t), abs(d))
        bbox = np.array([*center.tolist(), max(w, 0), max(h, 0), theta.item()])
        return bbox

    def _m2b_minarea_rect(self, mask_pts):
        min_rect = cv2.minAreaRect(mask_pts.cpu().numpy())
        bbox = np.array([*min_rect[0], *min_rect[1], min_rect[2] / 180 * np.pi])
        return bbox

    def _m2b_hybrid_auto(self, mask_pts, thres=0.8):
        box_rect = self._m2b_minarea_rect(mask_pts)
        box_pca = self._m2b_pca_minmax(mask_pts)

        overlap_rect = mask_rbox_iou(mask_pts.detach().cpu().numpy(), box_rect)
        overlap_pca = mask_rbox_iou(mask_pts.detach().cpu().numpy(), box_pca)

        if overlap_rect > overlap_pca:
            return box_rect
        else:
            return box_pca

    def _rbox_clip(self, boxes, gt_points, img_shape):
        """
        Clip rotated bounding boxes based on ground truth center points.
        
        Computes new boxes based on the offset between gt_points and box centers.
        If the new boxes' edge midpoints exceed image boundaries, the original boxes are used.
        
        Args:
            boxes (torch.Tensor): Rotated bounding boxes with shape (N, 5) in format 
                                 (x, y, w, h, angle), where angle is in radians
            gt_points (torch.Tensor): Ground truth center points with shape (N, 2) 
                                     in format (x, y)
            img_shape (tuple): Image shape in format (height, width)
            
        Returns:
            tuple: A tuple containing:
                - final_boxes (torch.Tensor): Processed rotated boxes with shape (N, 5)
                - new_boxes (torch.Tensor): Computed new boxes with shape (N, 5)
                - mid_points (torch.Tensor): Edge midpoints with shape (N, 4, 2)
                - all_in_range (torch.Tensor): Boolean mask indicating if all midpoints 
                                              are within image boundaries (N,)
        """
        # Split box components
        center, wh, angle = torch.split(boxes, [2, 2, 1], dim=-1)  # center: (N,2), wh: (N,2), angle: (N,1)
        
        # Calculate center offset
        offset = center - gt_points  # (N,2)
        
        # Compute rotation matrices for each box angle
        cos_theta = torch.cos(angle).squeeze(-1)  # (N,)
        sin_theta = torch.sin(angle).squeeze(-1)  # (N,)
        
        # Compute new width and height considering offset and rotation
        # This is a simplified approach; real applications may require more complex transformations
        offset = torch.bmm(torch.stack([cos_theta, sin_theta, -sin_theta, cos_theta], dim=-1) \
                                           .view(-1, 2, 2), offset.unsqueeze(-1)).squeeze(-1).abs()  # (N,2)
        larger_wh = 2 * (0.5 * wh + offset)
        smaller_wh = 2 * (0.5 * wh - offset)

        # Construct new boxes using gt_points as centers
        larger_boxes = torch.cat([gt_points, larger_wh, angle], dim=-1)  # (N,5)
        smaller_boxes = torch.cat([gt_points, smaller_wh, angle], dim=-1)  # (N,5)

        # Calculate edge midpoints for new boxes
        mid_points = rbox_to_mid_points(larger_boxes)  # (N,4,2)
        
        # Check if midpoints exceed image boundaries
        h, w = img_shape
        # Check if each midpoint is within image range
        in_range = (mid_points[..., 0] >= 0) & (mid_points[..., 0] < w) & \
                   (mid_points[..., 1] >= 0) & (mid_points[..., 1] < h)  # (N,4)
        
        # Use new box if all midpoints are within range, otherwise use original box
        all_in_range = torch.all(in_range, dim=1)  # (N,)
        
        # Select final boxes
        final_boxes = torch.where(all_in_range.unsqueeze(-1), smaller_boxes, larger_boxes)
        
        # Return final boxes and intermediate results for visualization
        return final_boxes#, new_boxes, mid_points, all_in_range
