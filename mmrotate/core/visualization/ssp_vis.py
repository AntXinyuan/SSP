import os
import colorsys, random
import torch
import numpy as np

import cv2
import mmcv
from mmrotate.core.visualization import imshow_det_rbboxes

            
def ssp_visualize_single(vis_results, img_metas, dataset, output_dir, scale=0.25, need_print=False):
    """
    Visualizes the results of a single SSP sample, including ground truth and predicted bounding boxes, CPM (Class Probability Map) targets and predictions, and various intermediate visualizations.
    Args:
        vis_results (dict): Dictionary containing various intermediate and final results for visualization, such as predicted boxes, labels, CPM masks, partitions, and metrics.
        img_metas (dict): Metadata for the image, including file path, image shape, and padding information.
        dataset (object): Dataset object or identifier used to retrieve class names and color palettes for visualization.
        output_dir (str): Directory where the visualization output image will be saved.
        scale (float, optional): Scaling factor for resizing images and bounding boxes. Default is 0.25.
    Returns:
        None. The function saves the visualization image to the specified output directory and prints the output path.
        
    Notes:
        - The function creates a composite visualization image showing: 
            1)img_with_gt_boxes, 2)cpm_target, 3)cpm_pred, 4)cls_cpm*, 5)img_partition, 6)img_growing 7)pse_boxes*
        - Supporting vis_results keys include:
            - `pse_boxes_ssp_hybrid`: Predicted boxes from the SSP hybrid method, shape (num_pse, 5).
            - `pse_boxes_gt`: Ground truth boxes, shape (num_gt, 5).
            - `pse_labels`: Labels for the predicted boxes, shape (num_pse,).
            - `cpm_target`: Class Probability Map target, shape (H, W).
            - `cpm_pred`: Class Probability, shape (num_cls, H, W).
            - `cpm_partition`: Partition masks for each class, shape (num_cls, H, W).
            - `cpm_growing`: Growing visualization for each class, shape (num_cls, H, W).
            - `img_partition`: Partition visualization for the entire image, shape (H, W).
            - `img_growing`: Growing visualization for the entire image, shape (H, W).
            - `pse_boxes_*`: Additional predicted boxes for visualization.
        - The output image includes a caption summarizing the metrics and the content of each visualization panel.
    """
    # img_with_gt_boxes, img_with_cpm_target, img_with_cpm_pred,
    # cls_cpm, cls_cpm_mask, cls_cpm_boxes
    # gt_boxes, gt_pts, gt_pts_with_parttion, gt_pts_with_growing
    # pse_boxes_ssp_hybrid, pse_boxes_gt 

    img_scale = scale
    box_scale = np.array([scale, scale, scale, scale, 1.0]).reshape(1, 5)

    
    scale = 0.25
    scale_factor = np.array([scale, scale, scale, scale, 1.0]).reshape(1, 5)
    
    # 0. prepare image
    image_name = os.path.basename(img_metas['filename'])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, image_name)
    image = ssp_load_raw_image(img_metas, scale=scale)

    white_image = np.ones_like(image) * 255

    CLASSES, PALETTE = get_visualize_database(dataset)
    num_cls = len(CLASSES)

    pse_boxes = vis_results['pse_boxes_ssp_hybrid'] * box_scale #(num_pse, 5)
    pse_labels = vis_results['pse_labels']     
    instance_pattle = get_rainbow_color(len(pse_labels), dtype=int, shuffle=True)
    
    gt_bboxes = vis_results['pse_boxes_gt'] * box_scale  #(num_gt, 5)
    gt_labels = vis_results['pse_labels']                #(num_gt,)
    img_with_gt_boxes = imshow_det_rbboxes(image, gt_bboxes, gt_labels, bbox_color=PALETTE, show_text=False)

    cpm_target = vis_results['cpm_target'] #(h, w)
    img_with_cpm_target = imshow_cpm_mask(cpm_target, PALETTE)

    cpm_pred = vis_results['cpm_pred'] #(num_cls, h, w)
    cpm_pred_cls = np.argmax(cpm_pred, axis=0)
    cpm_pred_prob = np.max(cpm_pred, axis=0)
    img_with_cpm_pred = imshow_cpm_mask(cpm_pred_cls, PALETTE, cpm_prob=cpm_pred_prob)

    fig_cls_cpm = []
    for cls_id in range(num_cls):
        num_cls_ins = sum(pse_labels == cls_id)
        if num_cls_ins == 0:
            continue
        #cls_ins_pattle = instance_pattle
        cls_ins_pattle = [PALETTE[cls_id],] * num_cls_ins

        cls_cpm_prob = cpm_pred[cls_id]
        cls_cpm_prob = (cls_cpm_prob - cls_cpm_prob.min()) / (cls_cpm_prob.max() - cls_cpm_prob.min())
        cls_cpm_cls = np.ones(cls_cpm_prob.shape, dtype=np.int32) * cls_id
        fig_cls_cpm_scores = imshow_cpm_mask(cls_cpm_cls, PALETTE, cpm_prob=cls_cpm_prob)

        cls_cpm_partition = vis_results['cpm_partition'][cls_id] #(h, w)
        fig_cls_cpm_partition = imshow_cpm_mask(cls_cpm_partition, cls_ins_pattle, bg_cls_id=-1, ign_cls_id=-2)
        
        cls_cpm_growing = vis_results['cpm_growing'][cls_id] #(h, w)
        fig_cls_cpm_gowing = imshow_cpm_mask(cls_cpm_growing, cls_ins_pattle, bg_cls_id=-1, ign_cls_id=-2)

        cls_cpm_boxes = pse_boxes[pse_labels==cls_id]
        cls_cpm_cls = pse_labels[pse_labels==cls_id]
        fig_cls_cpm_boxes = imshow_det_rbboxes(white_image.copy(), cls_cpm_boxes, cls_cpm_cls, bbox_color=PALETTE, show_text=False)

        fig_cls_cpm.extend([fig_cls_cpm_scores, fig_cls_cpm_partition, fig_cls_cpm_gowing, fig_cls_cpm_boxes])

    
    #cls_ins_pattle = instance_pattle
    ori_labels = vis_results['ori_labels']
    cls_ins_pattle = [PALETTE[ori_labels[ins_id]] for ins_id in range(len(ori_labels))]

    img_partition = vis_results['img_partition']
    fig_img_parttion = imshow_cpm_mask(img_partition, cls_ins_pattle, bg_cls_id=-1, ign_cls_id=-2)

    img_growing =  vis_results['img_growing']
    fig_img_growing = imshow_cpm_mask(img_growing, cls_ins_pattle, bg_cls_id=-1, ign_cls_id=-2)

    visualize_results = []
    visualize_results.extend([img_with_gt_boxes, img_with_cpm_target, img_with_cpm_pred, *fig_cls_cpm, fig_img_parttion, fig_img_growing])
    num_basic_results = len(visualize_results)
        
    metric_text = ','.join([f'{k}:{v:4.2f}' for k, v in vis_results['metric'].items()])
    visualize_caption = f'[{image_name}] ({metric_text}) '\
        + '1)img_with_gt_boxes, 2)cpm_target, 3)cpm_pred, 4)cls_cpm*, 5)img_partition, 6)img_growing'
        
    for idx, (field_name, extra_bboxes) in enumerate(filter(lambda x: x[0].find('pse_boxes_') != -1, vis_results.items())):
        extra_bboxes = extra_bboxes * box_scale
        img_with_pse_boxes = imshow_det_rbboxes(image, extra_bboxes, vis_results['pse_labels'], bbox_color=PALETTE, show_text=False)
        pse_boxes = imshow_det_rbboxes(255*np.ones_like(image), extra_bboxes, vis_results['pse_labels'], bbox_color=PALETTE, show_text=False)
        visualize_results.append(pse_boxes)
        visualize_results.append(img_with_pse_boxes)
        visualize_caption += f', {idx+num_basic_results}){field_name}'

    result_image = matrix_concat_images(visualize_results, cols=8)
    result_image = add_caption_to_image(result_image, visualize_caption)
    mmcv.imwrite(result_image, output_path)
    if need_print:
        print(f'Visualize {image_name} with {len(visualize_results)} results, saved to {output_path}')

def imshow_cpm_mask(cpm_cls, palette, cpm_prob=None, scale=None, bg_cls_id=None, ign_cls_id=None):
    """
    Displays the confidence mask for a specific class.

    This function generates a visual mask based on the confidence of classification results. It assigns a unique color from the palette to each class,
    with special handling for background and ignore classes.

    Parameters:
    - cpm_cls: Array of classification results, representing the predicted class for each pixel.
    - palette: List of colors, each representing the display color of a class.
    - cpm_prob: Optional, confidence probability mask, used to blend the mask color with the background.
    - scale: Optional, scaling factor for the output mask, adjusting the size of the mask.
    - bg_cls_id: Optional, class ID representing the background, defaults to a class ID of num_cls if not provided.
    - ignore_cls_id: Optional, class ID representing areas to be ignored, these areas will be displayed in black.

    # Defaultly, bg_cls_id = num_cls, ignore_cls_id = -1

    Returns:
    - cpm_mask: The generated class confidence mask image.
    """
    h, w = cpm_cls.shape
    color_bg, color_ingore = 255, 0  # white, black
    cpm_mask = np.ones((h, w, 3), dtype=np.uint8) * color_ingore
    
    num_cls = len(palette)
    for cls_id in range(num_cls):
        cls_mask = cpm_cls == cls_id
        cpm_mask[cls_mask, :] = np.array(palette[cls_id], dtype=np.uint8).reshape((1, 1, 3))
    
    bg_cls_id = num_cls if bg_cls_id is None else bg_cls_id
    cpm_mask[cpm_cls==bg_cls_id] = color_bg

    ign_cls_id = -1 if ign_cls_id is None else ign_cls_id
    cpm_mask[cpm_cls==ign_cls_id] = color_ingore

    if cpm_prob is not None:
        cpm_mask = (cpm_mask * cpm_prob[:, :, None] + color_bg * (1 - cpm_prob[:, :, None])).astype(np.uint8)

    cpm_mask = mmcv.rgb2bgr(cpm_mask)
    if scale is not None:
        cpm_mask = mmcv.imrescale(cpm_mask, scale, interpolation='nearest' if cpm_prob is None else 'bilinear')
    return cpm_mask

def matrix_concat_images(image_list, cols=None):
    num_imgs = len(image_list)
    if cols is None or num_imgs <= cols:
        return np.concatenate(image_list, axis=1)
    (h, w, c), dtype = image_list[0].shape, image_list[0].dtype
    rows = num_imgs // cols + int(num_imgs % cols > 0)
    big_image = np.zeros((rows*h, cols*w, c), dtype=dtype)
    for img_id in range(num_imgs):
        i, j = (img_id // cols) * h, (img_id % cols) * w
        big_image[i:i+h, j:j+w] = image_list[img_id]
    return big_image  
    
def add_caption_to_image(image, caption, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(caption, font, font_scale, thickness)
    caption_height = text_height + 10

    new_height = height + caption_height
    new_image = np.zeros((new_height, width, 3), dtype=np.uint8)
    new_image[:height, :] = image
    new_image[height:, :] = bg_color

    text_position = ((width - text_width) // 2, height + (caption_height - text_height) // 2 + text_height)

    cv2.putText(new_image, caption, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return new_image    

def get_visualize_database(dataset):
    DATASET, CLASSES, PALETTE = dataset
    return CLASSES, PALETTE

def get_rainbow_color(color_num, dtype=float, shuffle=False):
    rainbow_colors = []
    for i in range(color_num):
        hue = i / color_num 
        rgb = tuple([(val if dtype == float else int(val * 255)) for val in colorsys.hsv_to_rgb(hue, 1, 1)])
        rainbow_colors.append(rgb)
    if shuffle:
        random.shuffle(rainbow_colors)
    return rainbow_colors

###################
def ssp_load_raw_image(img_metas, scale=0.25, normalize=False):
    raw_img = mmcv.imread(img_metas['filename'])
    raw_img = mmcv.imresize(raw_img, (img_metas['img_shape'][1], img_metas['img_shape'][0]))
    raw_img = mmcv.impad(img=raw_img, shape=(img_metas['pad_shape'][1], img_metas['pad_shape'][0]))
    raw_img = mmcv.imrescale(raw_img, scale)
    raw_img = recover_images_and_bboxes(image=raw_img, img_metas=img_metas)
    if normalize:
        raw_img = ((raw_img - raw_img.min()) / (raw_img.max() - raw_img.min()) * 255).astype(np.uint8)
        raw_img = cv2.medianBlur(raw_img, 3)
    return raw_img

def recover_images_and_bboxes(image=None, bboxes=None, img_metas=None):
    # bbox (x,w, w, h, a) [n,5]
    direction = img_metas['flip_direction']
    h, w = img_metas['img_shape'][:2]
    results = dict()
    
    if image is not None:
        if direction == 'horizontal':
            image = np.fliplr(image)
        elif direction == 'vertical':
            image = np.flipud(image)
        elif direction == 'diagonal':
            image = np.flipud(np.fliplr(image))
        results['image'] = image
    if bboxes is not None:
        bboxes = bboxes.clone()
        if direction == 'horizontal':
            bboxes[:, 0] = w - bboxes[:, 0]
            bboxes[:, 4] = -bboxes[:, 4]                
        elif direction == 'vertical':
            bboxes[:, 1] = h - bboxes[:, 1]
            bboxes[:, 4] = -bboxes[:, 4]
        elif direction == 'diagonal':
            bboxes[:, 0] = w - bboxes[:, 0]
            bboxes[:, 1] = h - bboxes[:, 1]
        scale_factor = bboxes.new_tensor(img_metas['scale_factor']).unsqueeze(0)
        bboxes[:, :4] = bboxes[:, :4] / scale_factor
        results['bboxes'] = bboxes
    if image is None:
        return bboxes
    elif bboxes is None:
        return image
    else:
        return dict(image=image, bboxes=bboxes)

def simple_obb2poly_le90(obb):
    bbox, angle = obb[:, :4], obb[:, [4,]]
    cos_vals = torch.cos(angle).view(-1, 1, 1)
    sin_vals = torch.sin(angle).view(-1, 1, 1)

    rotation_matrix = torch.cat([torch.cat([cos_vals, -sin_vals], dim=2), torch.cat([sin_vals, cos_vals], dim=2)], dim=1)

    x1, y1 = - bbox[:, 2] / 2, - bbox[:, 3] / 2
    x2, y2 = bbox[:, 2] / 2, - bbox[:, 3] / 2
    x3, y3 = bbox[:, 2] / 2, bbox[:, 3] / 2
    x4, y4 = - bbox[:, 2] / 2, bbox[:, 3] / 2
  
    vertices = torch.stack([torch.stack([x1, x2, x3, x4], dim=1), torch.stack([y1, y2, y3, y4], dim=1)], dim=1)  # [n, 2, 4]

    rotated_vertices = torch.einsum('bij,bjk->bik', rotation_matrix, vertices)  # [n, 2, 4]

    # seperate the vertices after rotation
    x1_rot, y1_rot = bbox[:, 0] + rotated_vertices[:, 0, 0], bbox[:, 1] + rotated_vertices[:, 1, 0]
    x2_rot, y2_rot = bbox[:, 0] + rotated_vertices[:, 0, 1], bbox[:, 1] + rotated_vertices[:, 1, 1]
    x3_rot, y3_rot = bbox[:, 0] + rotated_vertices[:, 0, 2], bbox[:, 1] + rotated_vertices[:, 1, 2]
    x4_rot, y4_rot = bbox[:, 0] + rotated_vertices[:, 0, 3], bbox[:, 1] + rotated_vertices[:, 1, 3]
    
    poly = torch.stack([x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot], dim=1)
    return poly

def ssp_generate_label_single(labels, bboxes, img_metas, dataset, output_dir, format='poly', need_print=False):
    assert format in ['poly', 'le90',]

    CLASSES, _ = get_visualize_database(dataset)
    filename_raw = os.path.basename(img_metas['filename']).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename_raw + '.txt')

    if labels is None or len(labels) == 0:
        with open(output_path, 'w') as f:
            f.write('\n')
        return

    num_instances = len(labels)
    bboxes = recover_images_and_bboxes(bboxes=bboxes, img_metas=img_metas)

    if format == 'poly':
        polys = simple_obb2poly_le90(bboxes)
        with open(output_path, 'w') as f:
            for i in range(num_instances):
                x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot = polys[i]
                f.write(f"{x1_rot.item():.1f} {y1_rot.item():.1f} {x2_rot.item():.1f} {y2_rot.item():.1f} "
                    f"{x3_rot.item():.1f} {y3_rot.item():.1f} {x4_rot.item():.1f} {y4_rot.item():.1f} "
                    f"{CLASSES[labels[i]]} 0\n")
    else:
        with open(output_path, 'w') as f:
            for i in range(num_instances):
                x, y, w, h, a = bboxes[i]
                f.write(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} {a:.4f} {CLASSES[labels[i]]} 0\n")
    if need_print:
        print(f'Generate label for {filename_raw} with {num_instances} instances, saved to {output_path}')


def t2n(input):
    if isinstance(input, list):
        return [t2n(i) for i in input]
    elif isinstance(input, tuple):
        return tuple(t2n(i) for i in input)
    elif isinstance(input, np.ndarray):
        return input
    elif isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    else:
        raise TypeError('invalid input type: {}'.format(type(input)))
    