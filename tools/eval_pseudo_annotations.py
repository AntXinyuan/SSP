from mmcv import Config
from  mmrotate.datasets import *
import numpy as np

def anno2result(anno: dict, num_classes: int):
    if anno is None or len(anno['bboxes']) == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    bboxes = anno['bboxes'] #torch.zeros(0) # (n, 5)
    labels = anno['labels'] #torch.zeros(0) # (n,)
    result = [] # [num_cls, (n_det_cls, 5)]
    for cls_id in range(num_classes):
        ind = labels == cls_id
        cls_bboxes = np.concatenate([bboxes[ind], np.ones((len(bboxes[ind]), 1))], 1)
        result.append(cls_bboxes)
    return result

def format_results_anno(real_dataset: DOTADataset, pseudo_dataset: DOTADataset):
    pseudo_img2idx = {info['filename']:idx for idx, info in enumerate(pseudo_dataset.data_infos)}
    real_img_names = [info['filename'] for info in real_dataset.data_infos]

    pseudo_results = []
    for img_name in real_img_names:
        idx = pseudo_img2idx.get(img_name, None)
        anno = pseudo_dataset.data_infos[idx]['ann'] if idx is not None else None
        result = anno2result(anno, num_classes=len(real_dataset.get_classes()))
        pseudo_results.append(result)

    real_anno = [real_dataset.data_infos[idx]['ann'] for idx in range(len(real_dataset))]
    
    return real_anno, pseudo_results

if __name__ == '__main__':
    config_file = 'configs/_base_/datasets/dotav1.py'
    #config_file = 'configs/_base_/datasets/dotav15.py'
    #config_file = 'configs/_base_/datasets/dotav2.py'
    cfg = Config.fromfile(config_file)

    real_cfg = cfg.data.train.copy()
    real_dataset = build_dataset(real_cfg)
    print(real_dataset)

    pse_dir = 'pseudo_labels/ssp_dotav10_hybrid/'
    #pse_dir = 'pseudo_labels/ssp_dotav15_hybrid/'
    #pse_dir = 'pseudo_labels/ssp_dotav20_hybrid/'
    pse_cfg = cfg.data.train.copy()
    pse_cfg['ann_file'] = pse_dir
    pse_dataset = build_dataset(pse_cfg)
    print(pse_dataset)

    real_anno, pse_results = format_results_anno(real_dataset, pse_dataset)
    real_dataset.evaluate(pse_results, iou_thr=0.5)
    print(pse_dir)
