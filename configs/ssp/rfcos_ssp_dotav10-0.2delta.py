_base_ = [
    './rfcos_ssp_dotav10.py',
]

data = dict(
    train=dict(ann_file='pseudo_labels/ssp_dotav10_5e_0.2delta/vor_mix/',),)

